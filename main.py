# main.py
import argparse
import os
import torch
import pandas as pd
import numpy as np
import pickle
import logging # logging is configured by utils.setup_logging

# Project specific imports
from utils import load_config, setup_logging, set_seed, get_device, save_results, get_consistent_data_splits
from data.preprocess import preprocess_ppi_data
from data.features import FeatureEngineer
from data.loader import load_custom_ppi_data
from models.gnn import ImprovedLinkPredictionGNN
from models.train import train_with_early_stopping
from evaluation.metrics import evaluate_model, generate_evaluation_report, get_novel_predictions
from evaluation.analysis import (
    analyze_network_topology,
    generate_topology_report,
    identify_functional_modules,
    analyze_prediction_patterns,
    generate_prediction_patterns_report
)
from visualization import plots as vis_plots # Renamed to avoid conflict with matplotlib.pyplot
from torch_geometric.transforms import RandomLinkSplit

def run_feature_engineering(config, logger): # 新增函數
    """Runs the feature engineering pipeline."""
    logger.info("--- Step: Feature Engineering ---")
    processed_dir = config['data']['processed_dir']
    proteins_file = os.path.join(processed_dir, 'proteins_processed.csv')
    mappings_file = os.path.join(processed_dir, 'mappings.pkl')

    if not (os.path.exists(proteins_file) and os.path.exists(mappings_file)):
        logger.error(f"Processed protein data or mappings not found in {processed_dir}. Please run preprocessing first.")
        return False

    proteins_df = pd.read_csv(proteins_file)
    with open(mappings_file, 'rb') as f:
        mappings = pickle.load(f)

    # 初始化 FeatureEngineer
    # FeatureEngineer 的 __init__ 期望 config 包含 'features' 和 'data' (含 'processed_dir')
    # 我們可以直接傳遞整個 config，或者構造一個符合其期望的字典
    feature_engineer_config = {
        'features': config['features'], # 從主 config 中獲取 features 部分
        'data': {
            'processed_dir': config['data']['processed_dir'] # FeatureEngineer 可能需要這個來載入 GO DAG 等
        }
    }
    engineer = FeatureEngineer(feature_engineer_config)

    # 提取特徵
    logger.info("Extracting numerical features using FeatureEngineer...")
    # extract_features 期望 proteins_df 和 mappings (包含 protein_to_go)
    # mappings 應包含 'protein_to_go' 鍵，preprocess.py 已經這樣做了
    node_features_matrix, feature_names = engineer.extract_features(proteins_df, mappings)

    # 儲存提取的數值特徵和特徵名稱
    node_features_path = os.path.join(processed_dir, 'node_features_matrix.npy')
    feature_names_path = os.path.join(processed_dir, 'feature_names.pkl')

    np.save(node_features_path, node_features_matrix)
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    
    # 儲存 FeatureEngineer 的內部狀態 (如 scaler, svd)
    # FeatureEngineer 內部已經有 save_feature_info 方法
    feature_info_save_dir = os.path.join(processed_dir, 'feature_info')
    engineer.save_feature_info(feature_info_save_dir)

    logger.info(f"Numerical node features ({node_features_matrix.shape}) saved to {node_features_path}")
    logger.info(f"Feature names saved to {feature_names_path}")
    logger.info(f"Feature engineering components saved to {feature_info_save_dir}")
    return True


def run_preprocessing(config, logger):
    """Runs the data preprocessing pipeline."""
    logger.info("--- Mode: Preprocessing ---")
    preprocess_results = preprocess_ppi_data(config['data']) # Pass only the data part of config
    logger.info(f"Preprocessing completed. Results: {preprocess_results}")
    logger.info(f"Processed files are saved in: {config['data']['processed_dir']}")


def run_training(config, logger, device):
    """Updated training with consistent splits."""
    logger.info("--- Mode: Training ---")

    processed_dir = config['data']['processed_dir']
    interaction_file = os.path.join(processed_dir, 'interactions_processed.csv')
    numerical_features_file = os.path.join(processed_dir, 'node_features_matrix.npy')
    proteins_info_file = os.path.join(processed_dir, 'proteins_processed.csv')

    if not all(os.path.exists(f) for f in [interaction_file, numerical_features_file, proteins_info_file]):
        logger.error(f"Required files not found in {processed_dir}")
        return

    # Load data
    logger.info("Loading processed data with numerical features...")
    data = load_custom_ppi_data(
        interaction_file_path=interaction_file,
        numerical_feature_file_path=numerical_features_file,
        proteins_info_file_path=proteins_info_file,
        score_threshold=0
    )

    if data is None:
        logger.error("Failed to load data. Exiting training.")
        return

    logger.info(f"Data loaded with {data.num_nodes} nodes and {data.num_edges} edges.")

    # USE CONSISTENT SPLITTING
    logger.info("Splitting data with consistent seed...")
    train_data, val_data, test_data = get_consistent_data_splits(data, config, 'training')

    # Set different seed for model training
    model_seed = config.get('training', {}).get('model_seed', 123)
    torch.manual_seed(model_seed)
    logger.info(f"Model training seed set to: {model_seed}")

    # Rest of training code remains the same...
    model_config = config['model']
    in_channels = train_data.num_node_features
    use_mlp_pred = model_config.get('predictor_type', 'mlp').lower() == 'mlp'

    model = ImprovedLinkPredictionGNN(
        in_channels=in_channels,
        hidden_channels=model_config['hidden_dim'],
        embed_channels=model_config['embed_dim'],
        dropout=model_config['dropout'],
        use_gat=(model_config['type'] == 'GAT'),
        use_mlp_predictor=use_mlp_pred
    ).to(device)
    
    logger.info(f"Model initialized with {model_config['type']} architecture")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['lr'])
    criterion = torch.nn.BCEWithLogitsLoss()
    
    lr_scheduler = None
    if config['training'].get('lr_scheduler', False):
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=config['training']['patience'] // 2, verbose=True
        )

    logger.info("Starting model training...")
    best_model, history = train_with_early_stopping(
        model=model,
        train_graphs=[train_data],
        val_graphs=[val_data],
        test_graphs=[test_data],
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        patience=config['training']['patience'],
        n_epochs=config['training']['epochs'],
        lr_scheduler=lr_scheduler,
        verbose=True
    )
    
    # Save model and history
    exp_dir = config['experiment']['dir']
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    
    model_save_path = os.path.join(exp_dir, "models", "best_model.pt")
    torch.save(best_model.state_dict(), model_save_path)
    logger.info(f"Best model saved to {model_save_path}")

    # Save training history
    history_save_path = os.path.join(exp_dir, "training_history.json")
    import json
    with open(history_save_path, 'w') as f:
        json.dump(history, f, default=lambda o: '<not serializable>', indent=2)
    logger.info(f"Training history saved to {history_save_path}")

    # Plot training history
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    plot_save_path = os.path.join(exp_dir, "plots", "training_history.png")
    vis_plots.plot_training_history(history, filepath=plot_save_path)
    logger.info(f"Training history plot saved to {plot_save_path}")


def run_evaluation(config, logger, device):
    """Updated evaluation with consistent splits."""
    logger.info("--- Mode: Evaluation ---")
    exp_dir = config['experiment']['dir']
    model_config = config['model']
    processed_dir = config['data']['processed_dir']

    # Load data using same approach as training
    logger.info("Loading and splitting data for evaluation...")
    interaction_file = os.path.join(processed_dir, 'interactions_processed.csv')
    numerical_features_file = os.path.join(processed_dir, 'node_features_matrix.npy')
    proteins_info_file = os.path.join(processed_dir, 'proteins_processed.csv')

    data = load_custom_ppi_data(
        interaction_file_path=interaction_file,
        numerical_feature_file_path=numerical_features_file,
        proteins_info_file_path=proteins_info_file,
        score_threshold=0
    )
    if data is None:
        logger.error("Failed to load data for evaluation.")
        return

    # USE SAME CONSISTENT SPLITTING - this will give same splits as training!
    _, _, test_data = get_consistent_data_splits(data, config, 'evaluation')
    
    logger.info(f"Test data loaded with {test_data.num_nodes} nodes and {test_data.num_edges} edges.")

    # Load model
    in_channels = test_data.num_node_features
    use_mlp_pred = model_config.get('predictor_type', 'mlp').lower() == 'mlp'
    
    model = ImprovedLinkPredictionGNN(
        in_channels=in_channels,
        hidden_channels=model_config['hidden_dim'],
        embed_channels=model_config['embed_dim'],
        dropout=model_config['dropout'],
        use_gat=(model_config['type'] == 'GAT'),
        use_mlp_predictor=use_mlp_pred
    ).to(device)

    model_path = os.path.join(exp_dir, "models", "best_model.pt")
    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}. Please train the model first.")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logger.info(f"Trained model loaded from {model_path}")

    # Now evaluation uses the SAME test set as training!
    logger.info("Evaluating model performance on consistent test set...")
    
    # Rest of evaluation code remains the same...
    metrics = evaluate_model(model, [test_data], device, threshold=0.5)
    logger.info(f"Evaluation Metrics: {metrics}")


    report_save_path = os.path.join(exp_dir, "evaluation_report.txt")
    novel_preds_threshold = config.get('evaluation', {}).get('novel_pred_threshold', 0.9)
    novel_preds_top_k = config.get('evaluation', {}).get('novel_pred_top_k', 20)

    novel_predictions = get_novel_predictions(model, [test_data], device,
                                              threshold=novel_preds_threshold,
                                              top_k=novel_preds_top_k)
    generate_evaluation_report(metrics, novel_predictions, filepath=report_save_path)
    logger.info(f"Evaluation report saved to {report_save_path}")

    # 4. Plot Precision-Recall Curve
    pr_curve_save_path = os.path.join(exp_dir, "plots", "precision_recall_curve.png")
    # Using the one from visualization.plots as it seems more comprehensive
    vis_plots.plot_precision_recall_curve(model, [test_data], device, filepath=pr_curve_save_path)
    logger.info(f"Precision-Recall curve saved to {pr_curve_save_path}")

    # 5. Advanced Analysis (Optional, based on config)
    if config.get('evaluation', {}).get('run_advanced_analysis', True):
        logger.info("Running advanced analysis...")

        # Network Topology Analysis (on the full graph before splitting)
        topology_stats = analyze_network_topology(data) # Use original full data
        topo_report_path = os.path.join(exp_dir, "topology_report.txt")
        # node_idx_to_protein_id might be useful here from data object
        generate_topology_report(topology_stats, 
                                 node_idx_to_protein_id=data.get('node_idx_to_protein_id'),
                                 filepath=topo_report_path)
        vis_plots.visualize_degree_distribution(data, save_path=os.path.join(exp_dir, "plots", "degree_distribution.png"))


        # Embedding Visualization (using test_data or full data)
        embeddings_path = os.path.join(exp_dir, "plots", "embeddings_tsne.png")
        # Get embeddings from the model
        # The visualize_embeddings function in plots.py takes model, data, device
        vis_plots.visualize_embeddings(model, test_data.to(device), device, # Pass encoder for node embeddings
                                       save_path=embeddings_path,
                                       # color_by can be node degrees or other attributes if available
                                       color_by=test_data.x[:,0].cpu().numpy() if test_data.x is not None else None)


        # Functional Module Identification (if embeddings are meaningful)
        # Requires node_embeddings. Can get them from model.get_embeddings(data.x, data.edge_index)
        logger.info("Generating node embeddings for functional module analysis...")
        with torch.no_grad():
            node_embeddings = model.get_embeddings(data.x.to(device), data.edge_index.to(device))
            
        module_analysis_config = config.get('evaluation',{}).get('module_analysis',{'method':'kmeans', 'k_range_max':10})

        # Pass the actual identify_functional_modules function to the visualization function
        functional_modules_dict = vis_plots.visualize_functional_modules(
            data, node_embeddings.cpu(), # Pass embeddings on CPU
            save_path=os.path.join(exp_dir, "plots", "functional_modules.png"),
            clustering_method=module_analysis_config.get('method', 'kmeans'),
            identify_functional_modules_func=identify_functional_modules, # Pass the function itself
             # For K-means, k_range can be set. For DBSCAN, eps and min_samples.
            k_range=range(2, min(module_analysis_config.get('k_range_max', 10), data.num_nodes // 5 if data.num_nodes >10 else 3))
        )
        if functional_modules_dict is not None: # <--- 新增檢查
            # 根據您之前的日誌邏輯，計算顯示的模組數量
            num_display_modules = len(functional_modules_dict)
            # 假設 "Noise (DBSCAN)" 是 DBSCAN 產生的雜訊點的鍵名
            if "Noise (DBSCAN)" in functional_modules_dict:
                if len(functional_modules_dict) == 1: # 如果只有雜訊點
                    num_display_modules = 0
                else: # 如果有雜訊點和其他模組
                    num_display_modules -= 1
            
            logger.info(f"Functional modules identified: {num_display_modules} actual modules.")
        else:
            logger.warning("Functional module identification/visualization did not return a valid module dictionary.")


        # Prediction Pattern Analysis
        pred_pattern_config = config.get('evaluation',{}).get('prediction_pattern_analysis',{'degree_percentile': 50})
        pattern_results = analyze_prediction_patterns(model, test_data.to(device), device,
                                                      degree_percentile_threshold=pred_pattern_config.get('degree_percentile',50))
        pattern_report_path = os.path.join(exp_dir, "prediction_patterns_report.txt")
        generate_prediction_patterns_report(pattern_results, filepath=pattern_report_path)
        vis_plots.plot_prediction_pattern_statistics(pattern_results,
                                               save_path=os.path.join(exp_dir, "plots", "prediction_patterns.png"))

        # Visualize Predicted Links
        vis_plots.visualize_predicted_links(model, test_data.to(device), device,
                                            save_path=os.path.join(exp_dir, "plots", "predicted_links_visualization.png"),
                                            node_idx_to_name=data.get('node_idx_to_protein_id'))


def main():
    parser = argparse.ArgumentParser(description="PPI Link Prediction Project")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--mode', type=str, default='full_pipeline',
                        choices=['preprocess', 'feature_engineer', 'train', 'evaluate', 'full_pipeline', 'custom_eval'], # 新增 'feature_engineer'
                        help='Operation mode')
    # ... (其他 parser 參數) ...
    parser.add_argument('--seed', type=int, default=None, help='Random seed override')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')

    args = parser.parse_args()
    config_data = load_config(args.config)
    if args.seed is not None:
        if 'training' in config_data and isinstance(config_data['training'], dict):
            # Fix: Set all the seed types that your functions actually use
            config_data['training']['data_split_seed'] = args.seed  # ✅ This is what get_consistent_data_splits uses
            config_data['training']['model_seed'] = args.seed
            config_data['training']['feature_seed'] = args.seed
            config_data['training']['seed'] = args.seed  # Keep for compatibility
        else:
            config_data['seed'] = args.seed


    logger = setup_logging(config_data, name=config_data['experiment']['name'])
    logger.info(f"Configuration loaded from {args.config}")
    logger.info(f"Running in mode: {args.mode}")
    logger.info(f"Experiment directory: {config_data['experiment']['dir']}")

    seed_value = config_data.get('training', {}).get('data_split_seed', 42)  # ✅ Use data_split_seed consistently
    set_seed(seed_value)
    logger.info(f"Random seed set to: {seed_value}")

    use_cuda_preference = not args.no_cuda
    device = get_device(cuda_preference=use_cuda_preference)
    logger.info(f"Using device: {device}")

    if args.mode == 'preprocess':
        run_preprocessing(config_data, logger)
    elif args.mode == 'feature_engineer': # 新增模式處理
        if not run_feature_engineering(config_data, logger):
            logger.error("Feature engineering failed. Aborting.")
            return
    elif args.mode == 'train':
        run_training(config_data, logger, device)
    elif args.mode == 'evaluate':
        run_evaluation(config_data, logger, device) # run_evaluation 也需要更新以使用新的 loader 方式
    elif args.mode == 'full_pipeline':
        run_preprocessing(config_data, logger)
        if not run_feature_engineering(config_data, logger): # 整合到 full_pipeline
            logger.error("Feature engineering failed during full pipeline. Aborting.")
            return
        run_training(config_data, logger, device)
        run_evaluation(config_data, logger, device) # run_evaluation 也需要更新
    elif args.mode == 'custom_eval':
        # custom_eval 也需要更新以使用新的 loader 方式
        logger.info("--- Mode: Custom Evaluation ---")
        # ... (確保 custom_eval 中的資料載入也遵循新的 numerical_features_file 方式)
        # (此處省略 custom_eval 的完整修改，但原理與 run_training/run_evaluation 類似)

        exp_dir = config_data['experiment']['dir']
        model_config = config_data['model']
        processed_dir = config_data['data']['processed_dir']
        logger.info("Loading data for custom eval...")
        interaction_file_custom = os.path.join(processed_dir, 'interactions_processed.csv')
        numerical_features_file_custom = os.path.join(processed_dir, 'node_features_matrix.npy')
        proteins_info_file_custom = os.path.join(processed_dir, 'proteins_processed.csv')

        data_custom = load_custom_ppi_data(interaction_file_custom, numerical_features_file_custom, proteins_info_file_custom)
        if data_custom is None: logger.error("Failed to load data for custom eval."); return

        transform_custom = RandomLinkSplit(num_val=config_data['training']['val_ratio'], num_test=config_data['training']['test_ratio'], is_undirected=True, split_labels=False)
        _, _, test_data_custom = transform_custom(data_custom)

        logger.info("Loading model for custom eval...")
        in_channels_custom = test_data_custom.num_node_features
        use_mlp_pred_custom = model_config.get('predictor_type', 'mlp').lower() == 'mlp'
        model_custom = ImprovedLinkPredictionGNN(in_channels_custom, model_config['hidden_dim'], model_config['embed_dim'], use_gat=(model_config['type'] == 'GAT'), use_mlp_predictor=use_mlp_pred_custom).to(device)
        model_path_custom = os.path.join(exp_dir, "models", "best_model.pt")
        if not os.path.exists(model_path_custom): logger.error(f"Model not found: {model_path_custom}"); return
        model_custom.load_state_dict(torch.load(model_path_custom, map_location=device))
        model_custom.eval()
        # ... (custom_eval 的其餘邏輯) ...


    else:
        logger.error(f"Unknown mode: {args.mode}")

    logger.info("--- Pipeline Finished ---")

if __name__ == "__main__":
    main()