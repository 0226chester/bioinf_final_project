import torch
import torch.nn.functional as F
from torch.optim import Adam
# from torch_geometric.datasets import PPI # For fallback
import torch_geometric.transforms as T
from torch_geometric.data import Data # To check instance type
import os
import time
import argparse
import matplotlib.pyplot as plt
import numpy as np
import logging
import networkx as nx

# Import our custom modules
from models import ImprovedLinkPredictionGNN
from train import train_with_early_stopping, plot_training_history
from data_loader import load_custom_ppi_data

# Import functions from your uploaded utility scripts
from evaluate import evaluate_model, plot_precision_recall_curve, get_novel_predictions, generate_evaluation_report
from visualize import (visualize_graph, visualize_embeddings, visualize_predicted_links,
                      visualize_degree_distribution)
# Assuming bio_validation.py has functions compatible with the enhanced version discussed
from bio_validation import (analyze_network_topology, generate_topology_report,
                          visualize_functional_modules, analyze_prediction_patterns,
                          generate_prediction_patterns_report, plot_prediction_pattern_statistics, identify_functional_modules)

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='PPI Link Prediction with Custom Data Support')
    
    # Model parameters
    parser.add_argument('--hidden_channels', type=int, default=128, help='Number of hidden channels')
    parser.add_argument('--embed_channels', type=int, default=64, help='Number of embedding channels')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout rate')
    parser.add_argument('--use_gat', action='store_true', help='Use GAT instead of GraphSAGE')
    parser.add_argument('--use_mlp_predictor', action='store_true', help='Use MLP predictor instead of dot product')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Data parameters
    parser.add_argument('--interaction_file', type=str, default='./data/PPI/raw/filtered_interactions_with_protein_id_clean.csv', help='Path to custom interaction CSV file')
    parser.add_argument('--feature_file', type=str, default='./data/PPI/processed/engineered_features_hierarchical_with_ids.csv', help='Path to custom feature CSV file')
    parser.add_argument('--score_threshold', type=int, default=0, help='Minimum combined_score for interactions in custom data.')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='Ratio of validation links for custom data split.')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Ratio of test links for custom data split.')


    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output_dir', type=str, default='results', help='Output directory')
    parser.add_argument('--no_cuda', action='store_true', help='Disable CUDA')
    parser.add_argument('--visualize', action='store_true', help='Generate visualizations')
    
    return parser.parse_args()


def load_and_preprocess_data(args):
    """Load and preprocess data, either custom or default PPI."""
    logging.info("Loading and preprocessing data...")
    node_idx_to_id_mapping = None # Initialize mapping

    if args.interaction_file and args.feature_file:
        logging.info("Attempting to load custom PPI data.")
        # single_graph is the original, unsplit graph from your custom data
        single_graph = load_custom_ppi_data(args.interaction_file, args.feature_file, args.score_threshold)
        
        if single_graph is None or single_graph.num_nodes == 0 or single_graph.x is None:
            logging.error("Failed to load custom data or data is invalid. Exiting.")
            raise ValueError("Custom data loading failed.")

        # Capture the mapping from the loaded custom graph
        node_idx_to_id_mapping = getattr(single_graph, 'node_idx_to_protein_id', None)
        if node_idx_to_id_mapping:
            logging.info("Successfully captured node_idx_to_protein_id mapping from custom data.")
        else:
            logging.warning("Could not find 'node_idx_to_protein_id' attribute on the loaded custom graph.")


        if single_graph.edge_index is None or single_graph.edge_index.shape[1] == 0:
            logging.warning("Custom graph has no edges. Link prediction might not be meaningful.")

        min_edges_for_split = 10 
        if single_graph.edge_index.shape[1] < min_edges_for_split and (args.val_ratio > 0 or args.test_ratio > 0):
             logging.warning(f"Very few edges ({single_graph.edge_index.shape[1]}) in the custom graph. Splitting might be unstable or fail.")
             if single_graph.edge_index.shape[1] == 0:
                 logging.error("Cannot perform link split on a graph with no edges.")
                 raise ValueError("Cannot split a graph with no edges for link prediction.")

        transform = T.RandomLinkSplit(
            num_val=args.val_ratio,
            num_test=args.test_ratio,
            is_undirected=True,
            add_negative_train_samples=True, 
            split_labels=False 
        )
        
        try:
            train_data, val_data, test_data = transform(single_graph.clone()) 
        except Exception as e:
            logging.error(f"Error during RandomLinkSplit on custom data: {e}")
            raise

        train_graphs_lp = [train_data]
        val_graphs_lp = [val_data]
        test_graphs_lp = [test_data]
        
        num_node_features = single_graph.num_features
        logging.info(f"Loaded custom data: 1 graph split into train/val/test for link prediction.")
        # (Logging for link shapes can remain the same)

    else: # Fallback to default PPI dataset
        logging.info("Custom data files not provided. Loading default PPI dataset.")
        # node_idx_to_id_mapping will remain None for default PPI data unless you have a way to generate/load it.
        transform = T.RandomLinkSplit(
            num_val=0.1, 
            num_test=0.15,
            is_undirected=True,
            add_negative_train_samples=True,
            split_labels=False 
        )
        
        train_dataset_original = PPI(root='./data/PPI', split='train')
        val_dataset_original = PPI(root='./data/PPI', split='val')
        test_dataset_original = PPI(root='./data/PPI', split='test')
        
        num_node_features = train_dataset_original.num_features
        
        train_graphs_lp = [transform(graph.clone())[0] for graph in train_dataset_original]
        val_graphs_lp = [transform(graph.clone())[1] for graph in val_dataset_original]
        test_graphs_lp = [transform(graph.clone())[2] for graph in test_dataset_original]
        # (Logging for default PPI dataset can remain the same)

    logging.info(f"Node features dimension: {num_node_features}")
    # Return the mapping along with other data
    return train_graphs_lp, val_graphs_lp, test_graphs_lp, num_node_features, node_idx_to_id_mapping


def create_output_dirs(args):
    """Create output directories."""
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'plots'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'reports'), exist_ok=True)


def main(args):
    """Main function."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and not args.no_cuda:
        torch.cuda.manual_seed_all(args.seed) 
    
    create_output_dirs(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    logging.info(f"Using device: {device}")
    
    try:
        # Unpack the new mapping variable
        train_graphs, val_graphs, test_graphs, num_node_features, node_idx_to_name_map = load_and_preprocess_data(args)
    except ValueError as e:
        logging.error(f"Failed to load or preprocess data: {e}")
        return 

    # ... (rest of the data validation and moving to device remains the same) ...
    if not train_graphs or not val_graphs or not test_graphs:
        logging.error("Data loading resulted in empty graph lists. Cannot proceed.")
        return

    if not all(isinstance(g, Data) for g_list in [train_graphs, val_graphs, test_graphs] for g in g_list):
        logging.error("Some graph objects are not of type torch_geometric.data.Data after loading. Cannot proceed.")
        return

    try:
        train_graphs = [graph.to(device) for graph in train_graphs]
        val_graphs = [graph.to(device) for graph in val_graphs]
        test_graphs = [graph.to(device) for graph in test_graphs]
    except Exception as e:
        logging.error(f"Error moving data to device {device}: {e}")
        return

    logging.info("Initializing model...")
    if num_node_features <= 0:
        logging.error(f"Invalid number of node features: {num_node_features}. Cannot initialize model.")
        return

    model = ImprovedLinkPredictionGNN(
        in_channels=num_node_features,
        hidden_channels=args.hidden_channels,
        embed_channels=args.embed_channels,
        dropout=args.dropout,
        use_gat=args.use_gat,
        use_mlp_predictor=args.use_mlp_predictor
    ).to(device)
    
    # ... (model logging, optimizer, scheduler, criterion, training loop remain the same) ...
    logging.info(model)
    
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=max(1, args.patience // 2),
    )
    criterion = torch.nn.BCEWithLogitsLoss() 
    
    logging.info("\nTraining model...")
    start_time = time.time()
    
    model, history = train_with_early_stopping(
        model, train_graphs, val_graphs, test_graphs,
        optimizer, criterion, device,
        patience=args.patience, n_epochs=args.epochs,
        lr_scheduler=scheduler, verbose=True
    )
    
    training_time = time.time() - start_time
    logging.info(f"Training completed in {training_time:.2f} seconds")
    
    model_path = os.path.join(args.output_dir, 'models', 'ppi_link_prediction_model.pt')
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': vars(args),
        'num_node_features': num_node_features,
        # Optionally save the mapping if it exists (useful if loading model elsewhere)
        'node_idx_to_name_map': node_idx_to_name_map 
    }, model_path)
    logging.info(f"Model saved to {model_path}")
    
    if args.visualize and history.get('train_loss'): 
        history_plot_path = os.path.join(args.output_dir, 'plots', 'training_history.png')
        plot_training_history(history, filepath=history_plot_path)
    
    logging.info("\nEvaluating model on test set...")
    test_metrics = evaluate_model(model, test_graphs, device) 
    logging.info(f"Test Metrics: {test_metrics}")
    
    if test_graphs and all(isinstance(g, Data) for g in test_graphs):
        # Pass node_idx_to_name_map to functions that might use it in their reports (get_novel_predictions itself doesn't, but generate_evaluation_report might be enhanced)
        novel_predictions = get_novel_predictions(model, test_graphs, device, threshold=0.9) 
        report_path = os.path.join(args.output_dir, 'reports', 'evaluation_report.txt')
        # generate_evaluation_report could be enhanced to use node_idx_to_name_map for novel predictions
        generate_evaluation_report(test_metrics, novel_predictions, filepath=report_path) # node_idx_to_name_map could be passed here if the function supports it
    else:
        logging.warning("Skipping novel predictions and report generation due to empty or invalid test_graphs.")

    if args.visualize and test_graphs and all(isinstance(g, Data) for g in test_graphs):
        pr_curve_path = os.path.join(args.output_dir, 'plots', 'precision_recall_curve.png')
        plot_precision_recall_curve(model, test_graphs, device, filepath=pr_curve_path)
    
    if test_graphs and isinstance(test_graphs[0], Data):
        sample_graph_for_viz = test_graphs[0].cpu() 

        model.eval() 
        node_embeddings = None
        # ... (node embedding calculation remains the same) ...
        with torch.no_grad():
            if hasattr(sample_graph_for_viz, 'x') and sample_graph_for_viz.x is not None and \
               hasattr(sample_graph_for_viz, 'edge_index') and sample_graph_for_viz.edge_index is not None:
                node_embeddings = model.get_embeddings(sample_graph_for_viz.x.to(device), 
                                                       sample_graph_for_viz.edge_index.to(device)).cpu()
            else:
                logging.warning("Sample graph for visualization is missing 'x' or 'edge_index'. Skipping embedding-dependent visualizations.")
        
        logging.info("\nPerforming biological validation and analysis...")
        
        topology_stats = analyze_network_topology(sample_graph_for_viz) # Assumes analyze_network_topology doesn't need the map directly
        topology_report_path = os.path.join(args.output_dir, 'reports', 'topology_report.txt')
        # Pass the mapping to generate_topology_report
        generate_topology_report(topology_stats, node_idx_to_protein_id=node_idx_to_name_map, filepath=topology_report_path)
        
        prediction_patterns = analyze_prediction_patterns(model, test_graphs[0].to(device), device) 
        patterns_report_path = os.path.join(args.output_dir, 'reports', 'prediction_patterns_report.txt')
        # generate_prediction_patterns_report might also be enhanced to use the map
        generate_prediction_patterns_report(prediction_patterns, filepath=patterns_report_path)
        
        if args.visualize:
            logging.info("\nGenerating visualizations...")
            
            network_path = os.path.join(args.output_dir, 'plots', 'ppi_network.png')
            # Pass mapping to visualize_graph
            degrees_for_sizing = None
            if sample_graph_for_viz.edge_index is not None and sample_graph_for_viz.num_nodes > 0:
                 temp_g = nx.Graph()
                 temp_g.add_nodes_from(range(sample_graph_for_viz.num_nodes))
                 temp_g.add_edges_from(sample_graph_for_viz.edge_index.T.tolist())
                 degrees_for_sizing = np.array([d for _, d in temp_g.degree()])
            
            visualize_graph(sample_graph_for_viz, 
                            node_idx_to_name=node_idx_to_name_map, 
                            label_top_n_nodes=5, # Example: label top 5 degree nodes
                            node_size_attr=degrees_for_sizing,
                            save_path=network_path) 
            
            if node_embeddings is not None:
                # For coloring embeddings by module, you'd first identify modules
                module_assignments_for_coloring = None
                modules_dict_for_color, _ = identify_functional_modules(sample_graph_for_viz, node_embeddings, method='kmeans')
                if modules_dict_for_color:
                    module_assignments_for_coloring = np.zeros(sample_graph_for_viz.num_nodes, dtype=int)
                    for mod_idx, (mod_name, nodes) in enumerate(modules_dict_for_color.items()):
                        if "Noise" not in mod_name:
                             for node_idx_in_mod in nodes: # Renamed to avoid conflict
                                module_assignments_for_coloring[node_idx_in_mod] = mod_idx # Use mod_idx for coloring
                        else:
                             for node_idx_in_mod in nodes:
                                module_assignments_for_coloring[node_idx_in_mod] = -1 


                embeddings_path = os.path.join(args.output_dir, 'plots', 'protein_embeddings.png')
                visualize_embeddings(model, test_graphs[0].to(device), device, 
                                     color_by=module_assignments_for_coloring,
                                     cmap_name='tab20', # Good for categorical data
                                     save_path=embeddings_path)
            
            predicted_links_path = os.path.join(args.output_dir, 'plots', 'predicted_links.png')
            # Pass mapping to visualize_predicted_links
            visualize_predicted_links(model, test_graphs[0].to(device), device, 
                                      node_idx_to_name=node_idx_to_name_map,
                                      label_top_n_predictions=5, # Example
                                      save_path=predicted_links_path)
            
            degree_dist_path = os.path.join(args.output_dir, 'plots', 'degree_distribution.png')
            visualize_degree_distribution(sample_graph_for_viz, save_path=degree_dist_path) 
            
            if node_embeddings is not None:
                modules_path = os.path.join(args.output_dir, 'plots', 'functional_modules.png')
                # visualize_functional_modules can use the mapping internally if enhanced, or you can pass module data
                visualize_functional_modules(sample_graph_for_viz, node_embeddings, save_path=modules_path, clustering_method='kmeans') 
            
            if prediction_patterns: 
                patterns_plot_path = os.path.join(args.output_dir, 'plots', 'prediction_patterns.png')
                plot_prediction_pattern_statistics(prediction_patterns, save_path=patterns_plot_path)
    else:
        logging.warning("No valid test graph available for detailed visualization and analysis.")

    logging.info("\nPPI link prediction pipeline completed!")
    logging.info(f"Results saved in {args.output_dir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)