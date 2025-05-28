# data/loader.py
import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
import logging
import os # 新增

# Configure basic logging
logging.getLogger(__name__)

def load_custom_ppi_data(interaction_file_path,
                         numerical_feature_file_path, # 修改：接收 .npy 特徵檔案
                         proteins_info_file_path,     # 新增：接收包含 protein_id 順序的原始 CSV
                         score_threshold=0):
    """
    Loads custom PPI data and pre-computed numerical node features.

    Args:
        interaction_file_path (str): Path to the CSV file containing protein interactions.
                                     Expected columns: protein1_id, protein2_id, combined_score.
        numerical_feature_file_path (str): Path to the .npy file containing the numerical node feature matrix.
                                           Rows should correspond to proteins in proteins_info_file_path.
        proteins_info_file_path (str): Path to the CSV file containing protein information,
                                       primarily used to get the order and full list of protein_ids.
                                       Expected column: protein_id.
        score_threshold (int): Minimum combined_score to consider an interaction.

    Returns:
        torch_geometric.data.Data: A Data object.
        None: If loading fails.
    """
    logging.info(f"Loading interaction data from: {interaction_file_path}")
    try:
        interactions_df = pd.read_csv(interaction_file_path)
        required_interaction_cols = ['protein1_id', 'protein2_id', 'combined_score']
        if not all(col in interactions_df.columns for col in required_interaction_cols):
            logging.error(f"Interaction file missing required columns. Expected: {required_interaction_cols}, Got: {interactions_df.columns.tolist()}")
            return None
        logging.info(f"Successfully loaded interaction data. Shape: {interactions_df.shape}")
    except Exception as e:
        logging.error(f"Error loading interaction file {interaction_file_path}: {e}")
        return None

    logging.info(f"Loading numerical node features from: {numerical_feature_file_path}")
    try:
        x_numpy = np.load(numerical_feature_file_path)
        x = torch.tensor(x_numpy, dtype=torch.float)
        logging.info(f"Successfully loaded numerical features. Shape: {x.shape}")
    except Exception as e:
        logging.error(f"Error loading numerical feature file {numerical_feature_file_path}: {e}")
        return None

    logging.info(f"Loading protein information for ID mapping from: {proteins_info_file_path}")
    try:
        # 這個檔案主要用來確定 protein_id 的順序和完整列表，
        # 這個順序必須和 node_features_matrix.npy 的行順序一致。
        # FeatureEngineer 在 extract_features 時會基於 proteins_df 的順序，
        # 而 proteins_df 是從 proteins_processed.csv 載入的。
        protein_order_df = pd.read_csv(proteins_info_file_path)
        if 'protein_id' not in protein_order_df.columns:
            logging.error(f"Proteins info file {proteins_info_file_path} missing 'protein_id' column.")
            return None
        # 確保 protein_id 是唯一的，否則對映會有問題
        if not protein_order_df['protein_id'].is_unique:
            logging.warning(f"Protein IDs in {proteins_info_file_path} are not unique. Using the first occurrence for mapping.")
            # 可以選擇去重，但更好的做法是確保 preprocess 階段的 protein_id 是唯一的
            protein_order_df = protein_order_df.drop_duplicates(subset=['protein_id'], keep='first')

        all_protein_ids_ordered = protein_order_df['protein_id'].tolist()
        logging.info(f"Loaded protein ID order for {len(all_protein_ids_ordered)} proteins.")

        if x.shape[0] != len(all_protein_ids_ordered):
            logging.error(f"Mismatch between number of proteins in feature matrix ({x.shape[0]}) "
                          f"and protein info file ({len(all_protein_ids_ordered)}).")
            return None

    except Exception as e:
        logging.error(f"Error loading protein info file {proteins_info_file_path}: {e}")
        return None


    # Apply score threshold to interactions
    if 'combined_score' in interactions_df.columns and score_threshold > 0:
        original_interaction_count = len(interactions_df)
        interactions_df = interactions_df[interactions_df['combined_score'] >= score_threshold]
        logging.info(f"Filtered interactions by score >= {score_threshold}. Kept {len(interactions_df)} out of {original_interaction_count} interactions.")
        if interactions_df.empty:
            logging.warning("No interactions left after applying score threshold.")

    # Create mapping from original protein_id (from the ordered list) to new 0-indexed node_idx
    # 這個 protein_id 的順序來自 proteins_info_file_path，它應該與 features_matrix.npy 的行順序一致
    protein_id_to_node_idx = {protein_id: i for i, protein_id in enumerate(all_protein_ids_ordered)}
    node_idx_to_protein_id = {i: protein_id for protein_id, i in protein_id_to_node_idx.items()}
    num_nodes = len(all_protein_ids_ordered)
    logging.info(f"Total unique proteins (nodes) based on ordered list: {num_nodes}")


    # Prepare edge_index and optional edge_attr
    source_nodes = []
    target_nodes = []
    edge_attributes = []

    # 過濾掉交互作用中不包含在 all_protein_ids_ordered 中的蛋白質
    # (理論上，preprocess.py 產生的 interactions_processed.csv 中的 protein_id 應該都在 proteins_processed.csv 中)
    valid_interactions_df = interactions_df[
        interactions_df['protein1_id'].isin(protein_id_to_node_idx) &
        interactions_df['protein2_id'].isin(protein_id_to_node_idx)
    ].copy()

    if len(valid_interactions_df) < len(interactions_df):
        logging.warning(f"Filtered out {len(interactions_df) - len(valid_interactions_df)} interactions "
                        "due to proteins not found in the main protein ID list.")


    for _, row in valid_interactions_df.iterrows():
        p1_id = row['protein1_id']
        p2_id = row['protein2_id']
        score = row['combined_score']

        # p1_id 和 p2_id 應該已經在 protein_id_to_node_idx 中了，因為上面過濾了
        u = protein_id_to_node_idx[p1_id]
        v = protein_id_to_node_idx[p2_id]

        source_nodes.extend([u, v])
        target_nodes.extend([v, u])
        edge_attributes.extend([score, score])

    if not source_nodes:
        logging.warning("No edges were created. The graph will be empty of edges.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float) if edge_attributes else None
    else:
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        if edge_attributes:
            edge_attr = torch.tensor(edge_attributes, dtype=torch.float).unsqueeze(1)
        else:
            edge_attr = None
        logging.info(f"Number of edges (bi-directional): {edge_index.shape[1]}")

    data = Data(x=x, edge_index=edge_index)
    if edge_attr is not None:
        data.edge_attr = edge_attr
    
    data.protein_id_to_node_idx = protein_id_to_node_idx
    data.node_idx_to_protein_id = node_idx_to_protein_id
    data.num_nodes = num_nodes

    logging.info("Custom PPI data loaded and processed successfully using pre-computed numerical features.")
    logging.info(f"Data object created - Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}, Node Features: {data.x.shape[1]}")
    
    return data


