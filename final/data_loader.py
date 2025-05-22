import pandas as pd
import torch
from torch_geometric.data import Data
import numpy as np
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_custom_ppi_data(interaction_file_path, feature_file_path, score_threshold=0):
    """
    Loads custom PPI data from CSV files and converts it into a PyTorch Geometric Data object.

    Args:
        interaction_file_path (str): Path to the CSV file containing protein interactions.
                                     Expected columns: protein1_id, protein2_id, combined_score (and others like protein1, protein2).
        feature_file_path (str): Path to the CSV file containing protein features.
                                 Expected columns: protein_id, [feature_name_1], [feature_name_2], ...
        score_threshold (int): Minimum combined_score to consider an interaction. Default 0 (all interactions).

    Returns:
        torch_geometric.data.Data: A Data object representing the graph, with attributes:
                                   x: Node feature matrix [num_nodes, num_features].
                                   edge_index: Graph connectivity [2, num_edges].
                                   edge_attr: Edge features (combined_score) [num_edges, 1] (optional, if scores are used).
                                   protein_id_to_node_idx (dict): Mapping from original protein_id to new 0-indexed node_idx.
                                   node_idx_to_protein_id (dict): Mapping from new 0-indexed node_idx to original protein_id.
                                   num_nodes (int): Total number of unique proteins.
        None: If loading fails or data is invalid.
    """
    logging.info(f"Loading interaction data from: {interaction_file_path}")
    try:
        interactions_df = pd.read_csv(interaction_file_path)
        # Assuming columns are named: protein1, protein2, protein1_id, protein2_id, combined_score
        # Validate required columns for interactions
        required_interaction_cols = ['protein1_id', 'protein2_id', 'combined_score']
        if not all(col in interactions_df.columns for col in required_interaction_cols):
            logging.error(f"Interaction file missing required columns. Expected: {required_interaction_cols}, Got: {interactions_df.columns.tolist()}")
            return None
        logging.info(f"Successfully loaded interaction data. Shape: {interactions_df.shape}")
    except FileNotFoundError:
        logging.error(f"Interaction file not found at {interaction_file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"Interaction file is empty: {interaction_file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading interaction file: {e}")
        return None

    logging.info(f"Loading feature data from: {feature_file_path}")
    try:
        features_df = pd.read_csv(feature_file_path)
        # Assuming columns are named: protein_id, [feature_columns...]
        # Validate required columns for features
        if 'protein_id' not in features_df.columns:
            logging.error(f"Feature file missing 'protein_id' column. Got: {features_df.columns.tolist()}")
            return None
        if len(features_df.columns) < 2:
            logging.error(f"Feature file must have 'protein_id' and at least one feature column.")
            return None
        logging.info(f"Successfully loaded feature data. Shape: {features_df.shape}")
    except FileNotFoundError:
        logging.error(f"Feature file not found at {feature_file_path}")
        return None
    except pd.errors.EmptyDataError:
        logging.error(f"Feature file is empty: {feature_file_path}")
        return None
    except Exception as e:
        logging.error(f"Error loading feature file: {e}")
        return None

    # Apply score threshold
    if 'combined_score' in interactions_df.columns and score_threshold > 0:
        original_interaction_count = len(interactions_df)
        interactions_df = interactions_df[interactions_df['combined_score'] >= score_threshold]
        logging.info(f"Filtered interactions by score >= {score_threshold}. Kept {len(interactions_df)} out of {original_interaction_count} interactions.")
        if interactions_df.empty:
            logging.warning("No interactions left after applying score threshold.")
            # Decide if to return None or an empty graph object
            # For now, let's proceed and it will likely result in an empty graph
            # which should be handled by downstream processes.

    # Create a mapping from original protein IDs to new 0-indexed contiguous node IDs
    # Consider all unique protein IDs from both interactions (protein1_id, protein2_id)
    # and features (protein_id) to ensure all featured nodes are included even if isolated,
    # and all interacting nodes are included even if they lack features.
    
    # Extract unique protein IDs from interactions
    interaction_protein_ids = pd.concat([interactions_df['protein1_id'], interactions_df['protein2_id']]).unique()
    # Extract unique protein IDs from features
    feature_protein_ids = features_df['protein_id'].unique()
    
    # Combine all unique protein IDs
    all_protein_ids = np.union1d(interaction_protein_ids, feature_protein_ids)
    
    if len(all_protein_ids) == 0:
        logging.warning("No protein IDs found in interaction or feature files. Cannot build graph.")
        return None

    protein_id_to_node_idx = {protein_id: i for i, protein_id in enumerate(all_protein_ids)}
    node_idx_to_protein_id = {i: protein_id for protein_id, i in protein_id_to_node_idx.items()}
    num_nodes = len(all_protein_ids)
    logging.info(f"Total unique proteins (nodes): {num_nodes}")

    # Prepare node features
    # Initialize features with zeros (or a more sophisticated imputation if needed)
    # The number of feature columns is all columns in features_df except 'protein_id' and 'protein_name' (if it exists)
    feature_names = [col for col in features_df.columns if col not in ['protein_id', 'protein_name']]
    if not feature_names:
        logging.error("No feature columns found in the feature file (excluding 'protein_id' and 'protein_name').")
        return None
    num_features = len(feature_names)
    logging.info(f"Number of node features: {num_features}")
    
    x = torch.zeros((num_nodes, num_features), dtype=torch.float)
    
    # Fill features for nodes present in the feature file
    feature_nodes_found = 0
    for _, row in features_df.iterrows():
        protein_id = row['protein_id']
        if protein_id in protein_id_to_node_idx:
            node_idx = protein_id_to_node_idx[protein_id]
            try:
                # Ensure all feature values are numeric and handle potential errors
                feature_values = [pd.to_numeric(row[name], errors='coerce') for name in feature_names]
                if any(pd.isna(val) for val in feature_values):
                    logging.warning(f"NaN found in features for protein_id {protein_id}. Using 0 for these features.")
                    feature_values = [0 if pd.isna(val) else val for val in feature_values]
                x[node_idx] = torch.tensor(feature_values, dtype=torch.float)
                feature_nodes_found += 1
            except ValueError as e:
                logging.warning(f"Could not convert features to numeric for protein_id {protein_id}. Error: {e}. Using zeros.")
            except Exception as e: # Catch any other unexpected error during feature processing for a row
                logging.warning(f"Unexpected error processing features for protein_id {protein_id}. Error: {e}. Using zeros.")


    logging.info(f"Populated features for {feature_nodes_found}/{num_nodes} nodes.")
    if feature_nodes_found < num_nodes:
        logging.warning(f"{num_nodes - feature_nodes_found} nodes do not have features in the feature file and will have zero vectors.")


    # Prepare edge_index and optional edge_attr
    source_nodes = []
    target_nodes = []
    edge_attributes = []

    for _, row in interactions_df.iterrows():
        p1_id = row['protein1_id']
        p2_id = row['protein2_id']
        score = row['combined_score']

        # Ensure both proteins are in our mapping (they should be if all_protein_ids was constructed correctly)
        if p1_id in protein_id_to_node_idx and p2_id in protein_id_to_node_idx:
            u = protein_id_to_node_idx[p1_id]
            v = protein_id_to_node_idx[p2_id]

            # Add edges in both directions for an undirected graph
            source_nodes.extend([u, v])
            target_nodes.extend([v, u])
            
            # Add edge attributes (e.g., combined_score)
            # If using edge_attr, ensure it's consistent for both directions or average/process as needed
            edge_attributes.extend([score, score]) 
        else:
            logging.warning(f"Skipping interaction ({p1_id}, {p2_id}) as one or both proteins not in the consolidated ID list.")


    if not source_nodes: # No valid edges found
        logging.warning("No edges were created. The graph will be empty of edges.")
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 1), dtype=torch.float) if edge_attributes else None
    else:
        edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        if edge_attributes:
            edge_attr = torch.tensor(edge_attributes, dtype=torch.float).unsqueeze(1) # Shape: [num_edges, 1]
        else:
            edge_attr = None
        logging.info(f"Number of edges (bi-directional): {edge_index.shape[1]}")


    # Create PyTorch Geometric Data object
    data = Data(x=x, edge_index=edge_index)
    if edge_attr is not None:
        data.edge_attr = edge_attr
    
    data.protein_id_to_node_idx = protein_id_to_node_idx
    data.node_idx_to_protein_id = node_idx_to_protein_id
    data.num_nodes = num_nodes # Explicitly store num_nodes, though PyG can infer it

    logging.info("Custom PPI data loaded and processed successfully.")
    logging.info(f"Data object created - Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")
    
    return data


if __name__ == '__main__':
    # Example usage (replace with your actual file paths)
    # Create dummy CSV files for testing
    dummy_interactions_content = """protein1,protein2,protein1_id,protein2_id,combined_score
Q0010,Q0297,1,27,945
Q0010,Q0032,1,3,956
Q0032,Q0143,3,21,954
Q0297,Q0143,27,21,800
P_ISOLATED,P_ISOLATED_NEIGHBOR,100,101,700 
""" # P_ISOLATED_NEIGHBOR (101) will have no features. P_FEATURED_ISOLATED (200) will have features but no interactions.

    dummy_features_content = """protein_name,protein_id,featureA,featureB,featureC
ProteinA_for_1,1,0.1,0.2,0.3
ProteinB_for_3,3,0.4,0.5,0.6
ProteinC_for_27,27,0.7,0.8,0.9
ProteinD_for_21,21,1.0,1.1,1.2
P_FEATURED_ISOLATED_NAME,200,2.0,2.1,2.2 
""" # Protein 100 (P_ISOLATED) has no features here.

    with open("dummy_interactions.csv", "w") as f:
        f.write(dummy_interactions_content)
    with open("dummy_features.csv", "w") as f:
        f.write(dummy_features_content)

    logging.info("----- Running example data loading -----")
    interaction_file = "dummy_interactions.csv"
    feature_file = "dummy_features.csv"
    
    custom_data = load_custom_ppi_data(interaction_file, feature_file, score_threshold=750)
    
    if custom_data:
        logging.info(f"Loaded data: {custom_data}")
        logging.info(f"Node features shape: {custom_data.x.shape}")
        logging.info(f"Edge index shape: {custom_data.edge_index.shape}")
        if hasattr(custom_data, 'edge_attr') and custom_data.edge_attr is not None:
            logging.info(f"Edge attributes shape: {custom_data.edge_attr.shape}")
        else:
            logging.info("No edge attributes loaded.")
        logging.info(f"Number of nodes: {custom_data.num_nodes}")
        # logging.info(f"Protein ID to Node Index mapping: {custom_data.protein_id_to_node_idx}")

        # Verify feature assignment for a known protein
        protein_id_to_check = 1 # Expected features: [0.1, 0.2, 0.3]
        if protein_id_to_check in custom_data.protein_id_to_node_idx:
            node_idx = custom_data.protein_id_to_node_idx[protein_id_to_check]
            logging.info(f"Features for protein_id {protein_id_to_check} (node_idx {node_idx}): {custom_data.x[node_idx]}")
        else:
            logging.info(f"Protein_id {protein_id_to_check} not found in mapping.")

        # Check a protein that should have zero features (e.g., protein_id 100 from interactions but not features)
        protein_id_zero_feat = 100
        if protein_id_zero_feat in custom_data.protein_id_to_node_idx:
            node_idx_zero = custom_data.protein_id_to_node_idx[protein_id_zero_feat]
            logging.info(f"Features for protein_id {protein_id_zero_feat} (node_idx {node_idx_zero}): {custom_data.x[node_idx_zero]}")
        else:
            logging.info(f"Protein_id {protein_id_zero_feat} not in mapping (this might be okay if it was filtered or not in interactions).")
        
        # Check a protein that has features but no interactions (e.g. protein_id 200)
        protein_id_isolated_feat = 200
        if protein_id_isolated_feat in custom_data.protein_id_to_node_idx:
            node_idx_isolated = custom_data.protein_id_to_node_idx[protein_id_isolated_feat]
            logging.info(f"Features for protein_id {protein_id_isolated_feat} (node_idx {node_idx_isolated}): {custom_data.x[node_idx_isolated]}")
            # Check if this node is part of any edge
            is_in_edge = torch.any(custom_data.edge_index == node_idx_isolated).item()
            logging.info(f"Is protein_id {protein_id_isolated_feat} (node_idx {node_idx_isolated}) part of any edge? {is_in_edge}")

        # Clean up dummy files
        import os
        os.remove("dummy_interactions.csv")
        os.remove("dummy_features.csv")
    else:
        logging.error("Failed to load custom data.")
    logging.info("----- Example finished -----")
