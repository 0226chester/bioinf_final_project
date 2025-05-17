"""
Feature engineering module for creating node feature vectors.
Computes topological features and GO term embeddings.
"""

import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tqdm import tqdm


def calculate_topology_features(ppi_network):
    """
    Calculate topological features for each node in the network.
    
    Args:
        ppi_network: PPI network as NetworkX graph
        
    Returns:
        dict: Dictionary of node feature dictionaries
    """
    print("Calculating topological features...")
    
    # Calculate various centrality measures
    print("Computing degree centrality...")
    degree_centrality = nx.degree_centrality(ppi_network)
    
    print("Computing clustering coefficient...")
    clustering_coefficient = nx.clustering(ppi_network)
    
    # Betweenness centrality is computationally expensive for large networks
    # For very large networks, consider approximating or sampling
    print("Computing betweenness centrality (this may take a while)...")
    betweenness_centrality = nx.betweenness_centrality(ppi_network, k=min(500, ppi_network.number_of_nodes()))
    
    # Combine features into a single dictionary
    node_features = {}
    for node in ppi_network.nodes():
        node_features[node] = {
            'degree': degree_centrality[node],
            'clustering': clustering_coefficient[node],
            'betweenness': betweenness_centrality[node]
        }
    
    print(f"Calculated topological features for {len(node_features)} proteins")
    
    return node_features


def create_go_feature_vectors(ppi_network, go_data, id_mappings, aspects=None):
    """
    Create GO term feature vectors for each protein in the network.
    
    Args:
        ppi_network: PPI network as NetworkX graph
        go_data: GO term data from process_go_annotations
        id_mappings: ID mapping dictionaries
        aspects: List of GO aspects to include ('P', 'F', 'C'), defaults to ['C']
        
    Returns:
        dict: Dictionary mapping STRING IDs to GO term vectors
    """
    if aspects is None:
        aspects = ['C']  # Default to cellular component only
    
    gene_to_go = go_data['gene_to_go']
    go_indices = go_data['indices']
    
    # Map STRING IDs to GO terms and create feature vectors
    string_to_go = {}
    
    # For each protein in the network
    for string_id in tqdm(ppi_network.nodes(), desc="Creating GO vectors"):
        # Initialize a combined vector for all selected aspects
        combined_vector = []
        
        # Try to find corresponding MGI ID
        if string_id in id_mappings['string_to_gene']:
            gene_id = id_mappings['string_to_gene'][string_id]
            mgi_id = f"MGI:{gene_id}"
            
            # For each GO aspect
            for aspect in aspects:
                # Get the GO terms list and mapping for this aspect
                go_list = go_indices[aspect]['terms']
                go_to_index = go_indices[aspect]['mapping']
                
                # Create a binary vector for GO terms
                go_vector = np.zeros(len(go_list))
                
                # If we have GO annotations for this gene
                if mgi_id in gene_to_go:
                    for go_term in gene_to_go[mgi_id][aspect]:
                        if go_term in go_to_index:
                            go_vector[go_to_index[go_term]] = 1
                
                # Add to combined vector
                combined_vector.extend(go_vector)
        else:
            # No MGI mapping found, create a zero vector
            for aspect in aspects:
                go_list = go_indices[aspect]['terms']
                combined_vector.extend(np.zeros(len(go_list)))
        
        # Store the combined vector
        string_to_go[string_id] = np.array(combined_vector)
    
    # Calculate total dimensions
    total_dims = sum(len(go_indices[aspect]['terms']) for aspect in aspects)
    print(f"Created GO vectors with {total_dims} dimensions")
    
    return string_to_go


def reduce_go_dimensions(string_to_go, n_components=10):
    """
    Apply dimensionality reduction to GO term vectors.
    
    Args:
        string_to_go: Dictionary mapping STRING IDs to GO term vectors
        n_components: Number of components for dimensionality reduction
        
    Returns:
        dict: Dictionary mapping STRING IDs to reduced GO term vectors
    """
    print("Applying dimensionality reduction to GO vectors...")
    
    # Check if we have data to reduce
    if not string_to_go:
        print("No GO data to reduce")
        return string_to_go
    
    # Get dimensions
    sample_dim = next(iter(string_to_go.values())).shape[0]
    
    # Only reduce if dimension is large enough
    if sample_dim <= n_components:
        print(f"GO vector dimension ({sample_dim}) is already <= {n_components}, skipping reduction")
        return string_to_go
    
    # Get string IDs in a consistent order
    string_ids = list(string_to_go.keys())
    
    # Stack GO vectors into a matrix
    go_matrix = np.vstack([string_to_go[string_id] for string_id in string_ids])
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    go_matrix_reduced = pca.fit_transform(go_matrix)
    
    # Create new dictionary with reduced vectors
    string_to_go_reduced = {}
    for i, string_id in enumerate(string_ids):
        string_to_go_reduced[string_id] = go_matrix_reduced[i]
    
    print(f"Reduced GO vectors from {sample_dim} to {n_components} dimensions")
    print(f"Explained variance ratio: {np.sum(pca.explained_variance_ratio_):.2f}")
    
    return string_to_go_reduced


def create_combined_feature_vectors(ppi_network, topo_features, go_features):
    """
    Combine topological and GO features into feature vectors for each node.
    
    Args:
        ppi_network: PPI network as NetworkX graph
        topo_features: Dictionary of topological features
        go_features: Dictionary of GO term embeddings
        
    Returns:
        dict: Dictionary mapping node IDs to feature vectors
        dict: Feature information dictionary
    """
    print("Creating combined feature vectors...")
    
    # Initialize scaler for normalizing topological features
    scaler = MinMaxScaler()
    
    # Extract and reshape topological features for scaling
    topo_matrix = np.array([
        [topo_features[node]['degree'], 
         topo_features[node]['clustering'], 
         topo_features[node]['betweenness']]
        for node in ppi_network.nodes()
    ])
    
    # Scale the features
    topo_matrix_scaled = scaler.fit_transform(topo_matrix)
    
    # Get GO feature dimension
    go_feature_dim = next(iter(go_features.values())).shape[0] if go_features else 0
    
    # Combine scaled topological features with GO embeddings
    node_vectors = {}
    for i, node in enumerate(ppi_network.nodes()):
        # Combine topological features with GO features
        if node in go_features:
            combined_vector = np.concatenate([topo_matrix_scaled[i], go_features[node]])
        else:
            # If no GO features, use zeros
            combined_vector = np.concatenate([
                topo_matrix_scaled[i], 
                np.zeros(go_feature_dim)
            ])
        
        node_vectors[node] = combined_vector
    
    # Create feature information dictionary
    feature_info = {
        'topo_dim': 3,  # We used 3 topological features
        'go_dim': go_feature_dim,
        'total_dim': 3 + go_feature_dim
    }
    
    print(f"Created feature vectors of length {feature_info['total_dim']} for {len(node_vectors)} proteins")
    print(f"  Topological features: {feature_info['topo_dim']} dimensions")
    print(f"  GO features: {feature_info['go_dim']} dimensions")
    
    return node_vectors, feature_info


def export_feature_data(node_vectors, feature_info, ppi_network, output_dir):
    """
    Export feature vectors to files.
    
    Args:
        node_vectors: Dictionary mapping node IDs to feature vectors
        feature_info: Feature dimension information dictionary
        ppi_network: NetworkX graph of PPI network
        output_dir: Directory to save output files
    """
    import os
    
    print("Exporting feature data...")
    
    # Create numpy array from node vectors
    # Ensure consistent order by using the network nodes
    node_vectors_array = np.array([node_vectors[node] for node in ppi_network.nodes()])
    
    # Save node vectors as numpy array
    np.save(os.path.join(output_dir, "node_features.npy"), node_vectors_array)
    
    # Save feature dimension information
    with open(os.path.join(output_dir, "feature_info.txt"), "w", encoding='utf-8') as f:
        f.write(f"Total feature dimension: {feature_info['total_dim']}\n")
        f.write(f"Topological features: {feature_info['topo_dim']}\n")
        f.write(f"GO features: {feature_info['go_dim']}\n")
    
    print(f"Exported feature data to {output_dir}")