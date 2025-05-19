"""
Feature engineering module for creating node feature vectors.
Computes topological features and GO term embeddings.
"""

import time
import os
import logging
import networkx as nx
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tqdm import tqdm

# Import logger setup function from preprocessing module
# If this is in a different package structure, you may need to adjust the import
from .preprocessing import setup_logger

# Initialize default logger
logger = setup_logger()


def calculate_topology_features(ppi_network, log_level=None):
    """
    Calculate topological features for each node in the network.
    
    Args:
        ppi_network: PPI network as NetworkX graph
        log_level: Optional override for logging level
        
    Returns:
        dict: Dictionary of node feature dictionaries
    """
    if log_level:
        logger.setLevel(log_level)
        
    logger.info("Calculating topological features for %d nodes", ppi_network.number_of_nodes())
    
    node_features = {}
    
    try:
        # Calculate various centrality measures
        logger.info("Computing degree centrality...")
        degree_centrality = nx.degree_centrality(ppi_network)
        
        # Log degree centrality statistics
        degree_values = list(degree_centrality.values())
        logger.info("Degree centrality stats: min=%.4f, max=%.4f, mean=%.4f", 
                   min(degree_values), max(degree_values), np.mean(degree_values))
        
        logger.info("Computing clustering coefficient...")
        clustering_coefficient = nx.clustering(ppi_network)
        
        # Log clustering coefficient statistics
        cluster_values = list(clustering_coefficient.values())
        logger.info("Clustering coefficient stats: min=%.4f, max=%.4f, mean=%.4f", 
                   min(cluster_values), max(cluster_values), np.mean(cluster_values))
        
        # Betweenness centrality is computationally expensive for large networks
        if ppi_network.number_of_nodes() > 5000:
            logger.warning("Large network detected (%d nodes). Betweenness centrality calculation may take a long time.",
                         ppi_network.number_of_nodes())
        
        # For very large networks, consider approximating or sampling
        k_sample = min(500, ppi_network.number_of_nodes())
        logger.info("Computing betweenness centrality with k=%d (this may take a while)...", k_sample)
        start_time = time.time()
        betweenness_centrality = nx.betweenness_centrality(ppi_network, k=k_sample)
        end_time = time.time()
        
        # Log betweenness calculation time and statistics
        between_values = list(betweenness_centrality.values())
        logger.info("Betweenness calculation completed in %.2f seconds", end_time - start_time)
        logger.info("Betweenness centrality stats: min=%.6f, max=%.6f, mean=%.6f", 
                   min(between_values), max(between_values), np.mean(between_values))
        
        # Combine features into a single dictionary
        for node in ppi_network.nodes():
            node_features[node] = {
                'degree': degree_centrality[node],
                'clustering': clustering_coefficient[node],
                'betweenness': betweenness_centrality[node]
            }
        
        logger.info("Successfully calculated topological features for %d proteins", len(node_features))
        
    except MemoryError:
        logger.error("Memory error during topology calculation. Try using a smaller network or sampling approach.")
        raise
    except Exception as e:
        logger.error("Error calculating topological features: %s", str(e))
        logger.exception("Exception details:")
        raise
    
    # Validate features
    if len(node_features) < ppi_network.number_of_nodes():
        logger.warning("Not all nodes have topology features (%d/%d)", 
                     len(node_features), ppi_network.number_of_nodes())
    
    # Log sample node features for validation
    if node_features:
        sample_node = next(iter(node_features))
        logger.debug("Sample node features: %s: %s", sample_node, node_features[sample_node])
    
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
    
    logger.info("Creating GO feature vectors using aspects: %s", aspects)
    
    try:
        gene_to_go = go_data['gene_to_go']
        go_indices = go_data['indices']
        
        # Log GO term dimensions
        for aspect in aspects:
            if aspect in go_indices:
                logger.info("GO %s terms: %d unique terms", 
                           "Process" if aspect == 'P' else "Function" if aspect == 'F' else "Component",
                           len(go_indices[aspect]['terms']))
            else:
                logger.warning("Aspect '%s' not found in GO data", aspect)
        
        # Map STRING IDs to GO terms and create feature vectors
        string_to_go = {}
        
        # Statistics tracking
        mapped_proteins = 0
        unmapped_proteins = 0
        nonzero_vectors = 0
        
        # For each protein in the network
        for string_id in tqdm(ppi_network.nodes(), desc="Creating GO vectors"):
            # Initialize a combined vector for all selected aspects
            combined_vector = []
            has_annotations = False
            
            # Try to find corresponding MGI ID
            if string_id in id_mappings['string_to_gene']:
                mapped_proteins += 1
                gene_id = id_mappings['string_to_gene'][string_id]
                mgi_id = f"MGI:{gene_id}"
                
                # For each GO aspect
                for aspect in aspects:
                    if aspect not in go_indices:
                        logger.debug("Skipping missing aspect %s", aspect)
                        continue
                        
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
                                has_annotations = True
                    
                    # Add to combined vector
                    combined_vector.extend(go_vector)
            else:
                unmapped_proteins += 1
                # No MGI mapping found, create a zero vector
                for aspect in aspects:
                    if aspect in go_indices:
                        go_list = go_indices[aspect]['terms']
                        combined_vector.extend(np.zeros(len(go_list)))
            
            # Store the combined vector
            string_to_go[string_id] = np.array(combined_vector)
            
            if has_annotations:
                nonzero_vectors += 1
        
        # Calculate total dimensions
        total_dims = sum(len(go_indices[aspect]['terms']) for aspect in aspects if aspect in go_indices)
        logger.info("Created GO vectors with %d dimensions", total_dims)
        logger.info("Proteins mapped to MGI IDs: %d/%d (%.1f%%)", 
                   mapped_proteins, len(ppi_network), mapped_proteins/len(ppi_network)*100)
        logger.info("Proteins with at least one GO annotation: %d/%d (%.1f%%)",
                   nonzero_vectors, len(ppi_network), nonzero_vectors/len(ppi_network)*100)
        
        # Validate
        if nonzero_vectors == 0:
            logger.warning("No proteins have GO annotations! Check GO data and mapping.")
        
        # Log sample annotation for verification
        if string_to_go:
            sample_id = next(iter(string_to_go))
            sample_vec = string_to_go[sample_id]
            nonzero_count = np.count_nonzero(sample_vec)
            logger.debug("Sample GO vector: %s has %d/%d non-zero entries", 
                        sample_id, nonzero_count, len(sample_vec))
        
    except KeyError as ke:
        logger.error("Missing key in GO data or ID mappings: %s", str(ke))
        raise
    except Exception as e:
        logger.error("Error creating GO feature vectors: %s", str(e))
        logger.exception("Exception details:")
        raise
    
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
    logger.info("Applying dimensionality reduction to GO vectors (n_components=%d)...", n_components)
    
    # Check if we have data to reduce
    if not string_to_go:
        logger.warning("No GO data to reduce")
        return string_to_go
    
    try:
        # Get dimensions
        sample_dim = next(iter(string_to_go.values())).shape[0]
        logger.info("Original GO vector dimension: %d", sample_dim)
        
        # Only reduce if dimension is large enough
        if sample_dim <= n_components:
            logger.info("GO vector dimension (%d) is already <= %d, skipping reduction", 
                      sample_dim, n_components)
            return string_to_go
        
        # Get string IDs in a consistent order
        string_ids = list(string_to_go.keys())
        
        # Stack GO vectors into a matrix
        start_time = time.time()
        go_matrix = np.vstack([string_to_go[string_id] for string_id in string_ids])
        logger.debug("GO data matrix shape: %s", go_matrix.shape)
        
        # Check sparsity
        nonzero_ratio = np.count_nonzero(go_matrix) / go_matrix.size
        logger.info("GO data matrix sparsity: %.2f%% non-zero entries", nonzero_ratio * 100)
        
        # Apply PCA
        logger.info("Applying PCA...")
        pca = PCA(n_components=n_components)
        go_matrix_reduced = pca.fit_transform(go_matrix)
        end_time = time.time()
        
        logger.info("PCA completed in %.2f seconds", end_time - start_time)
        
        # Create new dictionary with reduced vectors
        string_to_go_reduced = {}
        for i, string_id in enumerate(string_ids):
            string_to_go_reduced[string_id] = go_matrix_reduced[i]
        
        # Log explained variance
        explained_var = np.sum(pca.explained_variance_ratio_)
        logger.info("Reduced GO vectors from %d to %d dimensions", sample_dim, n_components)
        logger.info("Explained variance ratio: %.2f", explained_var)
        
        if explained_var < 0.5:
            logger.warning("Low explained variance (%.2f). Consider increasing n_components.", explained_var)
            
        # Log individual component variance
        logger.debug("Individual component explained variance: %s", 
                    pca.explained_variance_ratio_[:min(5, len(pca.explained_variance_ratio_))])
            
        # Log sample reduced vector
        if string_to_go_reduced:
            sample_id = next(iter(string_to_go_reduced))
            logger.debug("Sample reduced GO vector: %s: %s", 
                        sample_id, string_to_go_reduced[sample_id])
        
    except Exception as e:
        logger.error("Error in dimensionality reduction: %s", str(e))
        logger.exception("Exception details:")
        raise
    
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
    logger.info("Creating combined feature vectors...")
    
    try:
        # Initialize scaler for normalizing topological features
        scaler = MinMaxScaler()
        
        # Check if all nodes have topology features
        missing_topo = [node for node in ppi_network.nodes() if node not in topo_features]
        if missing_topo:
            logger.warning("%d nodes missing topology features", len(missing_topo))
            if len(missing_topo) < 10:
                logger.debug("Nodes missing topology features: %s", missing_topo)
        
        # Extract and reshape topological features for scaling
        topo_matrix = np.array([
            [topo_features.get(node, {'degree': 0, 'clustering': 0, 'betweenness': 0})['degree'], 
             topo_features.get(node, {'degree': 0, 'clustering': 0, 'betweenness': 0})['clustering'], 
             topo_features.get(node, {'degree': 0, 'clustering': 0, 'betweenness': 0})['betweenness']]
            for node in ppi_network.nodes()
        ])
        
        # Scale the features
        logger.debug("Scaling topological features...")
        topo_matrix_scaled = scaler.fit_transform(topo_matrix)
        
        # Log scaling ranges
        logger.debug("Feature scaling ranges: %s", 
                    list(zip(['degree', 'clustering', 'betweenness'], 
                             scaler.data_min_, scaler.data_max_)))
        
        # Get GO feature dimension
        go_feature_dim = next(iter(go_features.values())).shape[0] if go_features else 0
        
        # Track statistics
        missing_go_count = 0
        
        # Combine scaled topological features with GO embeddings
        node_vectors = {}
        for i, node in enumerate(ppi_network.nodes()):
            # Combine topological features with GO features
            if node in go_features:
                combined_vector = np.concatenate([topo_matrix_scaled[i], go_features[node]])
            else:
                # If no GO features, use zeros
                missing_go_count += 1
                combined_vector = np.concatenate([
                    topo_matrix_scaled[i], 
                    np.zeros(go_feature_dim)
                ])
            
            node_vectors[node] = combined_vector
        
        # Create feature information dictionary
        feature_info = {
            'topo_dim': 3,  # We used 3 topological features
            'go_dim': go_feature_dim,
            'total_dim': 3 + go_feature_dim,
            'topo_names': ['degree', 'clustering', 'betweenness'],
            'go_reduced': go_feature_dim > 0
        }
        
        logger.info("Created feature vectors of length %d for %d proteins", 
                   feature_info['total_dim'], len(node_vectors))
        logger.info("  Topological features: %d dimensions", feature_info['topo_dim'])
        logger.info("  GO features: %d dimensions", feature_info['go_dim'])
        
        if missing_go_count > 0:
            logger.warning("%d nodes (%.1f%%) missing GO features, using zeros", 
                         missing_go_count, missing_go_count/len(ppi_network.nodes())*100)
        
        # Log sample vector
        if node_vectors:
            sample_node = next(iter(node_vectors))
            sample_vec = node_vectors[sample_node]
            logger.debug("Sample combined feature vector: %s [shape=%s]", 
                        sample_node, sample_vec.shape)
        
    except Exception as e:
        logger.error("Error creating combined feature vectors: %s", str(e))
        logger.exception("Exception details:")
        raise
    
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
    logger.info("Exporting feature data to %s...", output_dir)
    
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create numpy array from node vectors
        # Ensure consistent order by using the network nodes
        node_vectors_array = np.array([node_vectors[node] for node in ppi_network.nodes()])
        
        # Save node vectors as numpy array
        npy_path = os.path.join(output_dir, "node_features.npy")
        np.save(npy_path, node_vectors_array)
        logger.info("Saved node feature matrix [%s] to %s", 
                   node_vectors_array.shape, npy_path)
        
        # Save node IDs in same order as feature matrix
        ids_path = os.path.join(output_dir, "node_ids.txt")
        with open(ids_path, "w", encoding='utf-8') as f:
            for node in ppi_network.nodes():
                f.write(f"{node}\n")
        logger.info("Saved node IDs to %s", ids_path)
        
        # Save feature dimension information
        info_path = os.path.join(output_dir, "feature_info.txt")
        with open(info_path, "w", encoding='utf-8') as f:
            f.write(f"Feature Information\n")
            f.write(f"===================\n\n")
            f.write(f"Total feature dimension: {feature_info['total_dim']}\n")
            f.write(f"Topological features: {feature_info['topo_dim']}\n")
            f.write(f"  - {', '.join(feature_info['topo_names'])}\n")
            f.write(f"GO features: {feature_info['go_dim']}\n")
            if feature_info.get('go_reduced', False):
                f.write(f"  - Dimensionality reduced using PCA\n")
            
            # Add timestamp
            import datetime
            f.write(f"\nCreated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        logger.info("Saved feature information to %s", info_path)
        
        # Also save as pickle for easier loading with metadata
        import pickle
        pickle_path = os.path.join(output_dir, "feature_data.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump({
                'node_vectors': node_vectors,
                'feature_info': feature_info
            }, f)
        logger.info("Saved feature data dictionary to %s", pickle_path)
        
    except Exception as e:
        logger.error("Error exporting feature data: %s", str(e))
        logger.exception("Exception details:")
        raise


def main(processed_data_dir="data/processed", output_dir="data/features", 
         go_aspects=None, go_dim=10, log_file=None, log_level=logging.INFO):
    """
    Main feature engineering workflow function.
    
    Args:
        processed_data_dir: Directory containing processed network data
        output_dir: Directory to save feature data
        go_aspects: List of GO aspects to use, defaults to ['C']
        go_dim: Number of dimensions for GO feature reduction
        log_file: Optional log file path
        log_level: Logging level
    """
    # Setup logging
    global logger
    logger = setup_logger(log_file=log_file, log_level=log_level)
    
    # Set default GO aspects if not provided
    if go_aspects is None:
        go_aspects = ['C']
    
    # Log main execution parameters
    logger.info("="*80)
    logger.info("Starting feature engineering")
    logger.info("="*80)
    logger.info(f"Processed data directory: {processed_data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"GO aspects: {go_aspects}")
    logger.info(f"GO dimension reduction: {go_dim}")
    
    try:
        import pickle
        import time
        
        start_time = time.time()
        
        # Load network
        logger.info("Loading PPI network...")
        try:
            graphml_path = os.path.join(processed_data_dir, "ppi_network.graphml")
            if os.path.exists(graphml_path):
                ppi_network = nx.read_graphml(graphml_path)
                logger.info(f"Loaded network from GraphML with {ppi_network.number_of_nodes()} nodes and {ppi_network.number_of_edges()} edges")
            else:
                # Try loading from edge list if GraphML is not available
                edge_list_path = os.path.join(processed_data_dir, "ppi_network.edgelist")
                ppi_network = nx.read_weighted_edgelist(edge_list_path)
                logger.info(f"Loaded network from edge list with {ppi_network.number_of_nodes()} nodes and {ppi_network.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Failed to load network: {str(e)}")
            return False
        
        # Load ID mappings
        logger.info("Loading ID mappings...")
        try:
            with open(os.path.join(processed_data_dir, "id_mappings.pkl"), "rb") as f:
                id_mappings = pickle.load(f)
            logger.info(f"Loaded ID mappings with {len(id_mappings['string_to_gene'])} STRING to gene mappings")
        except Exception as e:
            logger.error(f"Failed to load ID mappings: {str(e)}")
            return False
        
        # Load GO data
        logger.info("Loading GO data...")
        try:
            with open(os.path.join(processed_data_dir, "go_data.pkl"), "rb") as f:
                go_data = pickle.load(f)
            
            # Log GO data statistics
            for aspect in ['P', 'F', 'C']:
                if aspect in go_data['indices']:
                    logger.info(f"GO {aspect} terms: {len(go_data['indices'][aspect]['terms'])}")
        except Exception as e:
            logger.error(f"Failed to load GO data: {str(e)}")
            return False
        
        # Calculate topological features
        logger.info("Step 1/4: Calculating topological features")
        topo_features = calculate_topology_features(ppi_network)
        
        # Create GO feature vectors
        logger.info("Step 2/4: Creating GO feature vectors")
        go_features = create_go_feature_vectors(ppi_network, go_data, id_mappings, aspects=go_aspects)
        
        # Reduce GO dimensions
        logger.info("Step 3/4: Reducing GO feature dimensions")
        go_features_reduced = reduce_go_dimensions(go_features, n_components=go_dim)
        
        # Create combined feature vectors
        logger.info("Step 4/4: Creating combined feature vectors")
        node_vectors, feature_info = create_combined_feature_vectors(ppi_network, topo_features, go_features_reduced)
        
        # Export feature data
        logger.info("Exporting feature data")
        export_feature_data(node_vectors, feature_info, ppi_network, output_dir)
        
        # Log execution time
        end_time = time.time()
        logger.info(f"Feature engineering completed in {end_time - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}")
        logger.exception("Exception details:")
        return False


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description="Create feature vectors for mouse PPI network")
    parser.add_argument("--data-dir", default="data/processed", help="Directory containing processed network data")
    parser.add_argument("--output-dir", default="data/features", help="Directory to save feature data")
    parser.add_argument("--go-aspects", default="C", help="GO aspects to use (P, F, C), comma-separated")
    parser.add_argument("--go-dim", type=int, default=10, help="Number of dimensions for GO feature reduction")
    parser.add_argument("--log-file", default=None, help="Log file path (default: log to console only)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    go_aspects = args.go_aspects.split(',')
    
    success = main(
        processed_data_dir=args.data_dir,
        output_dir=args.output_dir,
        go_aspects=go_aspects,
        go_dim=args.go_dim,
        log_file=args.log_file,
        log_level=log_level
    )
    
    sys.exit(0 if success else 1)