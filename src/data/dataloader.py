"""
Data loading utilities for the PPI network and protein complexes.
Provides PyTorch-compatible data loaders for training GVAE models.
"""

import os
import torch
import numpy as np
import networkx as nx
import logging
from torch.utils.data import Dataset, DataLoader

# Import logger setup function from preprocessing module
# If this is in a different package structure, you may need to adjust the import
from .preprocessing import setup_logger

# Initialize default logger
logger = setup_logger()


class PPINetworkDataset(Dataset):
    """
    Dataset class for loading PPI network data.
    """
    
    def __init__(self, processed_dir, max_nodes=20, log_level=None):
        """
        Initialize the dataset.
        
        Args:
            processed_dir: Directory containing processed data
            max_nodes: Maximum number of nodes in a complex
            log_level: Optional override for logging level
        """
        if log_level:
            logger.setLevel(log_level)
            
        logger.info(f"Initializing PPINetworkDataset from {processed_dir} with max_nodes={max_nodes}")
        
        self.processed_dir = processed_dir
        self.max_nodes = max_nodes
        
        # Statistics tracking
        self.skipped_complexes = 0
        self.missing_nodes = 0
        self.complex_sizes = []
        
        try:
            # Load node features
            features_path = os.path.join(processed_dir, 'features', 'node_features.npy')
            if not os.path.exists(features_path):
                # Try without 'features' subdirectory
                features_path = os.path.join(processed_dir, 'node_features.npy')
                
            logger.info(f"Loading node features from {features_path}")
            self.node_features = np.load(features_path)
            logger.info(f"Loaded node features with shape {self.node_features.shape}")
            
            # Load node IDs
            ids_path = os.path.join(processed_dir, 'features', 'node_ids.txt')
            if not os.path.exists(ids_path):
                # Try without 'features' subdirectory
                ids_path = os.path.join(processed_dir, 'node_ids.txt')
                
            logger.info(f"Loading node IDs from {ids_path}")
            with open(ids_path, 'r', encoding='utf-8') as f:
                self.node_ids = [line.strip() for line in f]
            logger.info(f"Loaded {len(self.node_ids)} node IDs")
            
            # Validate node features and IDs match
            if len(self.node_ids) != self.node_features.shape[0]:
                logger.warning(f"Mismatch between node IDs ({len(self.node_ids)}) and node features ({self.node_features.shape[0]})")
            
            # Create node ID to index mapping
            self.node_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
            
            # Load complexes
            self.complexes = []
            complex_dir = os.path.join(processed_dir, 'complexes')
            
            if not os.path.exists(complex_dir):
                logger.error(f"Complex directory not found: {complex_dir}")
                raise FileNotFoundError(f"Complex directory not found: {complex_dir}")
                
            logger.info(f"Loading complexes from {complex_dir}")
            
            complex_files = [f for f in os.listdir(complex_dir) if f.endswith('.edgelist')]
            logger.info(f"Found {len(complex_files)} complex files")
            
            for filename in complex_files:
                try:
                    complex_id = filename.split('.')[0]
                    complex_path = os.path.join(complex_dir, filename)
                    
                    # Load complex as graph
                    G = nx.read_weighted_edgelist(complex_path)
                    
                    # Skip complexes that are too large
                    if len(G.nodes()) > max_nodes:
                        logger.debug(f"Complex {complex_id} exceeds max_nodes ({len(G.nodes())} > {max_nodes}), skipping")
                        self.skipped_complexes += 1
                        continue
                    
                    # Track complex size
                    self.complex_sizes.append(len(G.nodes()))
                    
                    # Map nodes to indices
                    unmapped_nodes = []
                    node_indices = []
                    for node in G.nodes():
                        if node in self.node_to_idx:
                            node_indices.append(self.node_to_idx[node])
                        else:
                            unmapped_nodes.append(node)
                            self.missing_nodes += 1
                    
                    if unmapped_nodes:
                        logger.debug(f"Complex {complex_id}: {len(unmapped_nodes)} nodes not found in network")
                    
                    # Skip complexes with no mappable nodes
                    if not node_indices:
                        logger.warning(f"Complex {complex_id}: No nodes could be mapped to network, skipping")
                        self.skipped_complexes += 1
                        continue
                    
                    # Create adjacency matrix
                    adj_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
                    edge_count = 0
                    for i, node1_idx in enumerate(node_indices):
                        for j, node2_idx in enumerate(node_indices):
                            node1_id = self.node_ids[node1_idx]
                            node2_id = self.node_ids[node2_idx]
                            if i != j and G.has_edge(node1_id, node2_id):
                                adj_matrix[i, j] = 1.0
                                edge_count += 1
                    
                    # Create node feature matrix
                    node_matrix = np.zeros((max_nodes, self.node_features.shape[1]), dtype=np.float32)
                    for i, node_idx in enumerate(node_indices):
                        node_matrix[i] = self.node_features[node_idx]
                    
                    # Create mask
                    mask = np.zeros(max_nodes, dtype=np.float32)
                    mask[:len(node_indices)] = 1.0
                    
                    self.complexes.append({
                        'id': complex_id,
                        'node_indices': node_indices,
                        'adj_matrix': adj_matrix,
                        'node_features': node_matrix,
                        'mask': mask,
                        'size': len(node_indices),
                        'edges': edge_count
                    })
                    
                    # Log progress periodically
                    if len(self.complexes) % 50 == 0:
                        logger.debug(f"Loaded {len(self.complexes)} complexes...")
                        
                except Exception as e:
                    logger.warning(f"Error loading complex {filename}: {str(e)}")
            
            # Log final dataset statistics
            if self.complexes:
                logger.info(f"Successfully loaded {len(self.complexes)} complexes")
                
                # Calculate complex size statistics
                min_size = min(self.complex_sizes)
                max_size = max(self.complex_sizes)
                mean_size = np.mean(self.complex_sizes)
                median_size = np.median(self.complex_sizes)
                
                logger.info(f"Complex size statistics: min={min_size}, max={max_size}, mean={mean_size:.2f}, median={median_size:.2f}")
                
                # Create size distribution
                size_counts = {}
                for size in self.complex_sizes:
                    size_counts[size] = size_counts.get(size, 0) + 1
                
                logger.debug(f"Size distribution: {sorted(size_counts.items())}")
                
                # Calculate edge density statistics
                densities = [complex_data['edges'] / (complex_data['size'] * (complex_data['size'] - 1)) 
                           for complex_data in self.complexes if complex_data['size'] > 1]
                
                if densities:
                    logger.info(f"Complex edge density: min={min(densities):.2f}, max={max(densities):.2f}, mean={np.mean(densities):.2f}")
            else:
                logger.warning("No complexes were loaded!")
            
            # Log skipped complexes
            if self.skipped_complexes > 0:
                logger.info(f"Skipped {self.skipped_complexes} complexes (size > {max_nodes})")
            
            # Log missing nodes
            if self.missing_nodes > 0:
                logger.warning(f"Found {self.missing_nodes} nodes in complexes that were not in the network")
                
        except Exception as e:
            logger.error(f"Error initializing dataset: {str(e)}")
            logger.exception("Exception details:")
            raise
    
    def __len__(self):
        return len(self.complexes)
    
    def __getitem__(self, idx):
        try:
            complex_data = self.complexes[idx]
            
            return {
                'adj_matrix': torch.tensor(complex_data['adj_matrix']),
                'node_features': torch.tensor(complex_data['node_features']),
                'mask': torch.tensor(complex_data['mask']),
                'size': complex_data['size'],
                'id': complex_data['id']
            }
        except Exception as e:
            logger.error(f"Error retrieving item {idx}: {str(e)}")
            # Return a placeholder to prevent training crashes
            logger.warning(f"Returning empty placeholder for item {idx}")
            return {
                'adj_matrix': torch.zeros((self.max_nodes, self.max_nodes)),
                'node_features': torch.zeros((self.max_nodes, self.node_features.shape[1])),
                'mask': torch.zeros(self.max_nodes),
                'size': 0,
                'id': 'error'
            }


def get_complex_dataloader(processed_dir, max_nodes=20, batch_size=16, shuffle=True, log_level=None):
    """
    Create a DataLoader for protein complexes.
    
    Args:
        processed_dir: Directory containing processed data
        max_nodes: Maximum number of nodes in a complex
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        log_level: Optional override for logging level
        
    Returns:
        DataLoader: PyTorch DataLoader for complexes
    """
    if log_level:
        logger.setLevel(log_level)
    
    logger.info(f"Creating complex dataloader with batch_size={batch_size}, shuffle={shuffle}")
    
    try:
        # Create dataset
        dataset = PPINetworkDataset(processed_dir, max_nodes, log_level)
        
        # Check if dataset is empty
        if len(dataset) == 0:
            logger.error("Dataset is empty! No complexes to load.")
            raise ValueError("Empty dataset")
        
        # Calculate optimal batch size if necessary
        if batch_size > len(dataset):
            old_batch_size = batch_size
            batch_size = max(1, len(dataset) // 2)
            logger.warning(f"Batch size ({old_batch_size}) larger than dataset size ({len(dataset)}). Reducing to {batch_size}.")
        
        # Create dataloader
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        
        logger.info(f"Created dataloader with {len(dataset)} samples in {len(dataloader)} batches")
        
        return dataloader
    
    except Exception as e:
        logger.error(f"Failed to create dataloader: {str(e)}")
        logger.exception("Exception details:")
        raise


def load_ppi_network(processed_dir, log_level=None):
    """
    Load the full PPI network.
    
    Args:
        processed_dir: Directory containing processed data
        log_level: Optional override for logging level
        
    Returns:
        nx.Graph: NetworkX graph of PPI network
        np.ndarray: Node feature matrix
        list: Node IDs
    """
    if log_level:
        logger.setLevel(log_level)
    
    logger.info(f"Loading PPI network from {processed_dir}")
    
    try:
        # Load network
        network_file = os.path.join(processed_dir, 'ppi_network.edgelist')
        if not os.path.exists(network_file):
            logger.error(f"Network file not found: {network_file}")
            raise FileNotFoundError(f"Network file not found: {network_file}")
            
        logger.info(f"Loading network from {network_file}")
        G = nx.read_weighted_edgelist(network_file)
        logger.info(f"Loaded network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Load node features
        features_path = os.path.join(processed_dir, 'features', 'node_features.npy')
        if not os.path.exists(features_path):
            # Try without 'features' subdirectory
            features_path = os.path.join(processed_dir, 'node_features.npy')
            
        if not os.path.exists(features_path):
            logger.error(f"Node features file not found: {features_path}")
            raise FileNotFoundError(f"Node features file not found")
            
        logger.info(f"Loading node features from {features_path}")
        node_features = np.load(features_path)
        logger.info(f"Loaded node features with shape {node_features.shape}")
        
        # Load node IDs
        ids_path = os.path.join(processed_dir, 'features', 'node_ids.txt')
        if not os.path.exists(ids_path):
            # Try without 'features' subdirectory
            ids_path = os.path.join(processed_dir, 'node_ids.txt')
            
        if not os.path.exists(ids_path):
            logger.error(f"Node IDs file not found: {ids_path}")
            raise FileNotFoundError(f"Node IDs file not found")
            
        logger.info(f"Loading node IDs from {ids_path}")
        with open(ids_path, 'r', encoding='utf-8') as f:
            node_ids = [line.strip() for line in f]
        logger.info(f"Loaded {len(node_ids)} node IDs")
        
        # Validate data consistency
        if len(node_ids) != node_features.shape[0]:
            logger.warning(f"Mismatch between node IDs ({len(node_ids)}) and node features ({node_features.shape[0]})")
            
        # Check that network nodes match IDs
        network_nodes = set(G.nodes())
        id_nodes = set(node_ids)
        
        # Check for nodes in network but not in IDs
        network_only = network_nodes - id_nodes
        if network_only:
            count = len(network_only)
            logger.warning(f"{count} nodes in network but not in ID list")
            if count < 10:
                logger.debug(f"Network-only nodes: {network_only}")
                
        # Check for IDs not in network
        id_only = id_nodes - network_nodes
        if id_only:
            count = len(id_only)
            logger.warning(f"{count} nodes in ID list but not in network")
            if count < 10:
                logger.debug(f"ID-only nodes: {id_only}")
                
        # Calculate network statistics
        if G.number_of_nodes() > 0:
            # Check if graph is connected
            is_connected = nx.is_connected(G)
            components = list(nx.connected_components(G))
            largest_component = max(components, key=len)
            
            logger.info(f"Network is {'connected' if is_connected else 'disconnected with ' + str(len(components)) + ' components'}")
            logger.info(f"Largest component has {len(largest_component)} nodes ({len(largest_component)/G.number_of_nodes()*100:.1f}% of network)")
            
            # Calculate degree distribution
            degrees = [d for n, d in G.degree()]
            logger.info(f"Degree stats: min={min(degrees)}, max={max(degrees)}, mean={np.mean(degrees):.2f}, median={np.median(degrees):.2f}")
        
        logger.info(f"Successfully loaded PPI network and features")
        
        return G, node_features, node_ids
    
    except Exception as e:
        logger.error(f"Failed to load PPI network: {str(e)}")
        logger.exception("Exception details:")
        raise


def main(processed_dir="data/processed", max_nodes=20, batch_size=16, log_file=None, log_level=logging.INFO):
    """
    Main function to demonstrate dataloader usage.
    
    Args:
        processed_dir: Directory containing processed data
        max_nodes: Maximum number of nodes in a complex
        batch_size: Batch size for training
        log_file: Optional log file path
        log_level: Logging level
    """
    # Setup logging
    global logger
    logger = setup_logger(log_file=log_file, log_level=log_level)
    
    # Log main execution parameters
    logger.info("="*80)
    logger.info("Testing PPI network data loaders")
    logger.info("="*80)
    logger.info(f"Processed data directory: {processed_dir}")
    logger.info(f"Max nodes per complex: {max_nodes}")
    logger.info(f"Batch size: {batch_size}")
    
    try:
        # Load full network for reference
        logger.info("Loading full PPI network...")
        G, node_features, node_ids = load_ppi_network(processed_dir, log_level)
        
        # Create complex dataloader
        logger.info("Creating complex dataloader...")
        dataloader = get_complex_dataloader(
            processed_dir=processed_dir,
            max_nodes=max_nodes,
            batch_size=batch_size,
            shuffle=True,
            log_level=log_level
        )
        
        # Test iteration through dataloader
        logger.info("Testing dataloader iteration...")
        for batch_idx, batch in enumerate(dataloader):
            # Get batch statistics
            batch_size = batch['adj_matrix'].shape[0]
            adj_matrices = batch['adj_matrix']
            node_feature_matrices = batch['node_features']
            masks = batch['mask']
            
            # Log batch details
            logger.info(f"Batch {batch_idx+1}/{len(dataloader)}: {batch_size} complexes")
            logger.debug(f"  Adjacency matrix shape: {adj_matrices.shape}")
            logger.debug(f"  Node feature matrix shape: {node_feature_matrices.shape}")
            logger.debug(f"  Mask shape: {masks.shape}")
            
            # Calculate batch statistics
            sizes = [int(mask.sum().item()) for mask in masks]
            logger.debug(f"  Complex sizes in batch: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.2f}")
            
            # Only process first batch for testing
            if batch_idx == 0:
                break
        
        logger.info("Dataloader test completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Dataloader test failed: {str(e)}")
        logger.exception("Exception details:")
        return False


if __name__ == "__main__":
    import argparse
    import time
    import sys
    
    parser = argparse.ArgumentParser(description="Test PPI network data loaders")
    parser.add_argument("--data-dir", default="data/processed", help="Directory containing processed data")
    parser.add_argument("--max-nodes", type=int, default=20, help="Maximum number of nodes per complex")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for dataloader")
    parser.add_argument("--log-file", default=None, help="Log file path (default: log to console only)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    success = main(
        processed_dir=args.data_dir,
        max_nodes=args.max_nodes,
        batch_size=args.batch_size,
        log_file=args.log_file,
        log_level=log_level
    )
    
    sys.exit(0 if success else 1)