"""
Data loading utilities for the PPI network and protein complexes.
Provides PyTorch-compatible data loaders for training GVAE models.
"""

import os
import torch
import numpy as np
import networkx as nx
from torch.utils.data import Dataset, DataLoader


class PPINetworkDataset(Dataset):
    """
    Dataset class for loading PPI network data.
    """
    
    def __init__(self, processed_dir, max_nodes=20):
        """
        Initialize the dataset.
        
        Args:
            processed_dir: Directory containing processed data
            max_nodes: Maximum number of nodes in a complex
        """
        self.processed_dir = processed_dir
        self.max_nodes = max_nodes
        
        # Load node features
        self.node_features = np.load(os.path.join(processed_dir, 'node_features.npy'))
        
        # Load node IDs
        with open(os.path.join(processed_dir, 'node_ids.txt'), 'r', encoding='utf-8') as f:
            self.node_ids = [line.strip() for line in f]
        
        # Create node ID to index mapping
        self.node_to_idx = {node_id: i for i, node_id in enumerate(self.node_ids)}
        
        # Load complexes
        self.complexes = []
        complex_dir = os.path.join(processed_dir, 'complexes')
        for filename in os.listdir(complex_dir):
            if filename.endswith('.edgelist'):
                complex_id = filename.split('.')[0]
                
                # Load complex as graph
                G = nx.read_weighted_edgelist(os.path.join(complex_dir, filename))
                
                # Skip complexes that are too large
                if len(G.nodes()) > max_nodes:
                    continue
                
                # Map nodes to indices
                node_indices = [self.node_to_idx[node] for node in G.nodes() if node in self.node_to_idx]
                
                # Create adjacency matrix
                adj_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
                for i, node1 in enumerate(node_indices):
                    for j, node2 in enumerate(node_indices):
                        if i != j and G.has_edge(self.node_ids[node1], self.node_ids[node2]):
                            adj_matrix[i, j] = 1.0
                
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
                    'size': len(node_indices)
                })
    
    def __len__(self):
        return len(self.complexes)
    
    def __getitem__(self, idx):
        complex_data = self.complexes[idx]
        
        return {
            'adj_matrix': torch.tensor(complex_data['adj_matrix']),
            'node_features': torch.tensor(complex_data['node_features']),
            'mask': torch.tensor(complex_data['mask']),
            'size': complex_data['size'],
            'id': complex_data['id']
        }


def get_complex_dataloader(processed_dir, max_nodes=20, batch_size=16, shuffle=True):
    """
    Create a DataLoader for protein complexes.
    
    Args:
        processed_dir: Directory containing processed data
        max_nodes: Maximum number of nodes in a complex
        batch_size: Batch size for training
        shuffle: Whether to shuffle the data
        
    Returns:
        DataLoader: PyTorch DataLoader for complexes
    """
    dataset = PPINetworkDataset(processed_dir, max_nodes)
    
    # Print dataset statistics
    sizes = [complex_data['size'] for complex_data in dataset.complexes]
    print(f"Loaded {len(dataset)} complexes")
    print(f"Complex size statistics:")
    print(f"  Min: {min(sizes)}")
    print(f"  Max: {max(sizes)}")
    print(f"  Mean: {np.mean(sizes):.2f}")
    print(f"  Median: {np.median(sizes):.2f}")
    
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def load_ppi_network(processed_dir):
    """
    Load the full PPI network.
    
    Args:
        processed_dir: Directory containing processed data
        
    Returns:
        nx.Graph: NetworkX graph of PPI network
        np.ndarray: Node feature matrix
        list: Node IDs
    """
    # Load network
    network_file = os.path.join(processed_dir, 'ppi_network.edgelist')
    G = nx.read_weighted_edgelist(network_file)
    
    # Load node features
    node_features = np.load(os.path.join(processed_dir, 'node_features.npy'))
    
    # Load node IDs
    with open(os.path.join(processed_dir, 'node_ids.txt'), 'r', encoding='utf-8') as f:
        node_ids = [line.strip() for line in f]
    
    print(f"Loaded PPI network with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    return G, node_features, node_ids