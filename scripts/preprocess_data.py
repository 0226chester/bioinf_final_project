#!/usr/bin/env python3
"""
Data preprocessing script for the Mouse PPI GVAE project.
Processes data from STRING, CORUM and GO databases into a format suitable for the GVAE model.
"""

import os
import argparse
import sys

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import (
    process_aliases, process_protein_info, build_ppi_network, 
    process_corum_complexes, process_go_annotations, export_network_data
)
from src.data.feature_engineering import (
    calculate_topology_features, create_go_feature_vectors, 
    reduce_go_dimensions, create_combined_feature_vectors, export_feature_data
)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process PPI network data for GVAE model')
    
    # Data directories
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                        help='Directory containing raw data files')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                        help='Directory to save processed data')
    
    # File names
    parser.add_argument('--aliases_file', type=str, default='10090.protein.aliases.v12.0.txt',
                        help='STRING protein aliases file')
    parser.add_argument('--info_file', type=str, default='10090.protein.info.v12.0.txt',
                        help='STRING protein info file')
    parser.add_argument('--links_file', type=str, default='10090.protein.links.v12.0.txt',
                        help='STRING protein links file')
    parser.add_argument('--corum_file', type=str, default='corum_allComplexes.txt',
                        help='CORUM protein complexes file')
    parser.add_argument('--mgi_file', type=str, default='mgi.gaf',
                        help='MGI gene ontology annotation file')
    
    # Processing parameters
    parser.add_argument('--confidence_threshold', type=int, default=700,
                        help='Confidence threshold for PPI network (0-1000)')
    parser.add_argument('--pca_components', type=int, default=10,
                        help='Number of PCA components for GO term reduction')
    parser.add_argument('--go_aspects', type=str, default='C',
                        help='GO aspects to use (P,F,C)')
    
    return parser.parse_args()


def main():
    """Main preprocessing function."""
    # Parse arguments
    args = parse_args()
    
    # Create absolute paths
    raw_dir = os.path.abspath(args.raw_dir)
    processed_dir = os.path.abspath(args.processed_dir)
    
    # Create full file paths
    aliases_file = os.path.join(raw_dir, args.aliases_file)
    info_file = os.path.join(raw_dir, args.info_file)
    links_file = os.path.join(raw_dir, args.links_file)
    corum_file = os.path.join(raw_dir, args.corum_file)
    mgi_file = os.path.join(raw_dir, args.mgi_file)
    
    # Create output directory
    os.makedirs(processed_dir, exist_ok=True)
    
    print("Starting data processing pipeline...")
    print(f"Raw data directory: {raw_dir}")
    print(f"Processed data directory: {processed_dir}")
    
    # Step 1: Process protein aliases
    id_mappings = process_aliases(aliases_file)
    
    # Step 2: Process protein information
    protein_info = process_protein_info(info_file)
    
    # Step 3: Build PPI network
    ppi_network = build_ppi_network(links_file, confidence_threshold=args.confidence_threshold)
    
    # Step 4: Process CORUM complexes
    complexes = process_corum_complexes(corum_file, id_mappings, ppi_network)
    
    # Step 5: Process GO annotations
    go_data = process_go_annotations(mgi_file, id_mappings, ppi_network)
    
    # Step 6: Export network data
    export_network_data(ppi_network, protein_info, complexes, processed_dir)
    
    # Step 7: Calculate topological features
    topo_features = calculate_topology_features(ppi_network)
    
    # Step 8: Create GO feature vectors
    # Parse GO aspects
    go_aspects = [aspect for aspect in args.go_aspects.strip()]
    if not go_aspects:
        go_aspects = ['C']  # Default to cellular component
    
    go_features = create_go_feature_vectors(ppi_network, go_data, id_mappings, aspects=go_aspects)
    
    # Step 9: Apply dimensionality reduction to GO features
    go_features_reduced = reduce_go_dimensions(go_features, n_components=args.pca_components)
    
    # Step 10: Create combined feature vectors
    node_vectors, feature_info = create_combined_feature_vectors(ppi_network, topo_features, go_features_reduced)
    
    # Step 11: Export feature data
    export_feature_data(node_vectors, feature_info, ppi_network, processed_dir)
    
    print("Data processing complete!")


if __name__ == "__main__":
    main()