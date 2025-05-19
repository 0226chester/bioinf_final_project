#!/usr/bin/env python3
"""
Data preprocessing script for the Mouse PPI GVAE project.
Processes data from STRING, CORUM and GO databases into a format suitable for the GVAE model.
"""

import os
import argparse
import sys
import time
import logging
import datetime
import traceback

# Add the project root to the path so we can import the modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import processing modules
try:
    from src.data.preprocessing import (
        setup_logger, process_aliases, process_protein_info, build_ppi_network, 
        process_corum_complexes, process_go_annotations, export_network_data
    )
    from src.data.feature_engineering import (
        calculate_topology_features, create_go_feature_vectors, 
        reduce_go_dimensions, create_combined_feature_vectors, export_feature_data
    )
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure you're running this script from the project root or that the src module is in your PYTHONPATH")
    sys.exit(1)


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
    parser.add_argument('--corum_file', type=str, default='corum_allComplexes.json',
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
    
    # Logging parameters
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory to save log files')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose (DEBUG) logging')
    parser.add_argument('--log_to_file', action='store_true',
                        help='Save logs to file in addition to console output')
    
    return parser.parse_args()


def validate_files(raw_dir, file_paths):
    """
    Validate that all required files exist.
    
    Args:
        raw_dir: Raw data directory
        file_paths: Dictionary of file paths
        
    Returns:
        bool: True if all files exist, False otherwise
    """
    missing_files = []
    
    for name, path in file_paths.items():
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        logger.error("Missing input files:")
        for missing in missing_files:
            logger.error(f"  - {missing}")
        return False
    
    return True


def main():
    """Main preprocessing function."""
    # Parse arguments
    args = parse_args()
    
    # Setup logging
    global logger
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create timestamped log file name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(args.log_dir, f"preprocessing_{timestamp}.log") if args.log_to_file else None
    
    # Initialize logger
    logger = setup_logger(log_file=log_file, log_level=log_level)
    
    # Log script start
    logger.info("="*80)
    logger.info("Starting Mouse PPI network preprocessing")
    logger.info("="*80)
    
    try:
        # Start timing
        total_start_time = time.time()
        
        # Create absolute paths
        raw_dir = os.path.abspath(args.raw_dir)
        processed_dir = os.path.abspath(args.processed_dir)
        
        # Create full file paths
        file_paths = {
            'aliases': os.path.join(raw_dir, args.aliases_file),
            'info': os.path.join(raw_dir, args.info_file),
            'links': os.path.join(raw_dir, args.links_file),
            'corum': os.path.join(raw_dir, args.corum_file),
            'mgi': os.path.join(raw_dir, args.mgi_file)
        }
        
        # Create output directory
        os.makedirs(processed_dir, exist_ok=True)
        
        # Log configuration
        logger.info(f"Raw data directory: {raw_dir}")
        logger.info(f"Processed data directory: {processed_dir}")
        logger.info(f"Confidence threshold: {args.confidence_threshold}")
        logger.info(f"PCA components: {args.pca_components}")
        logger.info(f"GO aspects: {args.go_aspects}")
        
        # Validate input files
        logger.info("Validating input files...")
        if not validate_files(raw_dir, file_paths):
            logger.error("File validation failed. Aborting.")
            return 1
        
        logger.info("All input files found. Starting processing.")
        
        # Initialize processing step counter
        step = 1
        
        # Step 1: Process protein aliases
        logger.info(f"Step {step}/11: Processing protein aliases")
        step += 1
        step_start = time.time()
        id_mappings = process_aliases(file_paths['aliases'])
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Validate ID mappings
        if not id_mappings or not id_mappings.get('string_to_uniprot'):
            logger.warning("ID mappings may be incomplete. Check input file.")
        
        # Step 2: Process protein information
        logger.info(f"Step {step}/11: Processing protein information")
        step += 1
        step_start = time.time()
        protein_info = process_protein_info(file_paths['info'])
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Validate protein info
        if not protein_info:
            logger.warning("No protein information was extracted. Check input file.")
        
        # Step 3: Build PPI network
        logger.info(f"Step {step}/11: Building PPI network")
        step += 1
        step_start = time.time()
        ppi_network = build_ppi_network(file_paths['links'], confidence_threshold=args.confidence_threshold)
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Validate network
        if ppi_network.number_of_nodes() == 0 or ppi_network.number_of_edges() == 0:
            logger.error("Empty network created. Check confidence threshold or input file.")
            return 1
        
        # Step 4: Process CORUM complexes
        logger.info(f"Step {step}/11: Processing CORUM complexes")
        step += 1
        step_start = time.time()
        complexes = process_corum_complexes(file_paths['corum'], id_mappings, ppi_network)
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Validate complexes
        if not complexes:
            logger.warning("No protein complexes were extracted. Check input file or network.")
        
        # Step 5: Process GO annotations
        logger.info(f"Step {step}/11: Processing GO annotations")
        step += 1
        step_start = time.time()
        go_data = process_go_annotations(file_paths['mgi'], id_mappings, ppi_network)
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Validate GO data
        for aspect in ['P', 'F', 'C']:
            if aspect in go_data['indices'] and not go_data['indices'][aspect]['terms']:
                logger.warning(f"No GO terms found for aspect {aspect}. Check input file.")
        
        # Step 6: Export network data
        logger.info(f"Step {step}/11: Exporting network data")
        step += 1
        step_start = time.time()
        export_network_data(ppi_network, protein_info, complexes, processed_dir)
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Create features directory
        features_dir = os.path.join(processed_dir, 'features')
        os.makedirs(features_dir, exist_ok=True)
        
        # Step 7: Calculate topological features
        logger.info(f"Step {step}/11: Calculating topological features")
        step += 1
        step_start = time.time()
        topo_features = calculate_topology_features(ppi_network)
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Step 8: Create GO feature vectors
        logger.info(f"Step {step}/11: Creating GO feature vectors")
        step += 1
        
        # Parse GO aspects
        go_aspects = [aspect for aspect in args.go_aspects.strip()]
        if not go_aspects:
            go_aspects = ['C']  # Default to cellular component
        
        logger.info(f"Using GO aspects: {go_aspects}")
        
        step_start = time.time()
        go_features = create_go_feature_vectors(ppi_network, go_data, id_mappings, aspects=go_aspects)
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Step 9: Apply dimensionality reduction to GO features
        logger.info(f"Step {step}/11: Reducing GO feature dimensions")
        step += 1
        step_start = time.time()
        go_features_reduced = reduce_go_dimensions(go_features, n_components=args.pca_components)
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Step 10: Create combined feature vectors
        logger.info(f"Step {step}/11: Creating combined feature vectors")
        step += 1
        step_start = time.time()
        node_vectors, feature_info = create_combined_feature_vectors(ppi_network, topo_features, go_features_reduced)
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Step 11: Export feature data
        logger.info(f"Step {step}/11: Exporting feature data")
        step += 1
        step_start = time.time()
        export_feature_data(node_vectors, feature_info, ppi_network, features_dir)
        logger.info(f"Completed in {time.time() - step_start:.2f} seconds")
        
        # Calculate total processing time
        total_time = time.time() - total_start_time
        
        # Log processing summary
        logger.info("="*80)
        logger.info("Data processing completed successfully!")
        logger.info("="*80)
        logger.info(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
        logger.info(f"Network: {ppi_network.number_of_nodes()} nodes, {ppi_network.number_of_edges()} edges")
        logger.info(f"Complexes: {len(complexes)} protein complexes")
        logger.info(f"Feature vectors: {feature_info['total_dim']} dimensions ({feature_info['topo_dim']} topological + {feature_info['go_dim']} GO)")
        logger.info(f"All processed data saved to: {processed_dir}")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        logger.error("Exception details:")
        logger.error(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)