"""
Unified preprocessing module for PPI link prediction.

This module combines all data preprocessing steps into a single pipeline:
1. Process raw STRING database files
2. Add protein IDs and remove duplicates
3. Extract protein features (GO terms, sequences, graph metrics)
4. Clean low-quality proteins
5. Save processed data for model training
"""

import os
import logging
import pandas as pd
import numpy as np
import networkx as nx
from Bio import SeqIO
import pickle
import time
from typing import Dict, Tuple, Optional, Set, List
from collections import defaultdict, Counter
import warnings
from visualization.ppi_visualizer import PPIVisualizer


logger = logging.getLogger(__name__)


class PPIPreprocessor:
    """
    Unified preprocessor for PPI data.
    
    This class handles the complete preprocessing pipeline from raw STRING
    database files to cleaned, feature-rich data ready for link prediction.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the preprocessor with configuration.
        
        Args:
            config: Dictionary containing preprocessing parameters
                - raw_dir: Directory containing raw STRING files
                - processed_dir: Directory to save processed data
                - confidence_threshold: Minimum interaction confidence score
                - min_go_terms: Minimum GO terms for protein to be kept
                - min_degree: Minimum degree for protein to be kept
        """
        self.config = config
        self.raw_dir = config.get('raw_dir', 'raw_data/')
        self.processed_dir = config.get('processed_dir', 'processed_data/')
        self.confidence_threshold = config.get('confidence_threshold', 700)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.processed_dir, exist_ok=True)
        
        # Initialize data containers
        self.interactions_df = None
        self.proteins_df = None
        self.graph = None
        self.protein_sequences = {}
        self.protein_to_go = {}
        self.string_to_alias = {}
        
    def run_pipeline(self) -> Dict:
        """
        Run the complete preprocessing pipeline.
        
        Returns:
            Dictionary containing processed data and statistics
        """
        logger.info("Starting preprocessing pipeline...")
        start_time = time.time()
        
        try:
            # Step 1: Process raw STRING files
            logger.info("Step 1: Processing raw STRING files...")
            self._process_string_files()
            
            # Step 2: Extract protein features
            logger.info("Step 2: Extracting protein features...")
            self._extract_protein_features()
            
            # Step 3: Clean low-quality proteins
            logger.info("Step 3: Cleaning low-quality proteins...")
            self._clean_proteins()
            
            # Step 4: Calculate final network statistics
            logger.info("Step 4: Calculating network statistics...")
            stats = self._calculate_network_stats()
            
            # 確保 largest_cc 被計算並儲存在 self 中，如果 visualizer 需要
            # _calculate_network_stats 中已經計算了 largest_cc，但未賦值給 self.largest_cc
            # 我們可以這樣做：
            if self.graph and self.graph.number_of_nodes() > 0:
                if nx.number_connected_components(self.graph) > 0:
                    largest_cc_nodes = max(nx.connected_components(self.graph), key=len)
                    self.largest_cc = self.graph.subgraph(largest_cc_nodes).copy() # 確保 largest_cc 被設定
                else: # 如果圖存在但沒有連通分量 (例如，只有節點沒有邊)
                    self.largest_cc = self.graph.copy() if self.graph else nx.Graph()
            else:
                self.largest_cc = nx.Graph()
            
            # Step 5: Save processed data
            logger.info("Step 5: Saving processed data...")
            self._save_processed_data()
            
            # NEW: Step 6: Run visualizations by passing data directly
            logger.info("Step 6: Generating visualizations for processed data...")
            try:
                viz_output_dir = os.path.join(self.processed_dir, "initial_visualizations")
                # 如果您在 __init__ 中設定了 self.preprocess_viz_output_dir，可以使用它
                # viz_output_dir = self.preprocess_viz_output_dir
                
                if not os.path.exists(viz_output_dir):
                    os.makedirs(viz_output_dir)

                visualizer = PPIVisualizer(
                    output_dir=viz_output_dir,
                    graph=self.graph, # 直接傳遞圖物件
                    protein_sequences=self.protein_sequences, # 直接傳遞
                    protein_to_go=self.protein_to_go,       # 直接傳遞
                    string_to_alias=self.string_to_alias,   # 直接傳遞
                    protein_info_df=self.proteins_df,       # 傳遞 self.proteins_df 作為 protein_info
                    largest_cc=self.largest_cc              # 直接傳遞 largest_cc
                )
                visualizer.run_visualization_pipeline()
                logger.info(f"Initial data visualizations saved to {viz_output_dir}")
            except Exception as viz_e:
                logger.error(f"Data visualization during preprocessing failed: {viz_e}", exc_info=True)

            elapsed = time.time() - start_time
            logger.info(f"Preprocessing pipeline (including initial visualization) completed in {elapsed:.2f} seconds")
            
            return {
                'proteins': len(self.proteins_df) if self.proteins_df is not None else 0,
                'interactions': len(self.interactions_df) if self.interactions_df is not None else 0,
                'stats': stats,
                'visualization_output_dir': viz_output_dir
            }
            
        except Exception as e:
            logger.error(f"Preprocessing pipeline failed: {str(e)}", exc_info=True)
            raise        
            
            
    def _process_string_files(self):
        """
        Process raw STRING database files.
        
        This method:
        1. Loads protein-protein interactions
        2. Filters by confidence threshold
        3. Loads protein info and aliases
        4. Processes sequences and GO annotations
        """
        # Load and filter interactions
        interactions_file = os.path.join(self.raw_dir, '4932.protein.links.v12.0.txt')
        logger.info(f"Loading interactions from {interactions_file}")
        
        interactions_df = pd.read_csv(interactions_file, sep=' ')
        logger.info(f"Loaded {len(interactions_df)} raw interactions")
        
        # Filter by confidence score
        interactions_df = interactions_df[
            interactions_df['combined_score'] >= self.confidence_threshold
        ].copy()
        logger.info(f"Filtered to {len(interactions_df)} high-confidence interactions")
        
        # Remove organism prefix from protein IDs
        interactions_df['protein1'] = interactions_df['protein1'].str.replace('4932.', '')
        interactions_df['protein2'] = interactions_df['protein2'].str.replace('4932.', '')
        
        # Remove duplicate edges (keeping only one direction)
        interactions_df['sorted_proteins'] = interactions_df.apply(
            lambda row: tuple(sorted([row['protein1'], row['protein2']])), axis=1
        )
        interactions_df = interactions_df.drop_duplicates(subset=['sorted_proteins'])
        interactions_df = interactions_df.drop('sorted_proteins', axis=1)
        
        self.interactions_df = interactions_df
        logger.info(f"Kept {len(interactions_df)} unique interactions")
        
        # Load protein info
        self._load_protein_info()
        
        # Load protein aliases for ID mapping
        self._load_protein_aliases()
        
        # Load sequences
        self._load_sequences()
        
        # Load GO annotations
        self._load_go_annotations()
        
    def _load_protein_info(self):
        """Load protein information from STRING info file."""
        info_file = os.path.join(self.raw_dir, '4932.protein.info.v12.0.txt')
        
        if os.path.exists(info_file):
            logger.info(f"Loading protein info from {info_file}")
            info_df = pd.read_csv(info_file, sep='\t')
            
            # Clean protein IDs
            if '#string_protein_id' in info_df.columns:
                info_df['#string_protein_id'] = info_df['#string_protein_id'].str.replace('4932.', '')
            
            # Store for later use
            self.protein_info = info_df
            logger.info(f"Loaded info for {len(info_df)} proteins")
        else:
            logger.warning(f"Protein info file not found: {info_file}")
            self.protein_info = None
            
    def _load_protein_aliases(self):
        """Load and process protein aliases for ID mapping."""
        alias_file = os.path.join(self.raw_dir, '4932.protein.aliases.v12.0.txt')
        
        if not os.path.exists(alias_file):
            logger.warning(f"Alias file not found: {alias_file}")
            return
            
        logger.info(f"Loading protein aliases from {alias_file}")
        alias_df = pd.read_csv(alias_file, sep='\t')
        
        # Build mapping dictionaries
        self.string_to_alias = defaultdict(lambda: defaultdict(list))
        
        # Identify columns (handle different formats)
        protein_col = alias_df.columns[0]  # Usually string_protein_id
        alias_col = alias_df.columns[1]    # Usually alias
        source_col = alias_df.columns[2] if len(alias_df.columns) > 2 else None
        
        for _, row in alias_df.iterrows():
            string_id = str(row[protein_col]).replace('4932.', '')
            alias = row[alias_col]
            source = row[source_col] if source_col else 'unknown'
            
            self.string_to_alias[string_id][source].append(alias)
            
        logger.info(f"Loaded aliases for {len(self.string_to_alias)} proteins")
        
    def _load_sequences(self):
        """Load protein sequences from FASTA file."""
        fasta_file = os.path.join(self.raw_dir, 'UP000002311_559292.fasta')
        
        if not os.path.exists(fasta_file):
            logger.warning(f"FASTA file not found: {fasta_file}")
            return
            
        logger.info(f"Loading sequences from {fasta_file}")
        
        with open(fasta_file, 'r') as handle:
            for record in SeqIO.parse(handle, 'fasta'):
                # Extract UniProt ID
                uniprot_id = record.id.split('|')[1] if '|' in record.id else record.id
                self.protein_sequences[uniprot_id] = str(record.seq)
                
        logger.info(f"Loaded {len(self.protein_sequences)} protein sequences")
        
    def _load_go_annotations(self):
        """Load GO annotations from GAF file."""
        gaf_file = os.path.join(self.raw_dir, 'sgd.gaf')
        
        if not os.path.exists(gaf_file):
            logger.warning(f"GAF file not found: {gaf_file}")
            return
            
        logger.info(f"Loading GO annotations from {gaf_file}")
        
        # GAF file columns
        columns = [
            'DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
            'DB_Reference', 'Evidence_Code', 'With_From', 'Aspect',
            'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type',
            'Taxon', 'Date', 'Assigned_By', 'Annotation_Extension',
            'Gene_Product_Form_ID'
        ]
        
        # Read GAF file
        gaf_df = pd.read_csv(gaf_file, sep='\t', comment='!', 
                            header=None, names=columns, low_memory=False)
        
        # Build protein to GO mapping
        self.protein_to_go = defaultdict(lambda: {'P': set(), 'F': set(), 'C': set()})
        
        for _, row in gaf_df.iterrows():
            protein_id = row['DB_Object_ID']
            go_id = row['GO_ID']
            aspect = row['Aspect']  # P: Process, F: Function, C: Component
            
            self.protein_to_go[protein_id][aspect].add(go_id)
            
        logger.info(f"Loaded GO annotations for {len(self.protein_to_go)} proteins")
        
    def _extract_protein_features(self):
        """
        Extract features for all proteins in the network.
        
        Features include:
        - GO terms (concatenated)
        - Protein sequences
        - Graph-based features (degree, betweenness, closeness)
        """
        # Get all unique proteins from interactions
        all_proteins = set(self.interactions_df['protein1'].unique()) | \
                      set(self.interactions_df['protein2'].unique())
        
        logger.info(f"Extracting features for {len(all_proteins)} proteins")
        
        # Initialize protein dataframe
        self.proteins_df = pd.DataFrame({
            'protein_name': sorted(list(all_proteins)),
            'protein_id': range(1,len(all_proteins)+1)
        })
        
        # Extract GO terms
        self._extract_go_features()
        
        # Extract sequences
        self._extract_sequence_features()
        
        # Calculate graph features
        self._calculate_graph_features()
        
        # Add binary feature indicators
        self.proteins_df['has_GO_term'] = (
            ~self.proteins_df['GO_term'].isna() & 
            (self.proteins_df['GO_term'] != '')
        ).astype(int)
        
        self.proteins_df['has_sequence'] = (
            ~self.proteins_df['sequence'].isna() & 
            (self.proteins_df['sequence'] != '')
        ).astype(int)
        
        logger.info(f"Feature extraction complete. Proteins with GO terms: "
                   f"{self.proteins_df['has_GO_term'].sum()}, "
                   f"with sequences: {self.proteins_df['has_sequence'].sum()}")
        
    def _extract_go_features(self):
        """Extract GO term features for proteins."""
        go_terms_list = []
        
        for protein_id in self.proteins_df['protein_name']:
            go_terms = set()
            
            # Direct lookup
            if protein_id in self.protein_to_go:
                go_data = self.protein_to_go[protein_id]
                for aspect in ['P', 'F', 'C']:
                    go_terms.update(go_data.get(aspect, set()))
            
            # Try aliases if no direct match
            if not go_terms and protein_id in self.string_to_alias:
                for source, aliases in self.string_to_alias[protein_id].items():
                    if source in ['UniProt_DR_SGD', 'SGD_ID']:
                        for alias in aliases:
                            if alias in self.protein_to_go:
                                go_data = self.protein_to_go[alias]
                                for aspect in ['P', 'F', 'C']:
                                    go_terms.update(go_data.get(aspect, set()))
                                break
                        if go_terms:
                            break
            
            # Join GO terms with semicolon
            go_terms_list.append(';'.join(sorted(go_terms)) if go_terms else '')
        
        self.proteins_df['GO_term'] = go_terms_list
        
    def _extract_sequence_features(self):
        """Extract protein sequences."""
        sequences_list = []
        
        for protein_id in self.proteins_df['protein_name']:
            sequence = ''
            
            # Direct lookup
            if protein_id in self.protein_sequences:
                sequence = self.protein_sequences[protein_id]
            
            # Try aliases if no direct match
            elif protein_id in self.string_to_alias:
                for source, aliases in self.string_to_alias[protein_id].items():
                    if source in ['Ensembl_UniProt', 'UniProt_AC']:
                        for alias in aliases:
                            if alias in self.protein_sequences:
                                sequence = self.protein_sequences[alias]
                                break
                        if sequence:
                            break
            
            sequences_list.append(sequence)
        
        self.proteins_df['sequence'] = sequences_list
        
    def _calculate_graph_features(self):
        """Calculate graph-based features for proteins."""
        # Create network
        self.graph = nx.Graph()
        
        # Add edges from interactions
        edges = [(row['protein1'], row['protein2']) 
                for _, row in self.interactions_df.iterrows()]
        self.graph.add_edges_from(edges)
        
        # Calculate features
        logger.info("Calculating graph features...")
        
        # Degree
        degree_dict = dict(self.graph.degree())
        self.proteins_df['degree'] = self.proteins_df['protein_name'].map(
            degree_dict
        ).fillna(0).astype(int)
        
        # Betweenness centrality
        logger.info("Calculating betweenness centrality...")
        betweenness_dict = nx.betweenness_centrality(self.graph)
        self.proteins_df['betweenness'] = self.proteins_df['protein_name'].map(
            betweenness_dict
        ).fillna(0)
        
        # Closeness centrality
        logger.info("Calculating closeness centrality...")
        closeness_dict = nx.closeness_centrality(self.graph)
        self.proteins_df['closeness'] = self.proteins_df['protein_name'].map(
            closeness_dict
        ).fillna(0)
        
    def _clean_proteins(self):
        """
        Iteratively remove low-quality proteins.
        
        Removes proteins that have:
        - No GO terms AND
        - No sequence AND
        - Degree <= 1
        
        This is done iteratively until no more proteins meet the criteria.
        """
        iteration = 0
        initial_proteins = len(self.proteins_df)
        initial_interactions = len(self.interactions_df)
        
        while True:
            iteration += 1
            logger.info(f"Cleaning iteration {iteration}")
            
            # Recalculate degrees based on current interactions
            if not self.interactions_df.empty:
                degree_counts = pd.concat([
                    self.interactions_df['protein1'],
                    self.interactions_df['protein2']
                ]).value_counts()
                
                self.proteins_df['degree'] = self.proteins_df['protein_name'].map(
                    degree_counts
                ).fillna(0).astype(int)
            else:
                self.proteins_df['degree'] = 0
            
            # Find proteins to remove
            to_remove = self.proteins_df[
                (self.proteins_df['GO_term'] == '') &
                (self.proteins_df['sequence'] == '') &
                (self.proteins_df['degree'] <= 1)
            ]['protein_name'].tolist()
            
            if not to_remove:
                logger.info("No more proteins to remove")
                break
                
            logger.info(f"Removing {len(to_remove)} proteins")
            
            # Remove from proteins dataframe
            self.proteins_df = self.proteins_df[
                ~self.proteins_df['protein_name'].isin(to_remove)
            ].copy()
            
            # Remove from interactions
            self.interactions_df = self.interactions_df[
                (~self.interactions_df['protein1'].isin(to_remove)) &
                (~self.interactions_df['protein2'].isin(to_remove))
            ].copy()
            
            # Check if we have any proteins left
            if self.proteins_df.empty:
                logger.warning("All proteins removed during cleaning!")
                break
        
        # Update protein IDs to be consecutive
        self.proteins_df['protein_id'] = range(1, len(self.proteins_df) + 1)
        
        # Update interaction IDs
        name_to_id = dict(zip(
            self.proteins_df['protein_name'],
            self.proteins_df['protein_id']
        ))
        
        self.interactions_df['protein1_id'] = self.interactions_df['protein1'].map(name_to_id)
        self.interactions_df['protein2_id'] = self.interactions_df['protein2'].map(name_to_id)
        
        # Final recalculation of graph features
        logger.info("Recalculating graph features after cleaning...")
        self._calculate_graph_features()
        
        logger.info(f"Cleaning complete. Removed {initial_proteins - len(self.proteins_df)} "
                   f"proteins and {initial_interactions - len(self.interactions_df)} interactions")
        
    def _calculate_network_stats(self) -> Dict:
        """Calculate final network statistics."""
        # Recreate graph with cleaned data
        self.graph = nx.Graph()
        edges = [(row['protein1'], row['protein2']) 
                for _, row in self.interactions_df.iterrows()]
        self.graph.add_edges_from(edges)
        
        # Calculate statistics
        stats = {
            'num_proteins': len(self.proteins_df),
            'num_interactions': len(self.interactions_df),
            'avg_degree': np.mean(list(dict(self.graph.degree()).values())),
            'network_density': nx.density(self.graph),
            'num_connected_components': nx.number_connected_components(self.graph),
            'proteins_with_go': self.proteins_df['has_GO_term'].sum(),
            'proteins_with_sequence': self.proteins_df['has_sequence'].sum(),
            'proteins_with_both': (
                (self.proteins_df['has_GO_term'] == 1) & 
                (self.proteins_df['has_sequence'] == 1)
            ).sum()
        }
        
        # Get largest connected component
        largest_cc = max(nx.connected_components(self.graph), key=len)
        stats['largest_cc_size'] = len(largest_cc)
        stats['largest_cc_fraction'] = len(largest_cc) / len(self.graph)
        
        logger.info("Network statistics:")
        for key, value in stats.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")
        
        return stats
        
    def _save_processed_data(self):
        """Save all processed data to files."""
        # Save protein features
        proteins_file = os.path.join(self.processed_dir, 'proteins_processed.csv')
        self.proteins_df.to_csv(proteins_file, index=False)
        logger.info(f"Saved protein features to {proteins_file}")
        
        # Save interactions
        interactions_file = os.path.join(self.processed_dir, 'interactions_processed.csv')
        self.interactions_df.to_csv(interactions_file, index=False)
        logger.info(f"Saved interactions to {interactions_file}")
        
        # Save network as pickle
        network_file = os.path.join(self.processed_dir, 'ppi_network.pkl')
        with open(network_file, 'wb') as f:
            pickle.dump(self.graph, f)
        logger.info(f"Saved network to {network_file}")
        
        # Save mappings
        mappings = {
            'protein_to_go': dict(self.protein_to_go),
            'protein_sequences': self.protein_sequences,
            'string_to_alias': dict(self.string_to_alias),
            'name_to_id': dict(zip(
                self.proteins_df['protein_name'],
                self.proteins_df['protein_id']
            ))
        }
        
        mappings_file = os.path.join(self.processed_dir, 'mappings.pkl')
        with open(mappings_file, 'wb') as f:
            pickle.dump(mappings, f)
        logger.info(f"Saved mappings to {mappings_file}")
        
        # Save a summary report
        report_file = os.path.join(self.processed_dir, 'preprocessing_report.txt')
        with open(report_file, 'w') as f:
            f.write("PPI Preprocessing Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total proteins: {len(self.proteins_df)}\n")
            f.write(f"Total interactions: {len(self.interactions_df)}\n")
            f.write(f"Proteins with GO terms: {self.proteins_df['has_GO_term'].sum()}\n")
            f.write(f"Proteins with sequences: {self.proteins_df['has_sequence'].sum()}\n")
            f.write(f"Average degree: {self.proteins_df['degree'].mean():.2f}\n")
            f.write(f"Max degree: {self.proteins_df['degree'].max()}\n")
            f.write(f"Network density: {nx.density(self.graph):.4f}\n")
        
        logger.info(f"Saved preprocessing report to {report_file}")


def preprocess_ppi_data(config: Dict) -> Dict:
    """
    Main function to preprocess PPI data.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with preprocessing results
    """
    preprocessor = PPIPreprocessor(config)
    return preprocessor.run_pipeline()


if __name__ == "__main__":
    # Example usage
    config = {
        'raw_dir': 'raw_data/',
        'processed_dir': 'processed_data/',
        'confidence_threshold': 700,
        'min_go_terms': 0,
        'min_degree': 0
    }
    
    results = preprocess_ppi_data(config)
    print(f"\nPreprocessing complete!")
    print(f"Processed {results['proteins']} proteins and {results['interactions']} interactions")