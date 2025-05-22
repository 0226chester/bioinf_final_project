import os
import logging
import pandas as pd
import networkx as nx
from Bio import SeqIO
import pickle
import time
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("preprocess.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ppi_preprocessing")

class PPIPreprocessor:
    def __init__(self, 
                 ppi_file="4932.protein.links.v12.0.txt", 
                 info_file="4932.protein.info.v12.0.txt",
                 alias_file="4932.protein.aliases.v12.0.txt",
                 fasta_file="UP000002311_559292.fasta",
                 gaf_file="sgd.gaf",
                 confidence_threshold=700,
                 output_dir="processed_data"):
        """
        Initialize the PPI network preprocessor.
        
        Args:
            ppi_file: File containing protein-protein interactions
            info_file: File containing protein information
            alias_file: File containing protein aliases/ID mappings
            fasta_file: FASTA file containing protein sequences
            gaf_file: GAF file containing GO annotations
            confidence_threshold: Minimum confidence score for interactions (0-1000)
            output_dir: Directory to save processed data
        """
        self.ppi_file = ppi_file
        self.info_file = info_file
        self.alias_file = alias_file
        self.fasta_file = fasta_file
        self.gaf_file = gaf_file
        self.confidence_threshold = confidence_threshold
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
    
    def process_protein_links(self):
        """Process protein-protein interaction data with confidence filtering."""
        start_time = time.time()
        logger.info(f"Processing protein links from {self.ppi_file}")
        logger.info(f"Using confidence threshold: {self.confidence_threshold}")
        
        # Read PPI data
        try:
            df = pd.read_csv(self.ppi_file, sep=' ')
            logger.info(f"Loaded {len(df)} raw interactions")
            
            # Log column names for debugging
            logger.info(f"PPI file columns: {df.columns.tolist()}")
            
            # Check column names and adjust if necessary
            if 'protein1' not in df.columns and 'protein2' not in df.columns:
                if '#' in df.columns[0]:  # Handle comment character in header
                    new_cols = [col.replace('#', '') for col in df.columns]
                    df.columns = new_cols
                    logger.info(f"Renamed columns: {df.columns.tolist()}")
            
            # Filter by confidence score
            filtered_df = df[df['combined_score'] >= self.confidence_threshold].copy()
            logger.info(f"Filtered to {len(filtered_df)} interactions with confidence score >= {self.confidence_threshold}")
            
            # Strip the organism prefix (4932.) from protein IDs if present
            filtered_df.loc[:, 'protein1'] = filtered_df['protein1'].str.replace('4932.', '')
            filtered_df.loc[:, 'protein2'] = filtered_df['protein2'].str.replace('4932.', '')
            
            # Save filtered interactions
            filtered_df.to_csv(os.path.join(self.output_dir, 'filtered_interactions.csv'), index=False)
            logger.info(f"Saved filtered interactions to {self.output_dir}/filtered_interactions.csv")
            
            elapsed = time.time() - start_time
            logger.info(f"Processed protein links in {elapsed:.2f} seconds")
            return filtered_df
            
        except Exception as e:
            logger.error(f"Error processing protein links: {str(e)}")
            raise
    
    def process_protein_info(self):
        """Process protein information data."""
        start_time = time.time()
        logger.info(f"Processing protein info from {self.info_file}")
        
        try:
            # Read protein info
            df = pd.read_csv(self.info_file, sep='\t')
            logger.info(f"Loaded information for {len(df)} proteins")
            
            # Log column names for debugging
            logger.info(f"Protein info file columns: {df.columns.tolist()}")
            
            # Check for the correct column name for protein IDs
            protein_id_column = None
            for col in ['protein_external_id', '#string_protein_id', 'string_protein_id', 'protein_id']:
                if col in df.columns:
                    protein_id_column = col
                    break
            
            if protein_id_column:
                # Strip the organism prefix if present
                df[protein_id_column] = df[protein_id_column].str.replace('4932.', '')
            
            # Save processed protein info
            df.to_csv(os.path.join(self.output_dir, 'protein_info.csv'), index=False)
            logger.info(f"Saved protein info to {self.output_dir}/protein_info.csv")
            
            elapsed = time.time() - start_time
            logger.info(f"Processed protein info in {elapsed:.2f} seconds")
            return df
            
        except Exception as e:
            logger.error(f"Error processing protein info: {str(e)}")
            raise
    
    def process_protein_aliases(self):
        """Process protein aliases/mapping data."""
        start_time = time.time()
        logger.info(f"Processing protein aliases from {self.alias_file}")
        
        try:
            # Read alias data and inspect column names
            df = pd.read_csv(self.alias_file, sep='\t')
            logger.info(f"Loaded {len(df)} alias entries")
            
            # Log column names for debugging
            logger.info(f"Aliases file columns: {df.columns.tolist()}")
            
            # Identify the correct column names
            protein_id_column = None
            alias_column = None
            source_column = None
            
            # Check for protein ID column
            for col in ['string_protein_id', '#string_id', 'protein_id', 'STRING_id', '#protein_external_id']:
                if col in df.columns:
                    protein_id_column = col
                    break
            
            # Check for alias column
            for col in ['alias', 'Alias', 'external_id']:
                if col in df.columns:
                    alias_column = col
                    break
            
            # Check for source column
            for col in ['source', 'Source', 'alias_source']:
                if col in df.columns:
                    source_column = col
                    break
            
            # If columns are not found, try to use the first few columns
            if not protein_id_column and len(df.columns) > 0:
                protein_id_column = df.columns[0]
                logger.info(f"Using {protein_id_column} as protein ID column")
            
            if not alias_column and len(df.columns) > 1:
                alias_column = df.columns[1]
                logger.info(f"Using {alias_column} as alias column")
            
            if not source_column and len(df.columns) > 2:
                source_column = df.columns[2]
                logger.info(f"Using {source_column} as source column")
            
            # If we have the necessary columns, proceed
            if protein_id_column and alias_column:
                logger.info(f"Using columns: {protein_id_column}, {alias_column}, {source_column}")
                
                # Strip the organism prefix if present
                if '4932.' in str(df[protein_id_column].iloc[0]):
                    df[protein_id_column] = df[protein_id_column].str.replace('4932.', '')
                
                # Create mapping dictionaries
                string_to_alias = {}
                alias_to_string = {}
                
                # Build mappings
                for _, row in df.iterrows():
                    string_id = row[protein_id_column]
                    alias = row[alias_column]
                    source = row[source_column] if source_column else "unknown"
                    
                    if string_id not in string_to_alias:
                        string_to_alias[string_id] = {}
                        
                    if source not in string_to_alias[string_id]:
                        string_to_alias[string_id][source] = []
                        
                    string_to_alias[string_id][source].append(alias)
                    
                    # Also create reverse mapping
                    if source not in alias_to_string:
                        alias_to_string[source] = {}
                        
                    alias_to_string[source][alias] = string_id
                
                # Save processed aliases to pickle files for easy loading
                with open(os.path.join(self.output_dir, 'string_to_alias.pkl'), 'wb') as f:
                    pickle.dump(string_to_alias, f)
                    
                with open(os.path.join(self.output_dir, 'alias_to_string.pkl'), 'wb') as f:
                    pickle.dump(alias_to_string, f)
                
                # Also save as CSV for inspection
                df.to_csv(os.path.join(self.output_dir, 'protein_aliases.csv'), index=False)
                
                logger.info(f"Saved protein alias mappings to {self.output_dir}")
                elapsed = time.time() - start_time
                logger.info(f"Processed protein aliases in {elapsed:.2f} seconds")
                
                return string_to_alias, alias_to_string
            else:
                logger.error(f"Could not identify required columns in alias file. Found columns: {df.columns.tolist()}")
                raise ValueError(f"Missing required columns in alias file. Found: {df.columns.tolist()}")
            
        except Exception as e:
            logger.error(f"Error processing protein aliases: {str(e)}")
            raise
    
    def process_fasta_sequences(self):
        """Process protein sequences from FASTA file."""
        start_time = time.time()
        logger.info(f"Processing protein sequences from {self.fasta_file}")
        
        try:
            # Parse FASTA file
            sequences = {}
            sequence_df_rows = []
            
            with open(self.fasta_file, 'r') as handle:
                for record in SeqIO.parse(handle, 'fasta'):
                    # Extract UniProt ID from the FASTA header
                    uniprot_id = record.id.split('|')[1] if '|' in record.id else record.id
                    sequence = str(record.seq)
                    description = record.description
                    
                    sequences[uniprot_id] = sequence
                    sequence_df_rows.append({
                        'uniprot_id': uniprot_id,
                        'description': description,
                        'sequence': sequence,
                        'length': len(sequence)
                    })
            
            # Create a DataFrame for easier manipulation
            sequence_df = pd.DataFrame(sequence_df_rows)
            logger.info(f"Loaded {len(sequence_df)} protein sequences")
            
            # Save sequences
            sequence_df.to_csv(os.path.join(self.output_dir, 'protein_sequences.csv'), index=False)
            
            with open(os.path.join(self.output_dir, 'protein_sequences.pkl'), 'wb') as f:
                pickle.dump(sequences, f)
            
            logger.info(f"Saved protein sequences to {self.output_dir}")
            elapsed = time.time() - start_time
            logger.info(f"Processed protein sequences in {elapsed:.2f} seconds")
            
            return sequences, sequence_df
            
        except Exception as e:
            logger.error(f"Error processing protein sequences: {str(e)}")
            raise
    
    def process_go_annotations(self):
        """Process Gene Ontology annotations from GAF file."""
        start_time = time.time()
        logger.info(f"Processing GO annotations from {self.gaf_file}")
        
        try:
            # GAF file format has 17 columns with specific meaning
            columns = [
                'DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
                'DB_Reference', 'Evidence_Code', 'With_From', 'Aspect',
                'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type',
                'Taxon', 'Date', 'Assigned_By', 'Annotation_Extension',
                'Gene_Product_Form_ID'
            ]
            
            # Read GAF file, skipping comment lines
            gaf_df = pd.read_csv(self.gaf_file, sep='\t', comment='!', 
                                header=None, names=columns, low_memory=False)
            
            logger.info(f"Loaded {len(gaf_df)} GO annotations")
            
            # Create protein to GO term mappings
            protein_to_go = {}
            go_to_protein = {}
            
            for _, row in gaf_df.iterrows():
                protein_id = row['DB_Object_ID']
                go_id = row['GO_ID']
                aspect = row['Aspect']  # P: Biological Process, F: Molecular Function, C: Cellular Component
                
                # Initialize dictionaries if needed
                if protein_id not in protein_to_go:
                    protein_to_go[protein_id] = {'P': set(), 'F': set(), 'C': set()}
                
                if go_id not in go_to_protein:
                    go_to_protein[go_id] = {'P': set(), 'F': set(), 'C': set()}
                
                # Add mappings
                protein_to_go[protein_id][aspect].add(go_id)
                go_to_protein[go_id][aspect].add(protein_id)
            
            # Save GO annotations
            gaf_df.to_csv(os.path.join(self.output_dir, 'go_annotations.csv'), index=False)
            
            with open(os.path.join(self.output_dir, 'protein_to_go.pkl'), 'wb') as f:
                pickle.dump(protein_to_go, f)
                
            with open(os.path.join(self.output_dir, 'go_to_protein.pkl'), 'wb') as f:
                pickle.dump(go_to_protein, f)
            
            logger.info(f"Saved GO annotations to {self.output_dir}")
            elapsed = time.time() - start_time
            logger.info(f"Processed GO annotations in {elapsed:.2f} seconds")
            
            return protein_to_go, go_to_protein, gaf_df
            
        except Exception as e:
            logger.error(f"Error processing GO annotations: {str(e)}")
            raise
    
    def create_network(self, interactions_df):
        """Create NetworkX graph from interaction data."""
        start_time = time.time()
        logger.info("Creating NetworkX graph from filtered interactions")
        
        try:
            # Create undirected graph
            G = nx.Graph()
            
            # Add edges from filtered interactions
            for _, row in interactions_df.iterrows():
                protein1 = row['protein1']
                protein2 = row['protein2']
                score = row['combined_score']
                
                # Add edge with confidence score as an attribute
                # (even though we're creating an unweighted graph for now)
                G.add_edge(protein1, protein2, weight=score, confidence=score)
            
            logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            # Save graph in pickle format (modern version of nx.write_gpickle)
            with open(os.path.join(self.output_dir, 'ppi_network.gpickle'), 'wb') as f:
                pickle.dump(G, f)
            logger.info(f"Saved network to {self.output_dir}/ppi_network.gpickle")
            
            # Also save as edge list for easier inspection
            nx.write_edgelist(G, os.path.join(self.output_dir, 'ppi_network.edgelist'))
            logger.info(f"Saved edge list to {self.output_dir}/ppi_network.edgelist")
            
            elapsed = time.time() - start_time
            logger.info(f"Created network in {elapsed:.2f} seconds")
            
            return G
            
        except Exception as e:
            logger.error(f"Error creating network: {str(e)}")
            raise
    
    def analyze_network(self, G):
        """Perform basic network analysis."""
        logger.info("Performing basic network analysis")
        
        try:
            # Get connected components
            connected_components = list(nx.connected_components(G))
            largest_cc = max(connected_components, key=len)
            
            # Create subgraph of largest connected component
            largest_cc_graph = G.subgraph(largest_cc).copy()
            
            logger.info(f"Network has {len(connected_components)} connected components")
            logger.info(f"Largest connected component has {largest_cc_graph.number_of_nodes()} nodes " 
                      f"and {largest_cc_graph.number_of_edges()} edges")
            
            # Basic network statistics
            stats = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'components': len(connected_components),
                'largest_cc_nodes': largest_cc_graph.number_of_nodes(),
                'largest_cc_edges': largest_cc_graph.number_of_edges(),
                'average_clustering': nx.average_clustering(G),
            }
            
            # Save largest connected component (modern version of nx.write_gpickle)
            with open(os.path.join(self.output_dir, 'largest_cc.gpickle'), 'wb') as f:
                pickle.dump(largest_cc_graph, f)
            logger.info(f"Saved largest connected component to {self.output_dir}/largest_cc.gpickle")
            
            # Save statistics
            with open(os.path.join(self.output_dir, 'network_stats.txt'), 'w') as f:
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
            
            logger.info("Completed basic network analysis")
            return stats, largest_cc_graph
            
        except Exception as e:
            logger.error(f"Error analyzing network: {str(e)}")
            raise
    
    def run_pipeline(self):
        """Run the complete preprocessing pipeline."""
        logger.info("Starting preprocessing pipeline")
        
        try:
            # Process protein links first
            interactions_df = self.process_protein_links()
            
            # Process protein info
            self.process_protein_info()
            
            # Process protein aliases
            self.process_protein_aliases()
            
            # Create network from the interaction data
            G = self.create_network(interactions_df)
            
            # Analyze network
            stats, largest_cc = self.analyze_network(G)
            
            # Process the sequence and GO data after network creation
            # This allows the pipeline to continue even if these fail
            try:
                self.process_fasta_sequences()
            except Exception as e:
                logger.error(f"Error processing sequences, but continuing: {str(e)}")
                
            try:
                self.process_go_annotations()
            except Exception as e:
                logger.error(f"Error processing GO annotations, but continuing: {str(e)}")
            
            logger.info("Preprocessing pipeline completed successfully")
            
            return {
                'graph': G,
                'largest_cc': largest_cc,
                'stats': stats
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Create preprocessor with default settings
    # You can customize parameters here
    preprocessor = PPIPreprocessor(
        confidence_threshold=700,  # You can adjust this threshold
        output_dir="processed_data"
    )
    
    # Run the full pipeline
    results = preprocessor.run_pipeline()
    
    # Print some network statistics
    print("\nNetwork Statistics:")
    for key, value in results['stats'].items():
        print(f"{key}: {value}")