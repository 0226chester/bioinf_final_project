"""
Data preprocessing module for mouse PPI network data.
Handles loading and basic cleaning of STRING, CORUM, and GO data.
"""

import os
import logging
import sys
import json
import networkx as nx
import numpy as np
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
import re
from tqdm import tqdm
from goatools.anno.gaf_reader import GafReader


# Configure logging
def setup_logger(log_file=None, log_level=logging.INFO):
    """
    Setup logger with specified output file and verbosity level.
    
    Args:
        log_file: Path to log file (None for console only)
        log_level: Logging level (default: INFO)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logger
    logger = logging.getLogger("ppi_preprocessing")
    logger.setLevel(log_level)
    logger.handlers = []  # Remove any existing handlers
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")
        
    return logger

# Initialize default logger
logger = setup_logger()

def process_aliases(aliases_file, organism_id='10090', source_priorities=None, 
                    include_all_sources=False, return_format='dict', log_level=None):
    """
    Process STRING aliases file to create mapping dictionaries between different ID types.
    
    Args:
        aliases_file (str): Path to STRING aliases file
        organism_id (str): NCBI Taxonomy ID to filter proteins (default: '10090' for Mus musculus)
        source_priorities (dict): Optional dict mapping source types to priority values (lower = higher priority)
        include_all_sources (bool): Whether to include all sources or only common ones
        return_format (str): Output format - 'dict', 'dataframe', or 'both'
        log_level: Optional override for logging level
        
    Returns:
        dict or pd.DataFrame or tuple: Mappings between different ID types in requested format
        
    File format expected:
        Tab-separated file with columns: string_id, alias, source
        string_id format: <organism_id>.<protein_id>
    """

    # Configure logger
    logger = logging.getLogger(__name__)
    if log_level:
        logger.setLevel(log_level)
    
    # Set default source priorities if not provided
    if source_priorities is None:
        source_priorities = {
            'UniProt_AC': 1,                 # Canonical UniProt accession - highest priority
            'UniProt_ID': 2,                 # UniProt ID
            'Ensembl_protein_id': 3,         # Ensembl protein ID
            'Ensembl': 4,                    # Generic Ensembl ID
            'UniProt_GN_Name': 5,            # Gene name from UniProt
            'Ensembl_gene': 6,               # Ensembl gene ID
            'Ensembl_MGI': 7,                # Mouse Genome Informatics ID
            'KEGG_GENEID': 8,                # KEGG gene ID
            'KEGG_KEGGID_SHORT': 9,          # KEGG ID (short form)
            'UniProt_DE_RecName_Full': 10,   # Recommended full name
        }
    
    # Define source type categories for organized mapping
    source_categories = {
        'uniprot': [src for src in source_priorities if src.startswith('UniProt')],
        'ensembl': [src for src in source_priorities if src.startswith('Ensembl')],
        'gene': ['UniProt_GN_Name', 'Ensembl_gene', 'KEGG_GENEID'],
        'kegg': [src for src in source_priorities if src.startswith('KEGG')],
        'name': ['UniProt_DE_RecName_Full', 'KEGG_NAME', 'UniProt_DE_AltName_Full'],
        'mgi': ['UniProt_DR_MGI']
    }
    
    logger.info(f"Processing protein aliases from {aliases_file} for organism {organism_id}")
    
    # Regular expression to match STRING IDs for specified organism
    organism_pattern = re.compile(f"^{organism_id}\\..*")
    
    # Initialize mapping dictionaries - using defaultdict to handle multiple mappings
    id_mappings = {
        'string_to_uniprot': defaultdict(list),
        'string_to_gene': defaultdict(list),
        'string_to_ensembl': defaultdict(list),
        'string_to_name': defaultdict(list),
        'string_to_mgi': defaultdict(list),
        'uniprot_to_string': defaultdict(list),
        'gene_to_string': defaultdict(list),
        'ensembl_to_string': defaultdict(list),
        'name_to_string': defaultdict(list),
        'mgi_to_string': defaultdict(list),
        'all_mappings': []  # For DataFrame creation
    }
    
    # For each source, maintain a separate mapping to avoid overwrites
    # and allow source prioritization later
    source_to_mapping = defaultdict(lambda: defaultdict(list))
    
    # Track statistics for validation
    stats = {
        'total_lines': 0,
        'organism_matches': 0,
        'skipped_lines': 0,
        'source_counts': defaultdict(int)
    }
    
    try:
        # First check if file exists
        with open(aliases_file, 'r', encoding='utf-8') as f:
            pass
            
        # Read and process file
        df_chunks = pd.read_csv(aliases_file, sep='\t', comment='#', 
                                names=['string_id', 'alias', 'source'],
                                chunksize=500000)  # Process in chunks for large files
        
        for chunk in tqdm(df_chunks, desc="Processing alias chunks"):
            stats['total_lines'] += len(chunk)
            
            # Filter for the specified organism
            chunk = chunk[chunk['string_id'].str.match(organism_pattern, na=False)]
            stats['organism_matches'] += len(chunk)
            
            # Filter out rows with missing data
            chunk = chunk.dropna()
            
            # Process each row
            for _, row in chunk.iterrows():
                string_id = row['string_id']
                alias = row['alias'].strip()
                source = row['source']
                
                # Skip empty aliases
                if not alias:
                    stats['skipped_lines'] += 1
                    continue
                
                # Track source counts
                stats['source_counts'][source] += 1
                
                # Store in source-specific mapping
                source_to_mapping[source][string_id].append(alias)
                
                # Store all mappings for dataframe if needed
                id_mappings['all_mappings'].append({
                    'string_id': string_id,
                    'alias': alias,
                    'source': source
                })
                
                # Organize mappings by category
                # UniProt mappings
                # MGI mappings
                if source in ['UniProt_DR_MGI']:
                    id_mappings['string_to_mgi'][string_id].append((alias, source))
                    id_mappings['mgi_to_string'][alias].append((string_id, source))
                
                if source.startswith('UniProt'):
                    id_mappings['string_to_uniprot'][string_id].append((alias, source))
                    id_mappings['uniprot_to_string'][alias].append((string_id, source))
                
                # Ensembl mappings
                elif source.startswith('Ensembl'):
                    id_mappings['string_to_ensembl'][string_id].append((alias, source))
                    id_mappings['ensembl_to_string'][alias].append((string_id, source))
                
                # Gene name mappings
                elif source in ['UniProt_GN_Name', 'Ensembl_gene', 'KEGG_GENEID']:
                    id_mappings['string_to_gene'][string_id].append((alias, source))
                    id_mappings['gene_to_string'][alias].append((string_id, source))
                
                # Protein name mappings
                elif source in ['UniProt_DE_RecName_Full', 'KEGG_NAME', 'UniProt_DE_AltName_Full']:
                    id_mappings['string_to_name'][string_id].append((alias, source))
                    id_mappings['name_to_string'][alias].append((string_id, source))
        
        # Post-process mappings to apply source priorities
        for mapping_dict in ['string_to_uniprot', 'string_to_gene', 'string_to_ensembl', 'string_to_name', 'string_to_mgi',
                            'uniprot_to_string', 'gene_to_string', 'ensembl_to_string', 'name_to_string',  'mgi_to_string']:
            # For each ID in the mapping
            for id_key, aliases_with_sources in id_mappings[mapping_dict].items():
                # Sort aliases based on source priority
                sorted_aliases = sorted(
                    aliases_with_sources,
                    key=lambda x: source_priorities.get(x[1], 999)  # Default high value for unknown sources
                )
                # Replace with sorted list or just the highest priority
                id_mappings[mapping_dict][id_key] = sorted_aliases
        
        # Create clean dictionary output
        clean_mappings = {}
        for mapping_name, mapping_dict in id_mappings.items():
            # Skip the all_mappings list
            if mapping_name == 'all_mappings':
                continue
                
            # Convert defaultdict to regular dict
            clean_dict = {}
            for key, value in mapping_dict.items():
                # For simplicity, use only the highest priority alias for each ID
                if value and not include_all_sources:
                    clean_dict[key] = value[0][0]  # Just the alias, not the source
                else:
                    # Include all aliases if requested
                    clean_dict[key] = [v[0] for v in value]
            
            clean_mappings[mapping_name] = clean_dict
        
        # Create DataFrame if needed
        if return_format in ['dataframe', 'both']:
            df = pd.DataFrame(id_mappings['all_mappings'])
        
        # Log detailed stats
        logger.info(f"Processed {stats['total_lines']} total lines from aliases file")
        logger.info(f"Found {stats['organism_matches']} entries for organism {organism_id}")
        logger.info(f"Skipped {stats['skipped_lines']} lines due to missing/invalid data")
        
        # Log counts for most common sources
        top_sources = sorted(stats['source_counts'].items(), key=lambda x: x[1], reverse=True)[:10]
        logger.info("Top 10 sources:")
        for source, count in top_sources:
            logger.info(f"  {source}: {count} entries")
        
        # Log mappings created
        for mapping_name, mapping_dict in clean_mappings.items():
            logger.info(f"Created {len(mapping_dict)} {mapping_name} mappings")
        
        # Validation check
        for mapping_name, mapping_dict in clean_mappings.items():
            if len(mapping_dict) == 0:
                logger.warning(f"No mappings created for {mapping_name}!")
            else:
                sample = list(mapping_dict.items())[:3]
                logger.debug(f"Sample {mapping_name} mapping: {sample}")
        
        # Return based on requested format
        if return_format == 'dict':
            return clean_mappings
        elif return_format == 'dataframe':
            return df
        else:  # 'both'
            return clean_mappings, df
            
    except FileNotFoundError:
        logger.error(f"Aliases file not found: {aliases_file}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Aliases file is empty: {aliases_file}")
        raise
    except Exception as e:
        logger.error(f"Error processing aliases file: {str(e)}")
        raise


def process_protein_info(info_file):
    """
    Process STRING protein info file to extract protein names and metadata.
    
    Args:
        info_file: Path to STRING protein info file
        
    Returns:
        dict: Dictionary with STRING IDs as keys and protein info as values
    """
    logger.info(f"Processing protein information from {info_file}")
    
    protein_info = {}
    total_lines = 0
    skipped_lines = 0
    
    try:
        with open(info_file, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)
            
            for line in tqdm(f):
                total_lines += 1
                
                if line.startswith('#'):
                    skipped_lines += 1
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    logger.debug(f"Skipping incomplete line: {line.strip()}")
                    skipped_lines += 1
                    continue
                    
                string_id, name, size, annotation = parts
                
                # Store protein information
                protein_info[string_id] = {
                    'name': name,
                    'size': int(size),
                    'annotation': annotation
                }
    except FileNotFoundError:
        logger.error(f"Protein info file not found: {info_file}")
        raise
    except Exception as e:
        logger.error(f"Error processing protein info file: {str(e)}")
        raise
    
    logger.info(f"Processed {total_lines} total lines from protein info file")
    logger.info(f"Skipped {skipped_lines} lines")
    logger.info(f"Extracted information for {len(protein_info)} proteins")
    
    # Log sample and validation
    if protein_info:
        sample_id = next(iter(protein_info))
        logger.debug(f"Sample protein entry: {sample_id}: {protein_info[sample_id]}")
    else:
        logger.warning("No protein information was extracted!")
    
    return protein_info


def build_ppi_network(links_file, confidence_threshold=700):
    """
    Construct protein-protein interaction network from STRING links file.
    
    Args:
        links_file: Path to STRING links file
        confidence_threshold: Minimum confidence score to include an interaction
        
    Returns:
        nx.Graph: NetworkX graph of protein interactions
    """
    logger.info(f"Building PPI network from {links_file} with threshold {confidence_threshold}")
    
    # Create empty graph
    G = nx.Graph()
    
    # Count statistics
    total_edges = 0
    filtered_edges = 0
    malformed_lines = 0
    min_score = float('inf')
    max_score = 0
    
    try:
        with open(links_file, 'r', encoding='utf-8') as f:
            # Skip header
            next(f)
            
            for line_num, line in enumerate(tqdm(f), 1):
                try:
                    parts = line.strip().split()
                    if len(parts) < 3:
                        logger.debug(f"Line {line_num}: Malformed line (not enough fields): {line.strip()}")
                        malformed_lines += 1
                        continue
                        
                    protein1, protein2, score = parts
                    score = int(score)
                    total_edges += 1
                    
                    # Track score distribution
                    min_score = min(min_score, score)
                    max_score = max(max_score, score)
                    
                    # Filter by confidence score
                    if score >= confidence_threshold:
                        # Add nodes and edge to graph
                        G.add_edge(protein1, protein2, weight=score/1000.0)  # Normalize to [0,1]
                        filtered_edges += 1
                except ValueError as ve:
                    logger.warning(f"Line {line_num}: Could not parse score: {line.strip()}")
                    malformed_lines += 1
                except Exception as e:
                    logger.warning(f"Line {line_num}: Unexpected error: {str(e)}")
                    malformed_lines += 1
    except FileNotFoundError:
        logger.error(f"Links file not found: {links_file}")
        raise
    except Exception as e:
        logger.error(f"Error processing links file: {str(e)}")
        raise
    
    # Network statistics
    logger.info(f"Processed {total_edges} total interactions")
    logger.info(f"Encountered {malformed_lines} malformed lines")
    logger.info(f"Score range: {min_score} to {max_score}")
    logger.info(f"Built network with {G.number_of_nodes()} proteins and {G.number_of_edges()} high-confidence interactions")
    logger.info(f"Filtered out {total_edges - filtered_edges} low-confidence interactions (< {confidence_threshold/1000.0})")
    
    # Additional network analysis
    if G.number_of_nodes() > 0:
        # Check if graph is connected
        is_connected = nx.is_connected(G)
        components = list(nx.connected_components(G))
        
        logger.info(f"Network is {'connected' if is_connected else 'disconnected with ' + str(len(components)) + ' components'}")
        logger.info(f"Largest component has {len(max(components, key=len))} nodes ({(len(max(components, key=len))/G.number_of_nodes())*100:.1f}% of network)")
        
        # Calculate basic metrics
        avg_degree = sum(d for _, d in G.degree()) / G.number_of_nodes()
        logger.info(f"Average node degree: {avg_degree:.2f}")
        logger.info(f"Network density: {nx.density(G):.6f}")
        
        # Performance warning for large networks
        if G.number_of_nodes() > 10000:
            logger.warning(f"Large network detected ({G.number_of_nodes()} nodes). Some operations may be slow.")
    else:
        logger.warning("Empty network created! Check confidence threshold or input file.")
    
    return G


def process_corum_complexes(corum_json_file, id_mappings, ppi_network, organism_filter="Mouse", min_complex_size=3, log_level=None):
    """
    Extract protein complexes from CORUM database JSON and map to STRING network.
    
    Args:
        corum_json_file: Path to CORUM complexes JSON file
        id_mappings: Dictionaries for ID conversion
        ppi_network: PPI network as NetworkX graph
        organism_filter: Filter for organism (default "Mouse")
        min_complex_size: Minimum number of mapped proteins to consider a complex (default 3)
        log_level: Optional override for logging level
        
    Returns:
        dict: Dictionary of complex subgraphs
    """
    # Configure logger
    logger = logging.getLogger(__name__)
    if log_level:
        logger.setLevel(log_level)
        
    logger.info(f"Processing CORUM protein complexes from {corum_json_file}")
    
    # Initialize statistics counters
    complexes = {}
    stats = {
        'total_complexes': 0,
        'organism_complexes': 0,
        'mapped_complexes': 0,
        'singleton_complexes': 0,
        'disconnected_complexes': 0,
        'too_small_complexes': 0
    }
    
    # Track mapping statistics
    mapping_stats = {
        'total_proteins': 0,
        'mapped_proteins': 0,
        'unmapped_proteins': 0,
        'proteins_not_in_network': 0
    }
    
    try:
        # Read and parse JSON file
        with open(corum_json_file, 'r', encoding='utf-8') as f:
            logger.info("Loading CORUM JSON file...")
            corum_data = json.load(f)
            logger.info(f"Loaded {len(corum_data)} complexes from JSON")
            
        # Process each complex
        for complex_data in tqdm(corum_data, desc="Processing complexes"):
            stats['total_complexes'] += 1
            
            try:
                complex_id = complex_data.get('complex_id', '')
                complex_name = complex_data.get('complex_name', '')
                organism = complex_data.get('organism', '')
                
                # Filter for organism (Mouse)
                if organism_filter.lower() in organism.lower() or "10090" in organism:
                    stats['organism_complexes'] += 1
                    logger.debug(f"Found {organism_filter} complex: {complex_id} ({complex_name})")
                    
                    # Extract UniProt IDs from complex subunits
                    uniprot_ids = []
                    gene_names = []
                    
                    if 'subunits' in complex_data and isinstance(complex_data['subunits'], list):
                        for subunit in complex_data['subunits']:
                            if 'swissprot' in subunit and 'uniprot_id' in subunit['swissprot']:
                                uniprot_id = subunit['swissprot']['uniprot_id']
                                uniprot_ids.append(uniprot_id)
                                
                                # Extract gene name if available
                                if 'gene_name' in subunit['swissprot']:
                                    gene_names.append(subunit['swissprot']['gene_name'])
                    
                    mapping_stats['total_proteins'] += len(uniprot_ids)
                    
                    # Map UniProt IDs to STRING IDs
                    string_ids = []
                    unmapped_count = 0
                    not_in_network_count = 0
                    
                    for uniprot_id in uniprot_ids:
                        if uniprot_id in id_mappings['uniprot_to_string']:
                            string_id = id_mappings['uniprot_to_string'][uniprot_id]
                            mapping_stats['mapped_proteins'] += 1
                            
                            if string_id in ppi_network.nodes():
                                string_ids.append(string_id)
                            else:
                                not_in_network_count += 1
                                mapping_stats['proteins_not_in_network'] += 1
                        else:
                            unmapped_count += 1
                            mapping_stats['unmapped_proteins'] += 1
                    
                    # Log mapping issues if they exist
                    if unmapped_count > 0:
                        logger.debug(f"Complex {complex_id}: {unmapped_count}/{len(uniprot_ids)} proteins couldn't be mapped to STRING IDs")
                    
                    if not_in_network_count > 0:
                        logger.debug(f"Complex {complex_id}: {not_in_network_count} proteins mapped but not found in network")
                    
                    # Check if enough proteins mapped to network
                    if len(string_ids) >= min_complex_size:
                        # Create a subgraph for this complex
                        try:
                            complex_subgraph = ppi_network.subgraph(string_ids).copy()
                            
                            # Check if the subgraph is connected
                            if nx.is_connected(complex_subgraph):
                                # Extract GO terms if available
                                go_terms = []
                                if 'functions' in complex_data and isinstance(complex_data['functions'], list):
                                    for func in complex_data['functions']:
                                        if 'go' in func and 'name' in func['go']:
                                            go_name = func['go']['name']
                                            go_id = func['go'].get('go_id', '')
                                            go_type = func['go'].get('ontology', '')
                                            go_terms.append({
                                                'name': go_name,
                                                'id': go_id,
                                                'type': go_type
                                            })
                                
                                # Store complex data
                                complexes[f"{complex_id}"] = {
                                    'name': complex_name,
                                    'uniprot_ids': uniprot_ids,
                                    'gene_names': gene_names,
                                    'string_ids': string_ids,
                                    'subgraph': complex_subgraph,
                                    'size': len(string_ids),
                                    'go_terms': go_terms
                                }
                                stats['mapped_complexes'] += 1
                            else:
                                stats['disconnected_complexes'] += 1
                                logger.debug(f"Complex {complex_id}: Disconnected in network, skipping")
                        except Exception as e:
                            logger.warning(f"Error creating subgraph for complex {complex_id}: {str(e)}")
                    elif len(string_ids) == 1:
                        stats['singleton_complexes'] += 1
                        logger.debug(f"Complex {complex_id}: Only one protein mapped to network, skipping")
                    elif len(string_ids) < min_complex_size:
                        stats['too_small_complexes'] += 1
                        logger.debug(f"Complex {complex_id}: Not enough proteins ({len(string_ids)}/{min_complex_size}) mapped to network, skipping")
            except Exception as e:
                logger.warning(f"Error processing complex {complex_data.get('complex_id', 'unknown')}: {str(e)}")
                continue
                
        # Analyze size distribution of complexes
        if stats['mapped_complexes'] > 0:
            sizes = [data['size'] for data in complexes.values()]
            avg_size = np.mean(sizes) if sizes else 0
            size_distribution = {}
            
            for size in sizes:
                size_distribution[size] = size_distribution.get(size, 0) + 1
            
            # Log comprehensive statistics
            logger.info(f"Processed {stats['total_complexes']} total complexes from CORUM")
            logger.info(f"Found {stats['organism_complexes']} {organism_filter} complexes")
            logger.info(f"Successfully mapped {stats['mapped_complexes']} complexes to STRING network")
            
            logger.info(f"Complex size range: {min(sizes)} to {max(sizes)} proteins")
            logger.info(f"Average complex size: {avg_size:.2f} proteins")
            logger.info(f"Size distribution: {sorted(size_distribution.items())}")
        else:
            logger.warning("No protein complexes could be mapped to the network!")
        
        # Log mapping statistics
        if mapping_stats['total_proteins'] > 0:
            logger.info(f"Mapping statistics:")
            logger.info(f"  Total proteins in {organism_filter} complexes: {mapping_stats['total_proteins']}")
            logger.info(f"  Successfully mapped to STRING: {mapping_stats['mapped_proteins']} "
                      f"({mapping_stats['mapped_proteins']/mapping_stats['total_proteins']*100:.1f}%)")
            logger.info(f"  Unmapped proteins: {mapping_stats['unmapped_proteins']} "
                      f"({mapping_stats['unmapped_proteins']/mapping_stats['total_proteins']*100:.1f}%)")
            logger.info(f"  Mapped but not in network: {mapping_stats['proteins_not_in_network']} "
                      f"({mapping_stats['proteins_not_in_network']/mapping_stats['total_proteins']*100:.1f}%)")
        
        # Log filtering statistics
        logger.info(f"Filtering statistics:")
        logger.info(f"  Disconnected complexes: {stats['disconnected_complexes']}")
        logger.info(f"  Singleton complexes: {stats['singleton_complexes']}")
        logger.info(f"  Too small complexes (<{min_complex_size}): {stats['too_small_complexes']}")
        
    except FileNotFoundError:
        logger.error(f"CORUM file not found: {corum_json_file}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in CORUM file: {corum_json_file}")
        raise
    except Exception as e:
        logger.error(f"Error processing CORUM file: {str(e)}")
        raise
    
    return complexes


def process_go_annotations(mgi_file, id_mappings, ppi_network):
    """
    Process Gene Ontology annotations from MGI GAF file using goatools.
    
    Args:
        mgi_file: Path to MGI GAF file
        id_mappings: Dictionaries for ID conversion
        ppi_network: PPI network as NetworkX graph
        
    Returns:
        dict: Dictionary with GO term data by aspect
    """
    logger.info(f"Processing GO annotations from {mgi_file}")
    
    # Initialize dictionaries for GO terms by aspect
    go_terms = {'P': set(), 'F': set(), 'C': set()}  # Process, Function, Component
    gene_to_go = {}
    
    # Track statistics
    aspect_counts = {'P': 0, 'F': 0, 'C': 0, 'other': 0}
    gene_counts = set() # To count unique gene IDs from GAF
    
    try:
        # Initialize GafReader. For many GAF files, initialization itself reads the file
        # and populates the .associations attribute.
        gaf_reader = GafReader(mgi_file)
        
        # Get the detailed GAF annotation lines from the .associations attribute
        # This attribute should contain a list of ntgafobj (NamedTuple GAF Object)
        gaf_data_list = gaf_reader.associations 
        
        if not gaf_data_list or not isinstance(gaf_data_list, list):
            logger.error("Failed to load GAF data as a list from gaf_reader.associations. Data is not a list or is empty.")
            # You might want to raise an error or return an empty structure depending on desired behavior
            raise ValueError("gaf_reader.associations did not contain the expected list of annotations.")
            
        # Optional: Check the first element to ensure it's in the expected format (ntgafobj)
        if not (hasattr(gaf_data_list[0], 'DB') and hasattr(gaf_data_list[0], 'NS') and hasattr(gaf_data_list[0], 'GO_ID')):
            logger.error("First element in gaf_reader.associations does not appear to be a valid GAF annotation object (ntgafobj).")
            logger.error(f"Type: {type(gaf_data_list[0])}, Content (partial): {str(gaf_data_list[0])[:200]}")
            raise ValueError("Data from gaf_reader.associations is not in the expected ntgafobj format.")

        logger.info(f"Loaded {len(gaf_data_list)} detailed GO annotation lines from GAF file (via .associations)")
        
        # Mapping from GAF Namespace (NS) to your internal aspect codes
        gaf_ns_to_internal_aspect = {'BP': 'P', 'MF': 'F', 'CC': 'C'}

        # Process the GAF data (list of ntgafobj)
        for annotation in tqdm(gaf_data_list, desc="Processing GO annotations"):
            db = annotation.DB
            gene_id = annotation.DB_ID   # This is the gene/protein ID from the GAF file (e.g., MGI:xxxx)
            go_id = annotation.GO_ID
            gaf_namespace = annotation.NS  # 'BP', 'MF', or 'CC'
            
            # Skip non-MGI entries if needed (though mgi.gaf should only contain MGI)
            if db != 'MGI':
                continue
            
            gene_counts.add(gene_id) # Add the GAF's gene ID to our set
            
            internal_aspect_code = gaf_ns_to_internal_aspect.get(gaf_namespace)
            
            if internal_aspect_code: # Check if the namespace mapped successfully to 'P', 'F', or 'C'
                if internal_aspect_code in go_terms: # Ensure the mapped code is a valid key
                    go_terms[internal_aspect_code].add(go_id)
                    aspect_counts[internal_aspect_code] += 1
                
                # Initialize gene entry if needed
                if gene_id not in gene_to_go:
                    gene_to_go[gene_id] = {'P': set(), 'F': set(), 'C': set()}
                
                # Add GO term to gene's annotations
                if internal_aspect_code in gene_to_go[gene_id]: # Ensure the mapped code is a valid key
                    gene_to_go[gene_id][internal_aspect_code].add(go_id)
            else:
                aspect_counts['other'] += 1
                logger.debug(f"Unknown or unmapped GAF namespace '{gaf_namespace}' for GO ID {go_id} (Gene: {gene_id})")
                
    except FileNotFoundError:
        logger.error(f"MGI GAF file not found: {mgi_file}")
        raise
    except Exception as e:
        # Log the full traceback for better debugging
        logger.error(f"Error processing MGI GAF file: {str(e)}", exc_info=True)
        raise
    
    # Create go term index mapping for each aspect
    go_indices = {}
    for aspect_key in go_terms: # aspect_key will be 'P', 'F', 'C'
        go_list = sorted(list(go_terms[aspect_key]))
        go_indices[aspect_key] = {
            'terms': go_list,
            'mapping': {go: i for i, go in enumerate(go_list)}
        }
    
    go_data_processed = {
        'gene_to_go': gene_to_go, # Maps GAF DB_ID to its GO terms
        'indices': go_indices
    }
    
    # Log detailed statistics
    logger.info(f"Processed GO annotations for {len(gene_counts)} unique gene IDs (from GAF DB_ID field)")
    
    aspect_name_map = {'P': "Biological Process", 'F': "Molecular Function", 'C': "Cellular Component"}
    for aspect_internal_key, terms_set in go_terms.items():
        aspect_display_name = aspect_name_map.get(aspect_internal_key, "Unknown Aspect")
        logger.info(f"  {aspect_display_name} ({aspect_internal_key}): {len(terms_set)} unique GO terms, {aspect_counts[aspect_internal_key]} annotations")
    
    if aspect_counts['other'] > 0:
        logger.warning(f"Found {aspect_counts['other']} annotations with unmapped or unknown GAF namespaces")
    
    # Validation: Check overlap between GO-annotated genes (from GAF's DB_ID) and genes in the PPI network
    # This requires mapping STRING IDs from ppi_network to the same type of ID as GAF's DB_ID
    network_gene_ids_comparable_to_gaf = set()
    string_to_comparable_gene_id_map = id_mappings.get('string_to_mgi', {}) # Assuming this map provides IDs comparable to GAF's DB_ID
    
    for string_node_id in ppi_network.nodes():
        comparable_gene_id = string_to_comparable_gene_id_map.get(string_node_id)
        if comparable_gene_id:
            network_gene_ids_comparable_to_gaf.add(comparable_gene_id)
    
    if gene_counts: # Avoid division by zero if gene_counts is empty
        # gene_counts contains DB_IDs from GAF (e.g., MGI:xxxxxx)
        # network_gene_ids_comparable_to_gaf contains IDs mapped from STRING network nodes
        genes_in_network_and_gaf = gene_counts.intersection(network_gene_ids_comparable_to_gaf)
        percentage_in_network = (len(genes_in_network_and_gaf) / len(gene_counts)) * 100 if len(gene_counts) > 0 else 0
        logger.info(
            f"GO-annotated genes (from GAF DB_ID) also found in PPI network "
            f"(after mapping STRING IDs): {len(genes_in_network_and_gaf)} out of {len(gene_counts)} "
            f"({percentage_in_network:.1f}%)"
        )
        
        if len(genes_in_network_and_gaf) == 0 and len(gene_counts) > 0:
            logger.warning(
                "No overlap found between GO-annotated genes (from GAF) and genes in the PPI network. "
                "Check if `id_mappings['string_to_mgi']` correctly maps STRING IDs to IDs comparable "
                "with GAF DB_IDs (e.g., MGI IDs)."
            )
    else:
        logger.info("No gene IDs were successfully processed from the GAF file to perform network overlap analysis.")
        
    # Calculate distribution of annotations per gene (using GAF DB_IDs)
    annotations_per_gaf_gene = []
    if gene_to_go: # Check if gene_to_go is not empty
      annotations_per_gaf_gene = [sum(len(go_id_set) for go_id_set in aspect_dict.values()) for aspect_dict in gene_to_go.values()]
    
    if annotations_per_gaf_gene:
        logger.info(
            f"Annotations per GAF gene (DB_ID): "
            f"min={min(annotations_per_gaf_gene)}, "
            f"max={max(annotations_per_gaf_gene)}, "
            f"avg={np.mean(annotations_per_gaf_gene):.1f}"
        )
    else:
        logger.info("No annotation counts per gene to report (gene_to_go might be empty).")
        
    return go_data_processed


def export_network_data(ppi_network, protein_info, complexes, output_dir):
    """
    Export network and complex data to files.
    
    Args:
        ppi_network: NetworkX graph of PPI network
        protein_info: Dictionary of protein information
        complexes: Dictionary of protein complexes
        output_dir: Directory to save output files
    """
    logger.info(f"Exporting network data to {output_dir}")
    
    # Create output directory if it doesn't exist
    try:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created or verified output directory: {output_dir}")
    except Exception as e:
        logger.error(f"Failed to create output directory: {str(e)}")
        raise
    
    # Track export statistics
    files_written = 0
    
    # Export network as edge list
    try:
        edge_file = os.path.join(output_dir, "ppi_network.edgelist")
        with open(edge_file, "w", encoding='utf-8') as f:
            for u, v, data in ppi_network.edges(data=True):
                f.write(f"{u}\t{v}\t{data['weight']}\n")
        logger.info(f"Exported network edge list to {edge_file} ({ppi_network.number_of_edges()} edges)")
        files_written += 1
    except Exception as e:
        logger.error(f"Failed to export network edge list: {str(e)}")
    
    # Export protein info
    try:
        protein_file = os.path.join(output_dir, "protein_info.tsv")
        exported_count = 0
        with open(protein_file, "w", encoding='utf-8') as f:
            f.write("string_id\tname\tsize\n")
            for string_id, info in protein_info.items():
                if string_id in ppi_network.nodes():
                    f.write(f"{string_id}\t{info['name']}\t{info['size']}\n")
                    exported_count += 1
        logger.info(f"Exported protein info to {protein_file} ({exported_count} proteins)")
        files_written += 1
        
        # Validation
        if exported_count < ppi_network.number_of_nodes():
            logger.warning(f"Only {exported_count} out of {ppi_network.number_of_nodes()} network nodes have protein info")
    except Exception as e:
        logger.error(f"Failed to export protein info: {str(e)}")
    
    # Export complexes
    try:
        complex_dir = os.path.join(output_dir, "complexes")
        os.makedirs(complex_dir, exist_ok=True)
        logger.info(f"Created or verified complex directory: {complex_dir}")
        
        complex_file = os.path.join(output_dir, "complex_summary.tsv")
        with open(complex_file, "w", encoding='utf-8') as f:
            f.write("complex_id\tname\tsize\n")
            for complex_id, data in complexes.items():
                f.write(f"{complex_id}\t{data['name']}\t{data['size']}\n")
                
                # Export each complex as a separate edge list
                try:
                    c_file = os.path.join(complex_dir, f"{complex_id}.edgelist")
                    with open(c_file, "w", encoding='utf-8') as cf:
                        for u, v, data in data['subgraph'].edges(data=True):
                            cf.write(f"{u}\t{v}\t{data['weight']}\n")
                    files_written += 1
                except Exception as e:
                    logger.warning(f"Failed to export complex {complex_id}: {str(e)}")
        
        logger.info(f"Exported complex summary to {complex_file} ({len(complexes)} complexes)")
        logger.info(f"Exported {len(complexes)} individual complex networks to {complex_dir}")
        files_written += 1
    except Exception as e:
        logger.error(f"Failed to export complexes: {str(e)}")
    
    # Export node IDs
    try:
        node_file = os.path.join(output_dir, "node_ids.txt")
        with open(node_file, "w", encoding='utf-8') as f:
            for node in ppi_network.nodes():
                f.write(f"{node}\n")
        logger.info(f"Exported node IDs to {node_file} ({ppi_network.number_of_nodes()} nodes)")
        files_written += 1
    except Exception as e:
        logger.error(f"Failed to export node IDs: {str(e)}")
    
    # Save network statistics
    try:
        stats_file = os.path.join(output_dir, "network_stats.txt")
        with open(stats_file, "w", encoding='utf-8') as f:
            f.write(f"Network Statistics\n")
            f.write(f"=================\n\n")
            f.write(f"Number of proteins (nodes): {ppi_network.number_of_nodes()}\n")
            f.write(f"Number of interactions (edges): {ppi_network.number_of_edges()}\n")
            f.write(f"Network density: {nx.density(ppi_network):.6f}\n")
            
            # Calculate average degree
            degrees = [d for n, d in ppi_network.degree()]
            f.write(f"Average degree: {np.mean(degrees):.2f}\n")
            f.write(f"Degree range: {min(degrees)} to {max(degrees)}\n")
            
            # Degree distribution quartiles
            q1, q2, q3 = np.percentile(degrees, [25, 50, 75])
            f.write(f"Degree quartiles: Q1={q1}, Median={q2}, Q3={q3}\n")
            
            # Calculate clustering coefficient
            f.write(f"Average clustering coefficient: {nx.average_clustering(ppi_network):.4f}\n")
            
            # Count connected components
            components = list(nx.connected_components(ppi_network))
            f.write(f"Number of connected components: {len(components)}\n")
            largest_component = max(components, key=len)
            f.write(f"Size of largest component: {len(largest_component)} nodes ({len(largest_component)/ppi_network.number_of_nodes()*100:.1f}% of network)\n")
            
            # Complex statistics
            f.write(f"\nProtein Complexes:\n")
            f.write(f"=================\n\n")
            f.write(f"Total number of complexes: {len(complexes)}\n")
            
            if complexes:
                sizes = [data['size'] for data in complexes.values()]
                f.write(f"Average complex size: {np.mean(sizes):.2f} proteins\n")
                f.write(f"Min complex size: {min(sizes)} proteins\n")
                f.write(f"Max complex size: {max(sizes)} proteins\n")
                
                # Size distribution
                size_counts = {}
                for size in sizes:
                    size_counts[size] = size_counts.get(size, 0) + 1
                
                f.write(f"\nComplex size distribution:\n")
                for size in sorted(size_counts.keys()):
                    f.write(f"  {size} proteins: {size_counts[size]} complexes\n")
        
        logger.info(f"Exported network statistics to {stats_file}")
        files_written += 1
    except Exception as e:
        logger.error(f"Failed to export network statistics: {str(e)}")
    
    logger.info(f"Successfully exported {files_written} files to {output_dir}")
    
def main(data_dir="data/raw", output_dir="data/processed", confidence_threshold=700, log_file=None, log_level=logging.INFO):
    """
    Main preprocessing workflow function.
    
    Args:
        data_dir: Directory containing raw data files
        output_dir: Directory to save processed data
        confidence_threshold: Minimum confidence score for PPI interactions
        log_file: Optional log file path
        log_level: Logging level
    """
    # Setup logging
    global logger
    logger = setup_logger(log_file=log_file, log_level=log_level)
    
    # Log main execution parameters
    logger.info("="*80)
    logger.info("Starting PPI network preprocessing")
    logger.info("="*80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Confidence threshold: {confidence_threshold}")
    
    try:
        # Define file paths
        aliases_file = os.path.join(data_dir, "10090.protein.aliases.v12.0.txt")
        info_file = os.path.join(data_dir, "10090.protein.info.v12.0.txt")
        links_file = os.path.join(data_dir, "10090.protein.links.v12.0.txt")
        corum_file = os.path.join(data_dir, "corum_allComplexes.json")
        mgi_file = os.path.join(data_dir, "mgi.gaf")
        
        # Check if files exist
        missing_files = []
        for f in [aliases_file, info_file, links_file, corum_file, mgi_file]:
            if not os.path.exists(f):
                missing_files.append(f)
        
        if missing_files:
            logger.error(f"Missing input files: {missing_files}")
            return False
        
        # Process data
        start_time = time.time()
        
        logger.info("Step 1/5: Processing protein aliases")
        id_mappings = process_aliases(aliases_file)
        
        logger.info("Step 2/5: Processing protein information")
        protein_info = process_protein_info(info_file)
        
        logger.info("Step 3/5: Building PPI network")
        ppi_network = build_ppi_network(links_file, confidence_threshold=confidence_threshold)
        
        logger.info("Step 4/5: Processing protein complexes")
        complexes = process_corum_complexes(corum_file, id_mappings, ppi_network)
        
        logger.info("Step 5/5: Processing GO annotations")
        go_data = process_go_annotations(mgi_file, id_mappings, ppi_network)
        
        # Export data
        logger.info("Exporting processed data")
        export_network_data(ppi_network, protein_info, complexes, output_dir)
        
        # Save processed data as pickle files for easier loading
        try:
            import pickle
            
            # Save ID mappings
            with open(os.path.join(output_dir, "id_mappings.pkl"), "wb") as f:
                pickle.dump(id_mappings, f)
            
            # Save GO data
            with open(os.path.join(output_dir, "go_data.pkl"), "wb") as f:
                pickle.dump(go_data, f)
            
            # Can't directly pickle NetworkX graph with node objects
            # Save as GraphML instead
            nx.write_graphml(ppi_network, os.path.join(output_dir, "ppi_network.graphml"))
            
            logger.info("Saved processed data as binary files for faster loading")
        except Exception as e:
            logger.warning(f"Could not save binary data files: {str(e)}")
        
        # Log execution time
        end_time = time.time()
        logger.info(f"Preprocessing completed in {end_time - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        logger.exception("Exception details:")
        return False


if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description="Process mouse PPI network data")
    parser.add_argument("--data-dir", default="data/raw", help="Directory containing raw data files")
    parser.add_argument("--output-dir", default="data/processed", help="Directory to save processed data")
    parser.add_argument("--threshold", type=int, default=700, help="Confidence threshold for interactions (0-1000)")
    parser.add_argument("--log-file", default=None, help="Log file path (default: log to console only)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    log_level = logging.DEBUG if args.verbose else logging.INFO
    
    success = main(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        confidence_threshold=args.threshold,
        log_file=args.log_file,
        log_level=log_level
    )
    
    sys.exit(0 if success else 1)