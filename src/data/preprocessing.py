"""
Data preprocessing module for mouse PPI network data.
Handles loading and basic cleaning of STRING, CORUM, and GO data.
"""

import os
import networkx as nx
import numpy as np
from tqdm import tqdm


def process_aliases(aliases_file):
    """
    Process STRING aliases file to create mapping dictionaries between different ID types.
    
    Args:
        aliases_file: Path to STRING aliases file
        
    Returns:
        dict: Dictionary of mapping dictionaries between ID types
    """
    print("Processing protein aliases...")
    
    # Initialize mapping dictionaries
    string_to_uniprot = {}
    string_to_gene = {}
    string_to_ensembl = {}
    uniprot_to_string = {}
    gene_to_string = {}
    
    # Read file and create mappings
    with open(aliases_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line in tqdm(f):
            if line.startswith('#'):
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
                
            string_id, alias, source = parts
            
            # Map STRING IDs to various identifier types
            if 'UniProt' in source:
                string_to_uniprot[string_id] = alias
                uniprot_to_string[alias] = string_id
            elif source == 'Ensembl':
                string_to_ensembl[string_id] = alias
            elif source in ['KEGG_GENEID', 'KEGG_KEGGID_SHORT']:
                string_to_gene[string_id] = alias
                gene_to_string[alias] = string_id
                
    # Combine all mappings
    id_mappings = {
        'string_to_uniprot': string_to_uniprot,
        'string_to_gene': string_to_gene,
        'string_to_ensembl': string_to_ensembl,
        'uniprot_to_string': uniprot_to_string,
        'gene_to_string': gene_to_string
    }
    
    print(f"Created {len(string_to_uniprot)} STRING to UniProt mappings")
    print(f"Created {len(string_to_gene)} STRING to gene mappings")
    
    return id_mappings


def process_protein_info(info_file):
    """
    Process STRING protein info file to extract protein names and metadata.
    
    Args:
        info_file: Path to STRING protein info file
        
    Returns:
        dict: Dictionary with STRING IDs as keys and protein info as values
    """
    print("Processing protein information...")
    
    protein_info = {}
    
    with open(info_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line in tqdm(f):
            if line.startswith('#'):
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 4:
                continue
                
            string_id, name, size, annotation = parts
            
            # Store protein information
            protein_info[string_id] = {
                'name': name,
                'size': int(size),
                'annotation': annotation
            }
    
    print(f"Processed information for {len(protein_info)} proteins")
    
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
    print("Building PPI network...")
    
    # Create empty graph
    G = nx.Graph()
    
    # Count statistics
    total_edges = 0
    filtered_edges = 0
    
    with open(links_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line in tqdm(f):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
                
            protein1, protein2, score = parts
            score = int(score)
            total_edges += 1
            
            # Filter by confidence score
            if score >= confidence_threshold:
                # Add nodes and edge to graph
                G.add_edge(protein1, protein2, weight=score/1000.0)  # Normalize to [0,1]
                filtered_edges += 1
    
    print(f"Processed {total_edges} total interactions")
    print(f"Built network with {G.number_of_nodes()} proteins and {G.number_of_edges()} high-confidence interactions")
    print(f"Filtered out {total_edges - filtered_edges} low-confidence interactions (< {confidence_threshold/1000.0})")
    
    return G


def process_corum_complexes(corum_file, id_mappings, ppi_network):
    """
    Extract protein complexes from CORUM database and map to STRING network.
    
    Args:
        corum_file: Path to CORUM complexes file
        id_mappings: Dictionaries for ID conversion
        ppi_network: PPI network as NetworkX graph
        
    Returns:
        dict: Dictionary of complex subgraphs
    """
    print("Processing CORUM protein complexes...")
    
    complexes = {}
    mouse_complexes = 0
    mapped_complexes = 0
    
    # Read CORUM file
    with open(corum_file, 'r', encoding='utf-8') as f:
        # Skip header
        next(f)
        
        for line in tqdm(f):
            parts = line.strip().split('\t')
            if len(parts) < 15:  # Ensure we have enough columns
                continue
                
            complex_id = parts[0]
            complex_name = parts[1]
            organism = parts[3]
            uniprot_ids = parts[11].split(';') if parts[11] else []
            gene_names = parts[12].split(';') if parts[12] else []
            
            # Filter for mouse complexes (10090 is NCBI taxonomy ID for mouse)
            if "Mouse" in organism or "10090" in organism:
                mouse_complexes += 1
                
                # Map UniProt IDs to STRING IDs
                string_ids = []
                for uniprot_id in uniprot_ids:
                    if uniprot_id in id_mappings['uniprot_to_string']:
                        string_id = id_mappings['uniprot_to_string'][uniprot_id]
                        if string_id in ppi_network.nodes():
                            string_ids.append(string_id)
                
                # Only keep complexes with at least 3 proteins that exist in our network
                if len(string_ids) >= 3:
                    # Create a subgraph for this complex
                    complex_subgraph = ppi_network.subgraph(string_ids).copy()
                    
                    # Only keep connected complexes
                    if nx.is_connected(complex_subgraph):
                        complexes[complex_id] = {
                            'name': complex_name,
                            'uniprot_ids': uniprot_ids,
                            'gene_names': gene_names,
                            'string_ids': string_ids,
                            'subgraph': complex_subgraph,
                            'size': len(string_ids)
                        }
                        mapped_complexes += 1
    
    # Analyze size distribution
    sizes = [data['size'] for data in complexes.values()]
    avg_size = np.mean(sizes) if sizes else 0
    
    print(f"Found {mouse_complexes} mouse complexes in CORUM")
    print(f"Successfully mapped {mapped_complexes} complexes to STRING network")
    print(f"Average complex size: {avg_size:.2f} proteins")
    
    return complexes


def process_go_annotations(mgi_file, id_mappings, ppi_network):
    """
    Process Gene Ontology annotations from MGI GAF file.
    
    Args:
        mgi_file: Path to MGI GAF file
        id_mappings: Dictionaries for ID conversion
        ppi_network: PPI network as NetworkX graph
        
    Returns:
        dict: Dictionary with GO term data by aspect
    """
    print("Processing GO annotations...")
    
    # Initialize dictionaries for GO terms by aspect
    go_terms = {'P': set(), 'F': set(), 'C': set()}  # Process, Function, Component
    gene_to_go = {}
    
    # Read MGI GAF file
    with open(mgi_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            # Skip comment lines
            if line.startswith('!'):
                continue
                
            parts = line.strip().split('\t')
            if len(parts) < 10:
                continue
                
            # Extract relevant fields
            db, gene_id, gene_symbol = parts[0], parts[1], parts[2]
            relation, go_id = parts[3], parts[4]
            aspect = parts[8]  # P, F, or C
            
            # Skip non-MGI entries
            if db != 'MGI':
                continue
                
            # Add GO term to the appropriate aspect set
            if aspect in go_terms:
                go_terms[aspect].add(go_id)
            
            # Initialize gene entry if needed
            if gene_id not in gene_to_go:
                gene_to_go[gene_id] = {'P': set(), 'F': set(), 'C': set()}
            
            # Add GO term to gene's annotations
            if aspect in gene_to_go[gene_id]:
                gene_to_go[gene_id][aspect].add(go_id)
    
    # Create go term index mapping for each aspect
    go_indices = {}
    for aspect in go_terms:
        go_list = sorted(list(go_terms[aspect]))
        go_indices[aspect] = {
            'terms': go_list,
            'mapping': {go: i for i, go in enumerate(go_list)}
        }
    
    go_data = {
        'gene_to_go': gene_to_go,
        'indices': go_indices
    }
    
    print(f"Processed GO annotations:")
    for aspect, terms in go_terms.items():
        aspect_name = "Biological Process" if aspect == 'P' else "Molecular Function" if aspect == 'F' else "Cellular Component"
        print(f"  {aspect_name}: {len(terms)} unique terms")
    
    return go_data


def export_network_data(ppi_network, protein_info, complexes, output_dir):
    """
    Export network and complex data to files.
    
    Args:
        ppi_network: NetworkX graph of PPI network
        protein_info: Dictionary of protein information
        complexes: Dictionary of protein complexes
        output_dir: Directory to save output files
    """
    print("Exporting network data...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Export network as edge list
    with open(os.path.join(output_dir, "ppi_network.edgelist"), "w", encoding='utf-8') as f:
        for u, v, data in ppi_network.edges(data=True):
            f.write(f"{u}\t{v}\t{data['weight']}\n")
    
    # Export protein info
    with open(os.path.join(output_dir, "protein_info.tsv"), "w", encoding='utf-8') as f:
        f.write("string_id\tname\tsize\n")
        for string_id, info in protein_info.items():
            if string_id in ppi_network.nodes():
                f.write(f"{string_id}\t{info['name']}\t{info['size']}\n")
    
    # Export complexes
    complex_dir = os.path.join(output_dir, "complexes")
    os.makedirs(complex_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "complex_summary.tsv"), "w", encoding='utf-8') as f:
        f.write("complex_id\tname\tsize\n")
        for complex_id, data in complexes.items():
            f.write(f"{complex_id}\t{data['name']}\t{data['size']}\n")
            
            # Export each complex as a separate edge list
            with open(os.path.join(complex_dir, f"{complex_id}.edgelist"), "w", encoding='utf-8') as cf:
                for u, v, data in data['subgraph'].edges(data=True):
                    cf.write(f"{u}\t{v}\t{data['weight']}\n")
    
    # Export node IDs
    with open(os.path.join(output_dir, "node_ids.txt"), "w", encoding='utf-8') as f:
        for node in ppi_network.nodes():
            f.write(f"{node}\n")
    
    # Save network statistics
    with open(os.path.join(output_dir, "network_stats.txt"), "w", encoding='utf-8') as f:
        f.write(f"Number of proteins (nodes): {ppi_network.number_of_nodes()}\n")
        f.write(f"Number of interactions (edges): {ppi_network.number_of_edges()}\n")
        f.write(f"Network density: {nx.density(ppi_network):.6f}\n")
        
        # Calculate average degree
        degrees = [d for n, d in ppi_network.degree()]
        f.write(f"Average degree: {np.mean(degrees):.2f}\n")
        
        # Calculate clustering coefficient
        f.write(f"Average clustering coefficient: {nx.average_clustering(ppi_network):.4f}\n")
        
        # Count connected components
        components = list(nx.connected_components(ppi_network))
        f.write(f"Number of connected components: {len(components)}\n")
        f.write(f"Size of largest component: {len(max(components, key=len))}\n")
        
        # Complex statistics
        f.write(f"\nProtein Complexes:\n")
        f.write(f"Total number of complexes: {len(complexes)}\n")
        sizes = [data['size'] for data in complexes.values()]
        f.write(f"Average complex size: {np.mean(sizes):.2f} proteins\n")
        f.write(f"Min complex size: {min(sizes)} proteins\n")
        f.write(f"Max complex size: {max(sizes)} proteins\n")
    
    print(f"Exported network data to {output_dir}")