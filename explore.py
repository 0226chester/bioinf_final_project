#!/usr/bin/env python3
"""
Combined Protein-Protein Interaction Network Analyzer
Processes PPI data and generates comprehensive visualizations and statistics.
"""

import os
import argparse
import logging
import pandas as pd
import networkx as nx
from Bio import SeqIO
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter, defaultdict
from tqdm import tqdm
import math
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from community import community_louvain

# Configure logging (console only)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("ppi_analyzer")

class PPIAnalyzer:
    def __init__(self, 
                 ppi_file="4932.protein.links.v12.0.txt", 
                 info_file="4932.protein.info.v12.0.txt",
                 alias_file="4932.protein.aliases.v12.0.txt",
                 fasta_file="UP000002311_559292.fasta",
                 gaf_file="sgd.gaf",
                 confidence_threshold=700,
                 output_dir="visualization"):
        """
        Initialize the PPI network analyzer.
        
        Args:
            ppi_file: File containing protein-protein interactions
            info_file: File containing protein information
            alias_file: File containing protein aliases/ID mappings
            fasta_file: FASTA file containing protein sequences
            gaf_file: GAF file containing GO annotations
            confidence_threshold: Minimum confidence score for interactions (0-1000)
            output_dir: Directory to save visualizations
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
        
        # Data containers (in-memory only)
        self.graph = None
        self.largest_cc = None
        self.protein_sequences = {}
        self.protein_to_go = {}
        self.string_to_alias = {}
        self.alias_to_string = {}
        self.protein_info = None
        self.interactions_df = None
        self.network_stats = {}
        
        # Visualization settings
        self.colors = plt.cm.tab10.colors
        self.palettes = {
            'main': sns.color_palette('tab10', 10),
            'pastel': sns.color_palette('pastel', 10),
            'degree': sns.color_palette('YlOrRd', 9),
            'clustering': sns.color_palette('YlGnBu', 9),
            'go_terms': sns.color_palette('RdPu', 9)
        }
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig_size = (12, 10)
        self.font_size = 12
    
    def process_protein_links(self):
        """Process protein-protein interaction data with confidence filtering."""
        start_time = time.time()
        logger.info(f"Processing protein links from {self.ppi_file}")
        logger.info(f"Using confidence threshold: {self.confidence_threshold}")
        
        try:
            df = pd.read_csv(self.ppi_file, sep=' ')
            logger.info(f"Loaded {len(df)} raw interactions")
            
            # Handle column names
            if 'protein1' not in df.columns and 'protein2' not in df.columns:
                if '#' in df.columns[0]:
                    new_cols = [col.replace('#', '') for col in df.columns]
                    df.columns = new_cols
            
            # Filter by confidence score
            filtered_df = df[df['combined_score'] >= self.confidence_threshold].copy()
            logger.info(f"Filtered to {len(filtered_df)} interactions with confidence >= {self.confidence_threshold}")
            
            # Strip organism prefix
            filtered_df.loc[:, 'protein1'] = filtered_df['protein1'].str.replace('4932.', '')
            filtered_df.loc[:, 'protein2'] = filtered_df['protein2'].str.replace('4932.', '')
            
            self.interactions_df = filtered_df
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
            df = pd.read_csv(self.info_file, sep='\t')
            logger.info(f"Loaded information for {len(df)} proteins")
            
            # Find protein ID column
            protein_id_column = None
            for col in ['protein_external_id', '#string_protein_id', 'string_protein_id', 'protein_id']:
                if col in df.columns:
                    protein_id_column = col
                    break
            
            if protein_id_column:
                df[protein_id_column] = df[protein_id_column].str.replace('4932.', '')
            
            self.protein_info = df
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
            df = pd.read_csv(self.alias_file, sep='\t')
            logger.info(f"Loaded {len(df)} alias entries")
            
            # Identify columns
            protein_id_column = None
            alias_column = None
            source_column = None
            
            for col in ['string_protein_id', '#string_id', 'protein_id', 'STRING_id', '#protein_external_id']:
                if col in df.columns:
                    protein_id_column = col
                    break
            
            for col in ['alias', 'Alias', 'external_id']:
                if col in df.columns:
                    alias_column = col
                    break
            
            for col in ['source', 'Source', 'alias_source']:
                if col in df.columns:
                    source_column = col
                    break
            
            # Use first columns if not found
            if not protein_id_column and len(df.columns) > 0:
                protein_id_column = df.columns[0]
            if not alias_column and len(df.columns) > 1:
                alias_column = df.columns[1]
            if not source_column and len(df.columns) > 2:
                source_column = df.columns[2]
            
            if protein_id_column and alias_column:
                # Strip organism prefix
                if '4932.' in str(df[protein_id_column].iloc[0]):
                    df[protein_id_column] = df[protein_id_column].str.replace('4932.', '')
                
                # Build mappings
                string_to_alias = {}
                alias_to_string = {}
                
                for _, row in df.iterrows():
                    string_id = row[protein_id_column]
                    alias = row[alias_column]
                    source = row[source_column] if source_column else "unknown"
                    
                    if string_id not in string_to_alias:
                        string_to_alias[string_id] = {}
                    if source not in string_to_alias[string_id]:
                        string_to_alias[string_id][source] = []
                    string_to_alias[string_id][source].append(alias)
                    
                    if source not in alias_to_string:
                        alias_to_string[source] = {}
                    alias_to_string[source][alias] = string_id
                
                self.string_to_alias = string_to_alias
                self.alias_to_string = alias_to_string
                
                elapsed = time.time() - start_time
                logger.info(f"Processed protein aliases in {elapsed:.2f} seconds")
                return string_to_alias, alias_to_string
            else:
                logger.error(f"Could not identify required columns. Found: {df.columns.tolist()}")
                raise ValueError(f"Missing required columns. Found: {df.columns.tolist()}")
            
        except Exception as e:
            logger.error(f"Error processing protein aliases: {str(e)}")
            raise
    
    def process_fasta_sequences(self):
        """Process protein sequences from FASTA file."""
        start_time = time.time()
        logger.info(f"Processing protein sequences from {self.fasta_file}")
        
        try:
            sequences = {}
            with open(self.fasta_file, 'r') as handle:
                for record in SeqIO.parse(handle, 'fasta'):
                    uniprot_id = record.id.split('|')[1] if '|' in record.id else record.id
                    sequence = str(record.seq)
                    sequences[uniprot_id] = sequence
            
            self.protein_sequences = sequences
            logger.info(f"Loaded {len(sequences)} protein sequences")
            
            elapsed = time.time() - start_time
            logger.info(f"Processed protein sequences in {elapsed:.2f} seconds")
            return sequences
            
        except Exception as e:
            logger.error(f"Error processing protein sequences: {str(e)}")
            raise
    
    def process_go_annotations(self):
        """Process Gene Ontology annotations from GAF file."""
        start_time = time.time()
        logger.info(f"Processing GO annotations from {self.gaf_file}")
        
        try:
            columns = [
                'DB', 'DB_Object_ID', 'DB_Object_Symbol', 'Qualifier', 'GO_ID',
                'DB_Reference', 'Evidence_Code', 'With_From', 'Aspect',
                'DB_Object_Name', 'DB_Object_Synonym', 'DB_Object_Type',
                'Taxon', 'Date', 'Assigned_By', 'Annotation_Extension',
                'Gene_Product_Form_ID'
            ]
            
            gaf_df = pd.read_csv(self.gaf_file, sep='\t', comment='!', 
                                header=None, names=columns, low_memory=False)
            logger.info(f"Loaded {len(gaf_df)} GO annotations")
            
            # Create mappings
            protein_to_go = {}
            go_to_protein = {}
            
            for _, row in gaf_df.iterrows():
                protein_id = row['DB_Object_ID']
                go_id = row['GO_ID']
                aspect = row['Aspect']
                
                if protein_id not in protein_to_go:
                    protein_to_go[protein_id] = {'P': set(), 'F': set(), 'C': set()}
                if go_id not in go_to_protein:
                    go_to_protein[go_id] = {'P': set(), 'F': set(), 'C': set()}
                
                protein_to_go[protein_id][aspect].add(go_id)
                go_to_protein[go_id][aspect].add(protein_id)
            
            self.protein_to_go = protein_to_go
            
            elapsed = time.time() - start_time
            logger.info(f"Processed GO annotations in {elapsed:.2f} seconds")
            return protein_to_go, go_to_protein, gaf_df
            
        except Exception as e:
            logger.error(f"Error processing GO annotations: {str(e)}")
            raise
    
    def create_network(self):
        """Create NetworkX graph from interaction data."""
        start_time = time.time()
        logger.info("Creating NetworkX graph from filtered interactions")
        
        try:
            G = nx.Graph()
            
            for _, row in self.interactions_df.iterrows():
                protein1 = row['protein1']
                protein2 = row['protein2']
                score = row['combined_score']
                G.add_edge(protein1, protein2, weight=score, confidence=score)
            
            logger.info(f"Created graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
            
            self.graph = G
            elapsed = time.time() - start_time
            logger.info(f"Created network in {elapsed:.2f} seconds")
            return G
            
        except Exception as e:
            logger.error(f"Error creating network: {str(e)}")
            raise
    
    def analyze_network(self):
        """Perform basic network analysis."""
        logger.info("Performing basic network analysis")
        
        try:
            G = self.graph
            connected_components = list(nx.connected_components(G))
            largest_cc = max(connected_components, key=len)
            largest_cc_graph = G.subgraph(largest_cc).copy()
            
            logger.info(f"Network has {len(connected_components)} connected components")
            logger.info(f"Largest connected component has {largest_cc_graph.number_of_nodes()} nodes "
                      f"and {largest_cc_graph.number_of_edges()} edges")
            
            # Calculate network statistics
            stats = {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'components': len(connected_components),
                'largest_cc_nodes': largest_cc_graph.number_of_nodes(),
                'largest_cc_edges': largest_cc_graph.number_of_edges(),
                'average_clustering': nx.average_clustering(G),
            }
            
            self.largest_cc = largest_cc_graph
            self.network_stats = stats
            
            logger.info("Completed basic network analysis")
            return stats, largest_cc_graph
            
        except Exception as e:
            logger.error(f"Error analyzing network: {str(e)}")
            raise
    
    def visualize_network_overview(self):
        """Create basic visualizations of the network structure."""
        logger.info("Generating network overview visualizations")
        
        # 1. Degree Distribution
        plt.figure(figsize=(10, 6))
        degrees = [d for n, d in self.graph.degree()]
        sns.histplot(degrees, kde=True, bins=50)
        plt.title('Node Degree Distribution', fontsize=self.font_size+2)
        plt.xlabel('Degree', fontsize=self.font_size)
        plt.ylabel('Count', fontsize=self.font_size)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'degree_distribution.png'), dpi=300)
        plt.close()
        
        # 2. Clustering Coefficient Distribution
        plt.figure(figsize=(10, 6))
        clustering = list(nx.clustering(self.graph).values())
        sns.histplot(clustering, kde=True, bins=50)
        plt.title('Clustering Coefficient Distribution', fontsize=self.font_size+2)
        plt.xlabel('Clustering Coefficient', fontsize=self.font_size)
        plt.ylabel('Count', fontsize=self.font_size)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'clustering_distribution.png'), dpi=300)
        plt.close()
        
        # 3. Connected Components Size Distribution
        plt.figure(figsize=(10, 6))
        component_sizes = [len(c) for c in nx.connected_components(self.graph)]
        component_sizes.sort(reverse=True)
        display_components = min(20, len(component_sizes))
        plt.bar(range(1, display_components+1), component_sizes[:display_components])
        plt.title('Connected Component Sizes', fontsize=self.font_size+2)
        plt.xlabel('Component Rank', fontsize=self.font_size)
        plt.ylabel('Number of Nodes', fontsize=self.font_size)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'component_sizes.png'), dpi=300)
        plt.close()
        
        # 4. Edge Weight Distribution
        if nx.get_edge_attributes(self.graph, 'weight'):
            plt.figure(figsize=(10, 6))
            weights = [d['weight'] for u, v, d in self.graph.edges(data=True)]
            sns.histplot(weights, kde=True, bins=50)
            plt.title('Edge Weight Distribution', fontsize=self.font_size+2)
            plt.xlabel('Weight', fontsize=self.font_size)
            plt.ylabel('Count', fontsize=self.font_size)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'weight_distribution.png'), dpi=300)
            plt.close()
        
        logger.info("Saved network overview visualizations")
    
    def visualize_degree_vs_centrality(self):
        """Visualize the relationship between degree and different centrality measures."""
        logger.info("Generating degree vs centrality visualizations")
        
        G = self.largest_cc
        
        # Calculate centrality measures
        logger.info("Calculating node centrality measures")
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), 
                                                         normalized=True, seed=42)
        closeness_centrality = nx.closeness_centrality(G)
        
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            has_eigenvector = True
        except:
            has_eigenvector = False
            logger.warning("Eigenvector centrality calculation did not converge")
        
        # Create scatter plots
        fig, axes = plt.subplots(1, 3 if has_eigenvector else 2, figsize=(16, 5))
        
        axes[0].scatter(list(degree_centrality.values()), list(betweenness_centrality.values()), 
                      alpha=0.6, s=10)
        axes[0].set_title('Degree vs Betweenness Centrality')
        axes[0].set_xlabel('Degree Centrality')
        axes[0].set_ylabel('Betweenness Centrality')
        
        axes[1].scatter(list(degree_centrality.values()), list(closeness_centrality.values()), 
                      alpha=0.6, s=10)
        axes[1].set_title('Degree vs Closeness Centrality')
        axes[1].set_xlabel('Degree Centrality')
        axes[1].set_ylabel('Closeness Centrality')
        
        if has_eigenvector:
            axes[2].scatter(list(degree_centrality.values()), list(eigenvector_centrality.values()), 
                          alpha=0.6, s=10)
            axes[2].set_title('Degree vs Eigenvector Centrality')
            axes[2].set_xlabel('Degree Centrality')
            axes[2].set_ylabel('Eigenvector Centrality')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'degree_vs_centrality.png'), dpi=300)
        plt.close()
        
        # Top nodes by centrality
        top_n = 20
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Get node labels
        node_labels = {}
        if self.protein_info is not None:
            for col in ['#string_protein_id', 'string_protein_id', 'protein_id']:
                if col in self.protein_info.columns:
                    for _, row in self.protein_info.iterrows():
                        if 'preferred_name' in self.protein_info.columns and not pd.isna(row['preferred_name']):
                            node_labels[row[col]] = row['preferred_name']
                    break
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        axes[0].barh([i for i in range(top_n)], [x[1] for x in top_degree], color=self.palettes['main'][0])
        axes[0].set_title(f'Top {top_n} Proteins by Degree Centrality')
        axes[0].set_xlabel('Degree Centrality')
        y_labels = [node_labels.get(node, node) for node, _ in top_degree]
        axes[0].set_yticks([i for i in range(top_n)])
        axes[0].set_yticklabels(y_labels)
        
        axes[1].barh([i for i in range(top_n)], [x[1] for x in top_betweenness], color=self.palettes['main'][1])
        axes[1].set_title(f'Top {top_n} Proteins by Betweenness Centrality')
        axes[1].set_xlabel('Betweenness Centrality')
        y_labels = [node_labels.get(node, node) for node, _ in top_betweenness]
        axes[1].set_yticks([i for i in range(top_n)])
        axes[1].set_yticklabels(y_labels)
        
        axes[2].barh([i for i in range(top_n)], [x[1] for x in top_closeness], color=self.palettes['main'][2])
        axes[2].set_title(f'Top {top_n} Proteins by Closeness Centrality')
        axes[2].set_xlabel('Closeness Centrality')
        y_labels = [node_labels.get(node, node) for node, _ in top_closeness]
        axes[2].set_yticks([i for i in range(top_n)])
        axes[2].set_yticklabels(y_labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_proteins_by_centrality.png'), dpi=300)
        plt.close()
        logger.info("Saved centrality analysis visualizations")
    
    def visualize_go_term_distribution(self):
        """Visualize the distribution of GO terms across proteins."""
        if not self.protein_to_go:
            logger.warning("GO data not available, skipping GO term distribution visualization")
            return
        
        logger.info("Generating GO term distribution visualizations")
        
        aspect_labels = {'P': 'Biological Process', 'F': 'Molecular Function', 'C': 'Cellular Component'}
        bp_counts = []
        mf_counts = []
        cc_counts = []
        
        for protein, go_dict in self.protein_to_go.items():
            bp_counts.append(len(go_dict.get('P', set())))
            mf_counts.append(len(go_dict.get('F', set())))
            cc_counts.append(len(go_dict.get('C', set())))
        
        plt.figure(figsize=(12, 6))
        data = [bp_counts, mf_counts, cc_counts]
        labels = ['Biological Process', 'Molecular Function', 'Cellular Component']
        
        parts = plt.violinplot(data, showmeans=False, showmedians=True)
        plt.boxplot(data, widths=0.1, showfliers=False)
        
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        for i, d in enumerate(data):
            mean_val = sum(d) / len(d)
            max_val = max(d)
            plt.text(i+1, mean_val, f'Mean: {mean_val:.2f}', ha='center', va='bottom', fontweight='bold')
            plt.text(i+1, max_val, f'Max: {max_val}', ha='center', va='bottom')
        
        plt.xticks([1, 2, 3], labels)
        plt.ylabel('Number of GO Terms per Protein')
        plt.title('Distribution of GO Terms by Ontology Type')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'go_term_distribution.png'), dpi=300)
        plt.close()
        
        # Most common GO terms
        go_frequencies = {'P': Counter(), 'F': Counter(), 'C': Counter()}
        
        for protein, go_dict in self.protein_to_go.items():
            for aspect, terms in go_dict.items():
                if aspect in go_frequencies:
                    go_frequencies[aspect].update(terms)
        
        for aspect, counter in go_frequencies.items():
            top_terms = counter.most_common(15)
            if not top_terms:
                continue
                
            plt.figure(figsize=(12, 8))
            y_pos = range(len(top_terms))
            
            bars = plt.barh(y_pos, [count for _, count in top_terms], 
                          color=self.palettes['go_terms'][int(aspect == 'P') + int(aspect == 'F')*2])
            
            for i, (term, count) in enumerate(top_terms):
                plt.text(count + (max([c for _, c in top_terms]) * 0.01), 
                       i, str(count), va='center')
            
            plt.yticks(y_pos, [term for term, _ in top_terms])
            plt.xlabel('Frequency')
            plt.title(f'Top GO Terms in {aspect_labels[aspect]}')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f'top_go_terms_{aspect}.png'), dpi=300)
            plt.close()
        
        logger.info("Saved GO term distribution visualizations")
    
    def visualize_network_community_structure(self):
        """Visualize the community structure of the network."""
        logger.info("Generating community structure visualizations")
        
        G = self.largest_cc
        
        # Detect communities
        logger.info("Detecting communities using Louvain method")
        partition = community_louvain.best_partition(G)
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        
        sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Community size distribution
        plt.figure(figsize=(12, 6))
        community_sizes = [len(members) for _, members in sorted_communities]
        plt.bar(range(1, len(community_sizes) + 1), community_sizes, color=self.palettes['main'][0])
        plt.title('Community Size Distribution')
        plt.xlabel('Community Rank (by size)')
        plt.ylabel('Number of Nodes')
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'community_size_distribution.png'), dpi=300)
        plt.close()
        
        # Calculate modularity
        modularity = community_louvain.modularity(partition, G)
        logger.info(f"Network modularity: {modularity:.4f}")
        
        # Community distribution pie chart
        top_community_count = min(8, len(sorted_communities))
        top_communities = sorted_communities[:top_community_count]
        
        plt.figure(figsize=(10, 10))
        sizes = [len(members) for _, members in top_communities]
        labels = [f"Community {i+1}\n({len(members)} nodes)" for i, (_, members) in enumerate(top_communities)]
        
        if len(sorted_communities) > top_community_count:
            sizes.append(sum(len(members) for _, members in sorted_communities[top_community_count:]))
            labels.append(f"Other\n({sizes[-1]} nodes)")
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=self.palettes['pastel'])
        plt.axis('equal')
        plt.title(f'Node Distribution Across Top {top_community_count} Communities\nModularity: {modularity:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'community_distribution_pie.png'), dpi=300)
        plt.close()
        
        # Network visualization with community colors (for smaller networks)
        if G.number_of_nodes() <= 2000:
            logger.info("Generating network visualization with community coloring")
            
            if G.number_of_nodes() > 500:
                degrees = dict(G.degree())
                degree_threshold = sorted(degrees.values(), reverse=True)[500]
                nodes_to_keep = [n for n in G.nodes() if degrees[n] >= degree_threshold]
                if len(nodes_to_keep) < 200:
                    nodes_to_keep = sorted(G.nodes(), key=lambda x: degrees[x], reverse=True)[:500]
                vis_graph = G.subgraph(nodes_to_keep).copy()
            else:
                vis_graph = G
            
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(vis_graph, k=0.15, iterations=50, seed=42)
            
            community_colors = {}
            for i, (comm_id, _) in enumerate(top_communities):
                color_idx = i % len(self.palettes['main'])
                community_colors[comm_id] = self.palettes['main'][color_idx]
            
            for comm_id, color in community_colors.items():
                nodes = [n for n in vis_graph.nodes() if partition.get(n) == comm_id]
                nx.draw_networkx_nodes(vis_graph, pos, nodelist=nodes, node_color=color, 
                                     node_size=50, alpha=0.8, label=f"Community {comm_id}")
            
            nx.draw_networkx_edges(vis_graph, pos, alpha=0.2, width=0.5)
            
            plt.title(f'Network Visualization with Community Structure\nModularity: {modularity:.4f}')
            plt.axis('off')
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'community_network_visualization.png'), dpi=300)
            plt.close()
        
        logger.info("Saved community structure visualizations")
    
    def visualize_network_embedding(self):
        """Visualize the network using dimensionality reduction techniques."""
        logger.info("Generating network embedding visualizations")
        
        G = self.largest_cc
        
        # Sample for large networks
        if G.number_of_nodes() > 1000:
            logger.info(f"Sampling large network ({G.number_of_nodes()} nodes) for visualization")
            degrees = dict(G.degree())
            sorted_nodes = sorted(G.nodes(), key=lambda x: degrees[x], reverse=True)
            top_nodes = sorted_nodes[:500]
            random_nodes = np.random.choice(sorted_nodes[500:], size=min(500, len(sorted_nodes[500:])), replace=False)
            nodes_to_keep = list(top_nodes) + list(random_nodes)
            G = G.subgraph(nodes_to_keep).copy()
        
        # Calculate node features
        logger.info("Calculating node features for embedding")
        features = {}
        
        for node in G.nodes():
            features[node] = [G.degree(node)]
        
        clustering = nx.clustering(G)
        for node in G.nodes():
            features[node].append(clustering[node])
        
        try:
            betweenness = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()), seed=42)
            for node in G.nodes():
                features[node].append(betweenness[node])
        except:
            logger.warning("Could not calculate betweenness centrality for embedding")
        
        feature_matrix = np.array([features[node] for node in G.nodes()])
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Apply t-SNE
        logger.info("Applying t-SNE for dimensionality reduction")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, G.number_of_nodes() // 5))
        embedding = tsne.fit_transform(feature_matrix_scaled)
        
        # Community-colored embedding
        try:
            partition = community_louvain.best_partition(G)
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            
            sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
            top_community_count = min(10, len(sorted_communities))
            top_community_ids = [comm_id for comm_id, _ in sorted_communities[:top_community_count]]
            
            node_list = list(G.nodes())
            community_ids = [partition[node] for node in node_list]
            
            color_map = {}
            for i, comm_id in enumerate(set(community_ids)):
                if comm_id in top_community_ids:
                    color_idx = top_community_ids.index(comm_id) % len(self.palettes['main'])
                    color_map[comm_id] = self.palettes['main'][color_idx]
                else:
                    color_map[comm_id] = 'gray'
            
            node_colors = [color_map[comm_id] for comm_id in community_ids]
            
            plt.figure(figsize=(12, 10))
            plt.scatter(embedding[:, 0], embedding[:, 1], c=node_colors, alpha=0.7, s=30)
            
            legend_elements = []
            for i, comm_id in enumerate(top_community_ids):
                if i < top_community_count:
                    color = color_map[comm_id]
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                                   markersize=10, label=f'Community {comm_id}'))
            
            if len(set(community_ids)) > top_community_count:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                               markersize=10, label='Other Communities'))
            
            plt.legend(handles=legend_elements, title="Communities", bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.title('t-SNE Embedding of Network by Community Structure')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'tsne_embedding_communities.png'), dpi=300)
            plt.close()
        except Exception as e:
            logger.warning(f"Could not create community-colored embedding: {str(e)}")
        
        # Degree-colored embedding
        plt.figure(figsize=(12, 10))
        degrees = [G.degree(node) for node in G.nodes()]
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=degrees, cmap='YlOrRd', alpha=0.7, s=30)
        cbar = plt.colorbar(scatter)
        cbar.set_label('Node Degree')
        plt.title('t-SNE Embedding of Network by Node Degree')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'tsne_embedding_degree.png'), dpi=300)
        plt.close()
        
        logger.info("Saved network embedding visualizations")
    
    def visualize_sequence_length_distribution(self):
        """Visualize the distribution of protein sequence lengths."""
        if not self.protein_sequences:
            logger.warning("Protein sequence data not available, skipping sequence length visualization")
            return
        
        logger.info("Generating protein sequence length distribution visualization")
        
        sequence_lengths = [len(seq) for seq in self.protein_sequences.values()]
        
        plt.figure(figsize=(12, 6))
        sns.histplot(sequence_lengths, kde=True, bins=50)
        plt.title('Protein Sequence Length Distribution')
        plt.xlabel('Sequence Length (amino acids)')
        plt.ylabel('Count')
        plt.axvline(np.mean(sequence_lengths), color='r', linestyle='--', label=f'Mean: {np.mean(sequence_lengths):.1f}')
        plt.axvline(np.median(sequence_lengths), color='g', linestyle='--', label=f'Median: {np.median(sequence_lengths):.1f}')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'sequence_length_distribution.png'), dpi=300)
        plt.close()
        
        # Sequence length vs node degree
        if self.graph:
            logger.info("Generating sequence length vs. node degree scatter plot")
            
            degrees = {}
            for node in self.graph.nodes():
                degrees[node] = self.graph.degree(node)
            
            degree_seq_pairs = []
            
            for node, degree in degrees.items():
                if node in self.protein_sequences:
                    degree_seq_pairs.append((degree, len(self.protein_sequences[node])))
                    continue
                
                if self.string_to_alias and node in self.string_to_alias:
                    for source, aliases in self.string_to_alias[node].items():
                        if source in ['Ensembl_UniProt', 'UniProt_AC']:
                            for alias in aliases:
                                if alias in self.protein_sequences:
                                    degree_seq_pairs.append((degree, len(self.protein_sequences[alias])))
                                    break
            
            if degree_seq_pairs:
                degrees, seq_lengths = zip(*degree_seq_pairs)
                
                plt.figure(figsize=(10, 6))
                plt.scatter(degrees, seq_lengths, alpha=0.5, color=self.palettes['main'][0])
                plt.title('Protein Sequence Length vs. Node Degree')
                plt.xlabel('Node Degree')
                plt.ylabel('Sequence Length (amino acids)')
                
                if len(degree_seq_pairs) > 10:
                    z = np.polyfit(degrees, seq_lengths, 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(degrees), p(sorted(degrees)), "r--", alpha=0.8)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'sequence_length_vs_degree.png'), dpi=300)
                plt.close()
        
        logger.info("Saved sequence length visualizations")
    
    def print_statistics(self):
        """Print comprehensive network statistics."""
        print("\n" + "="*60)
        print("PROTEIN-PROTEIN INTERACTION NETWORK ANALYSIS RESULTS")
        print("="*60)
        
        print(f"\nDATA PROCESSING SUMMARY:")
        print(f"  Confidence threshold: {self.confidence_threshold}")
        print(f"  Filtered interactions: {len(self.interactions_df) if self.interactions_df is not None else 0}")
        if self.protein_sequences:
            print(f"  Protein sequences: {len(self.protein_sequences)}")
        if self.protein_to_go:
            print(f"  Proteins with GO annotations: {len(self.protein_to_go)}")
        
        print(f"\nNETWORK STATISTICS:")
        for key, value in self.network_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        if self.graph:
            print(f"\nTOPOLOGY ANALYSIS:")
            degrees = [d for n, d in self.graph.degree()]
            print(f"  Average degree: {np.mean(degrees):.2f}")
            print(f"  Max degree: {max(degrees)}")
            print(f"  Degree std: {np.std(degrees):.2f}")
            
            if self.protein_sequences:
                sequence_lengths = [len(seq) for seq in self.protein_sequences.values()]
                print(f"\nSEQUENCE ANALYSIS:")
                print(f"  Average sequence length: {np.mean(sequence_lengths):.1f} amino acids")
                print(f"  Median sequence length: {np.median(sequence_lengths):.1f} amino acids")
                print(f"  Max sequence length: {max(sequence_lengths)} amino acids")
        
        print(f"\nVISUALIZATIONS:")
        print(f"  All plots saved to: {self.output_dir}/")
        print("="*60)
    
    def run_pipeline(self):
        """Run the complete analysis pipeline."""
        logger.info("Starting PPI network analysis pipeline")
        
        try:
            # Process data
            self.process_protein_links()
            self.process_protein_info()
            self.process_protein_aliases()
            
            # Create and analyze network
            self.create_network()
            self.analyze_network()
            
            # Process optional data
            try:
                self.process_fasta_sequences()
            except Exception as e:
                logger.error(f"Error processing sequences: {str(e)}")
                
            try:
                self.process_go_annotations()
            except Exception as e:
                logger.error(f"Error processing GO annotations: {str(e)}")
            
            # Generate all visualizations
            self.visualize_network_overview()
            self.visualize_degree_vs_centrality()
            self.visualize_go_term_distribution()
            self.visualize_network_community_structure()
            self.visualize_network_embedding()
            self.visualize_sequence_length_distribution()
            
            # Print final statistics
            self.print_statistics()
            
            logger.info("Analysis pipeline completed successfully")
            
            return {
                'graph': self.graph,
                'largest_cc': self.largest_cc,
                'stats': self.network_stats
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise


def main():
    parser = argparse.ArgumentParser(description='Analyze PPI network data with comprehensive visualizations')
    
    parser.add_argument('--ppi_file', type=str, default='raw_data/4932.protein.links.v12.0.txt',
                        help='File containing protein-protein interactions')
    parser.add_argument('--info_file', type=str, default='raw_data/4932.protein.info.v12.0.txt',
                        help='File containing protein information')
    parser.add_argument('--alias_file', type=str, default='raw_data/4932.protein.aliases.v12.0.txt',
                        help='File containing protein aliases/ID mappings')
    parser.add_argument('--fasta_file', type=str, default='raw_data/UP000002311_559292.fasta',
                        help='FASTA file containing protein sequences')
    parser.add_argument('--gaf_file', type=str, default='raw_data/sgd.gaf',
                        help='GAF file containing GO annotations')
    parser.add_argument('--confidence', type=int, default=700,
                        help='Minimum confidence score for interactions (0-1000)')
    parser.add_argument('--output_dir', type=str, default='visualization',
                        help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = PPIAnalyzer(
        ppi_file=args.ppi_file,
        info_file=args.info_file,
        alias_file=args.alias_file,
        fasta_file=args.fasta_file,
        gaf_file=args.gaf_file,
        confidence_threshold=args.confidence,
        output_dir=args.output_dir
    )
    
    # Run analysis
    print(f"Starting PPI network analysis with confidence threshold {args.confidence}")
    results = analyzer.run_pipeline()
    
    print(f"\nAnalysis completed! All visualizations saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
