import os
import pickle
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from collections import Counter, defaultdict
import logging
from tqdm import tqdm
import math
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from community import community_louvain

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("visualization.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("ppi_visualization")

class PPIVisualizer:
    def __init__(self, data_dir="processed_data", output_dir="visualizations"):
        """
        Initialize the PPI network visualizer.
        
        Args:
            data_dir: Directory containing processed data files
            output_dir: Directory to save visualizations
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created output directory: {output_dir}")
        
        # Initialize data structures
        self.graph = None
        self.largest_cc = None
        self.protein_sequences = None
        self.protein_to_go = None
        self.string_to_alias = None
        self.protein_info = None
        
        # Set up color schemes
        self.colors = plt.cm.tab10.colors
        self.palettes = {
            'main': sns.color_palette('tab10', 10),
            'pastel': sns.color_palette('pastel', 10),
            'degree': sns.color_palette('YlOrRd', 9),
            'clustering': sns.color_palette('YlGnBu', 9),
            'go_terms': sns.color_palette('RdPu', 9)
        }
        
        # Set up figure size and style
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig_size = (12, 10)
        self.font_size = 12
        
    def load_data(self):
        """Load all required data files."""
        logger.info(f"Loading data from {self.data_dir}")
        
        # Load network graph
        graph_path = os.path.join(self.data_dir, 'ppi_network.gpickle')
        logger.info(f"Loading network from {graph_path}")
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        logger.info(f"Loaded network with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        
        # Load largest connected component if available
        largest_cc_path = os.path.join(self.data_dir, 'largest_cc.gpickle')
        if os.path.exists(largest_cc_path):
            logger.info(f"Loading largest connected component from {largest_cc_path}")
            with open(largest_cc_path, 'rb') as f:
                self.largest_cc = pickle.load(f)
            logger.info(f"Loaded largest connected component with {self.largest_cc.number_of_nodes()} nodes")
        else:
            # Find largest connected component
            logger.info("Largest connected component file not found, computing it now")
            connected_components = list(nx.connected_components(self.graph))
            largest_cc = max(connected_components, key=len)
            self.largest_cc = self.graph.subgraph(largest_cc).copy()
            logger.info(f"Computed largest connected component with {self.largest_cc.number_of_nodes()} nodes")
        
        # Load protein sequences if available
        seq_path = os.path.join(self.data_dir, 'protein_sequences.pkl')
        if os.path.exists(seq_path):
            logger.info(f"Loading protein sequences from {seq_path}")
            with open(seq_path, 'rb') as f:
                self.protein_sequences = pickle.load(f)
            logger.info(f"Loaded {len(self.protein_sequences)} protein sequences")
        
        # Load GO annotations if available
        go_path = os.path.join(self.data_dir, 'protein_to_go.pkl')
        if os.path.exists(go_path):
            logger.info(f"Loading GO annotations from {go_path}")
            with open(go_path, 'rb') as f:
                self.protein_to_go = pickle.load(f)
            logger.info(f"Loaded GO annotations for {len(self.protein_to_go)} proteins")
        
        # Load alias mappings if available
        alias_path = os.path.join(self.data_dir, 'string_to_alias.pkl')
        if os.path.exists(alias_path):
            logger.info(f"Loading protein aliases from {alias_path}")
            with open(alias_path, 'rb') as f:
                self.string_to_alias = pickle.load(f)
        
        # Load protein info if available
        info_path = os.path.join(self.data_dir, 'protein_info.csv')
        if os.path.exists(info_path):
            logger.info(f"Loading protein info from {info_path}")
            self.protein_info = pd.read_csv(info_path)
            logger.info(f"Loaded info for {len(self.protein_info)} proteins")
    
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
        logger.info(f"Saved degree distribution plot to {self.output_dir}/degree_distribution.png")
        
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
        logger.info(f"Saved clustering coefficient distribution to {self.output_dir}/clustering_distribution.png")
        
        # 3. Connected Components Size Distribution
        plt.figure(figsize=(10, 6))
        component_sizes = [len(c) for c in nx.connected_components(self.graph)]
        component_sizes.sort(reverse=True)
        # Only show up to 20 components for clarity
        display_components = min(20, len(component_sizes))
        plt.bar(range(1, display_components+1), component_sizes[:display_components])
        plt.title('Connected Component Sizes', fontsize=self.font_size+2)
        plt.xlabel('Component Rank', fontsize=self.font_size)
        plt.ylabel('Number of Nodes', fontsize=self.font_size)
        plt.yscale('log')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'component_sizes.png'), dpi=300)
        plt.close()
        logger.info(f"Saved component size distribution to {self.output_dir}/component_sizes.png")
        
        # 4. Edge Weight Distribution (if weights are available)
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
            logger.info(f"Saved edge weight distribution to {self.output_dir}/weight_distribution.png")
    
    def visualize_degree_vs_centrality(self):
        """Visualize the relationship between degree and different centrality measures."""
        logger.info("Generating degree vs centrality visualizations")
        
        # Use largest connected component for centrality measures
        G = self.largest_cc
        
        # Calculate different centrality measures
        logger.info("Calculating node centrality measures (this may take a while for large networks)")
        degree_centrality = nx.degree_centrality(G)
        logger.info("Calculated degree centrality")
        
        betweenness_centrality = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()), 
                                                         normalized=True, seed=42)
        logger.info("Calculated betweenness centrality")
        
        closeness_centrality = nx.closeness_centrality(G)
        logger.info("Calculated closeness centrality")
        
        # Eigenvector centrality may not converge for large networks, so we use try-except
        try:
            eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
            has_eigenvector = True
            logger.info("Calculated eigenvector centrality")
        except:
            has_eigenvector = False
            logger.warning("Eigenvector centrality calculation did not converge")
        
        # Create scatter plots
        fig, axes = plt.subplots(1, 3 if has_eigenvector else 2, figsize=(16, 5))
        
        # Degree vs Betweenness
        axes[0].scatter(list(degree_centrality.values()), list(betweenness_centrality.values()), 
                      alpha=0.6, s=10)
        axes[0].set_title('Degree vs Betweenness Centrality')
        axes[0].set_xlabel('Degree Centrality')
        axes[0].set_ylabel('Betweenness Centrality')
        
        # Degree vs Closeness
        axes[1].scatter(list(degree_centrality.values()), list(closeness_centrality.values()), 
                      alpha=0.6, s=10)
        axes[1].set_title('Degree vs Closeness Centrality')
        axes[1].set_xlabel('Degree Centrality')
        axes[1].set_ylabel('Closeness Centrality')
        
        # Degree vs Eigenvector (if available)
        if has_eigenvector:
            axes[2].scatter(list(degree_centrality.values()), list(eigenvector_centrality.values()), 
                          alpha=0.6, s=10)
            axes[2].set_title('Degree vs Eigenvector Centrality')
            axes[2].set_xlabel('Degree Centrality')
            axes[2].set_ylabel('Eigenvector Centrality')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'degree_vs_centrality.png'), dpi=300)
        plt.close()
        logger.info(f"Saved degree vs centrality plots to {self.output_dir}/degree_vs_centrality.png")
        
        # Create a plot for top nodes by different centrality measures
        top_n = 20
        top_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:top_n]
        
        # Get node names if we have protein info
        node_labels = {}
        if self.protein_info is not None and '#string_protein_id' in self.protein_info.columns and 'preferred_name' in self.protein_info.columns:
            for _, row in self.protein_info.iterrows():
                protein_id = row['#string_protein_id']
                if 'preferred_name' in row and not pd.isna(row['preferred_name']):
                    node_labels[protein_id] = row['preferred_name']
        
        # Create bar plots for top proteins by centrality
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # Top Degree
        bars = axes[0].barh([i for i in range(top_n)], [x[1] for x in top_degree], color=self.palettes['main'][0])
        axes[0].set_title(f'Top {top_n} Proteins by Degree Centrality')
        axes[0].set_xlabel('Degree Centrality')
        y_labels = [node_labels.get(node, node) for node, _ in top_degree]
        axes[0].set_yticks([i for i in range(top_n)])
        axes[0].set_yticklabels(y_labels)
        
        # Top Betweenness
        bars = axes[1].barh([i for i in range(top_n)], [x[1] for x in top_betweenness], color=self.palettes['main'][1])
        axes[1].set_title(f'Top {top_n} Proteins by Betweenness Centrality')
        axes[1].set_xlabel('Betweenness Centrality')
        y_labels = [node_labels.get(node, node) for node, _ in top_betweenness]
        axes[1].set_yticks([i for i in range(top_n)])
        axes[1].set_yticklabels(y_labels)
        
        # Top Closeness
        bars = axes[2].barh([i for i in range(top_n)], [x[1] for x in top_closeness], color=self.palettes['main'][2])
        axes[2].set_title(f'Top {top_n} Proteins by Closeness Centrality')
        axes[2].set_xlabel('Closeness Centrality')
        y_labels = [node_labels.get(node, node) for node, _ in top_closeness]
        axes[2].set_yticks([i for i in range(top_n)])
        axes[2].set_yticklabels(y_labels)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'top_proteins_by_centrality.png'), dpi=300)
        plt.close()
        logger.info(f"Saved top proteins by centrality to {self.output_dir}/top_proteins_by_centrality.png")
    
    def visualize_go_term_distribution(self):
        """Visualize the distribution of GO terms across proteins."""
        if not self.protein_to_go:
            logger.warning("GO data not available, skipping GO term distribution visualization")
            return
        
        logger.info("Generating GO term distribution visualizations")
        
        # Count GO terms per aspect
        aspect_labels = {'P': 'Biological Process', 'F': 'Molecular Function', 'C': 'Cellular Component'}
        bp_counts = []
        mf_counts = []
        cc_counts = []
        
        for protein, go_dict in self.protein_to_go.items():
            bp_counts.append(len(go_dict.get('P', set())))
            mf_counts.append(len(go_dict.get('F', set())))
            cc_counts.append(len(go_dict.get('C', set())))
        
        # Create violin plots for GO term counts
        plt.figure(figsize=(12, 6))
        data = [bp_counts, mf_counts, cc_counts]
        labels = ['Biological Process', 'Molecular Function', 'Cellular Component']
        
        parts = plt.violinplot(data, showmeans=False, showmedians=True)
        
        # Add box plots within the violin plots
        plt.boxplot(data, widths=0.1, showfliers=False)
        
        # Customize the violin plot
        colors = ['#3498db', '#2ecc71', '#9b59b6']
        for i, pc in enumerate(parts['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.7)
        
        for i, d in enumerate(data):
            # Add mean value
            mean_val = sum(d) / len(d)
            plt.text(i+1, mean_val, f'Mean: {mean_val:.2f}', 
                    ha='center', va='bottom', fontweight='bold')
            
            # Add max value
            max_val = max(d)
            plt.text(i+1, max_val, f'Max: {max_val}', 
                    ha='center', va='bottom')
        
        # Set plot labels and title
        plt.xticks([1, 2, 3], labels)
        plt.ylabel('Number of GO Terms per Protein')
        plt.title('Distribution of GO Terms by Ontology Type')
        plt.tight_layout()
        
        plt.savefig(os.path.join(self.output_dir, 'go_term_distribution.png'), dpi=300)
        plt.close()
        logger.info(f"Saved GO term distribution to {self.output_dir}/go_term_distribution.png")
        
        # Find the most common GO terms in each category
        go_frequencies = {
            'P': Counter(),
            'F': Counter(),
            'C': Counter()
        }
        
        for protein, go_dict in self.protein_to_go.items():
            for aspect, terms in go_dict.items():
                if aspect in go_frequencies:
                    go_frequencies[aspect].update(terms)
        
        # Plot top GO terms for each aspect
        for aspect, counter in go_frequencies.items():
            top_terms = counter.most_common(15)
            if not top_terms:
                continue
                
            plt.figure(figsize=(12, 8))
            y_pos = range(len(top_terms))
            
            # Create horizontal bar chart
            bars = plt.barh(y_pos, [count for _, count in top_terms], 
                          color=self.palettes['go_terms'][int(aspect == 'P') + int(aspect == 'F')*2])
            
            # Add frequencies as text
            for i, (term, count) in enumerate(top_terms):
                plt.text(count + (max([c for _, c in top_terms]) * 0.01), 
                       i, str(count), va='center')
            
            plt.yticks(y_pos, [term for term, _ in top_terms])
            plt.xlabel('Frequency')
            plt.title(f'Top GO Terms in {aspect_labels[aspect]}')
            plt.tight_layout()
            
            plt.savefig(os.path.join(self.output_dir, f'top_go_terms_{aspect}.png'), dpi=300)
            plt.close()
            logger.info(f"Saved top GO terms for {aspect_labels[aspect]} to {self.output_dir}/top_go_terms_{aspect}.png")
    
    def visualize_network_community_structure(self):
        """Visualize the community structure of the network."""
        logger.info("Generating community structure visualizations")
        
        # Use largest connected component for community detection
        G = self.largest_cc
        
        # Detect communities using Louvain method
        logger.info("Detecting communities using Louvain method")
        partition = community_louvain.best_partition(G)
        communities = defaultdict(list)
        for node, community_id in partition.items():
            communities[community_id].append(node)
        
        # Sort communities by size
        sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Create community size distribution plot
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
        logger.info(f"Saved community size distribution to {self.output_dir}/community_size_distribution.png")
        
        # Get top communities for visualization
        top_community_count = min(8, len(sorted_communities))
        top_communities = sorted_communities[:top_community_count]
        
        # Calculate modularity to assess community quality
        modularity = community_louvain.modularity(partition, G)
        logger.info(f"Network modularity: {modularity:.4f}")
        
        # Visualize node distribution by community
        # Create a pie chart of community sizes
        plt.figure(figsize=(10, 10))
        sizes = [len(members) for _, members in top_communities]
        labels = [f"Community {i+1}\n({len(members)} nodes)" for i, (_, members) in enumerate(top_communities)]
        
        # Add an "Other" category if there are more communities
        if len(sorted_communities) > top_community_count:
            sizes.append(sum(len(members) for _, members in sorted_communities[top_community_count:]))
            labels.append(f"Other\n({sizes[-1]} nodes)")
        
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=self.palettes['pastel'])
        plt.axis('equal')
        plt.title(f'Node Distribution Across Top {top_community_count} Communities\nModularity: {modularity:.4f}')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'community_distribution_pie.png'), dpi=300)
        plt.close()
        logger.info(f"Saved community distribution pie chart to {self.output_dir}/community_distribution_pie.png")
        
        # Visualize a simplified network layout with communities colored
        if G.number_of_nodes() <= 2000:  # Only do this for smaller networks due to layout complexity
            logger.info("Generating a network visualization with community coloring")
            
            # Create a simplified version of the graph for visualization
            # If the graph is still too big, sample it
            if G.number_of_nodes() > 500:
                # Sample the graph by taking a higher percentage of high-degree nodes
                degrees = dict(G.degree())
                degree_threshold = sorted(degrees.values(), reverse=True)[500]
                nodes_to_keep = [n for n in G.nodes() if degrees[n] >= degree_threshold]
                if len(nodes_to_keep) < 200:  # Ensure we have enough nodes
                    nodes_to_keep = sorted(G.nodes(), key=lambda x: degrees[x], reverse=True)[:500]
                vis_graph = G.subgraph(nodes_to_keep).copy()
                logger.info(f"Created a visualization subgraph with {vis_graph.number_of_nodes()} nodes and {vis_graph.number_of_edges()} edges")
            else:
                vis_graph = G
            
            plt.figure(figsize=(12, 12))
            pos = nx.spring_layout(vis_graph, k=0.15, iterations=50, seed=42)
            
            # Color nodes by community
            community_colors = {}
            for i, (comm_id, _) in enumerate(top_communities):
                color_idx = i % len(self.palettes['main'])
                community_colors[comm_id] = self.palettes['main'][color_idx]
            
            # Draw nodes with community colors
            for comm_id, color in community_colors.items():
                nodes = [n for n in vis_graph.nodes() if partition.get(n) == comm_id]
                nx.draw_networkx_nodes(vis_graph, pos, nodelist=nodes, node_color=color, 
                                     node_size=50, alpha=0.8, label=f"Community {comm_id}")
            
            # Draw edges
            nx.draw_networkx_edges(vis_graph, pos, alpha=0.2, width=0.5)
            
            plt.title(f'Network Visualization with Community Structure\nModularity: {modularity:.4f}')
            plt.axis('off')
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.0))
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'community_network_visualization.png'), dpi=300)
            plt.close()
            logger.info(f"Saved community network visualization to {self.output_dir}/community_network_visualization.png")
    
    def visualize_network_embedding(self):
        """Visualize the network using dimensionality reduction techniques."""
        logger.info("Generating network embedding visualizations")
        
        # Use largest connected component for embedding
        G = self.largest_cc
        
        # If the network is very large, sample it for visualization
        if G.number_of_nodes() > 1000:
            logger.info(f"Network is large ({G.number_of_nodes()} nodes), sampling for visualization")
            # Preferentially sample high-degree nodes
            degrees = dict(G.degree())
            sorted_nodes = sorted(G.nodes(), key=lambda x: degrees[x], reverse=True)
            # Take top nodes and some random nodes
            top_nodes = sorted_nodes[:500]
            random_nodes = np.random.choice(sorted_nodes[500:], size=min(500, len(sorted_nodes[500:])), replace=False)
            nodes_to_keep = list(top_nodes) + list(random_nodes)
            G = G.subgraph(nodes_to_keep).copy()
            logger.info(f"Sampled network has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        # Calculate node features for embedding
        logger.info("Calculating node features for embedding")
        features = {}
        
        # Add degree
        for node in G.nodes():
            features[node] = [G.degree(node)]
        
        # Add clustering coefficient
        clustering = nx.clustering(G)
        for node in G.nodes():
            features[node].append(clustering[node])
        
        # Optionally, add more topological features
        try:
            # Betweenness centrality (with approximation for large networks)
            betweenness = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()), seed=42)
            for node in G.nodes():
                features[node].append(betweenness[node])
        except:
            logger.warning("Could not calculate betweenness centrality, skipping this feature")
        
        # Create feature matrix
        feature_matrix = np.array([features[node] for node in G.nodes()])
        
        # Standardize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Apply t-SNE for dimensionality reduction
        logger.info("Applying t-SNE for dimensionality reduction")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, G.number_of_nodes() // 5))
        embedding = tsne.fit_transform(feature_matrix_scaled)
        
        # If we have community information, use it for coloring
        try:
            partition = community_louvain.best_partition(G)
            
            # Get top communities
            communities = defaultdict(list)
            for node, comm_id in partition.items():
                communities[comm_id].append(node)
            
            sorted_communities = sorted(communities.items(), key=lambda x: len(x[1]), reverse=True)
            top_community_count = min(10, len(sorted_communities))
            top_community_ids = [comm_id for comm_id, _ in sorted_communities[:top_community_count]]
            
            # Convert partition to colors
            node_list = list(G.nodes())
            community_ids = [partition[node] for node in node_list]
            
            # Map community IDs to colors
            unique_communities = set(community_ids)
            color_map = {}
            for i, comm_id in enumerate(unique_communities):
                if comm_id in top_community_ids:
                    color_idx = top_community_ids.index(comm_id) % len(self.palettes['main'])
                    color_map[comm_id] = self.palettes['main'][color_idx]
                else:
                    color_map[comm_id] = 'gray'
            
            node_colors = [color_map[comm_id] for comm_id in community_ids]
            
            # Create scatter plot with community colors
            plt.figure(figsize=(12, 10))
            scatter = plt.scatter(embedding[:, 0], embedding[:, 1], c=node_colors, alpha=0.7, s=30)
            
            # Create legend
            legend_elements = []
            for i, comm_id in enumerate(top_community_ids):
                if i < top_community_count:  # Limit legend size
                    color = color_map[comm_id]
                    legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                                   markersize=10, label=f'Community {comm_id}'))
            
            # Add "Other" to legend if needed
            if len(unique_communities) > top_community_count:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                               markersize=10, label='Other Communities'))
            
            plt.legend(handles=legend_elements, title="Communities", bbox_to_anchor=(1.05, 1), loc='upper left')
            
            plt.title('t-SNE Embedding of Network by Community Structure')
            plt.xlabel('t-SNE Dimension 1')
            plt.ylabel('t-SNE Dimension 2')
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'tsne_embedding_communities.png'), dpi=300)
            plt.close()
            logger.info(f"Saved t-SNE embedding by communities to {self.output_dir}/tsne_embedding_communities.png")
        except Exception as e:
            logger.warning(f"Could not create community-colored embedding: {str(e)}")
        
        # Create scatter plot with degree coloring
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
        logger.info(f"Saved t-SNE embedding by degree to {self.output_dir}/tsne_embedding_degree.png")
    
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
        logger.info(f"Saved sequence length distribution to {self.output_dir}/sequence_length_distribution.png")
        
        # Additional visualization: sequence length vs node degree
        if self.graph:
            logger.info("Generating sequence length vs. node degree scatter plot")
            
            # Get degrees and sequence lengths for nodes that have both
            degrees = {}
            for node in self.graph.nodes():
                degrees[node] = self.graph.degree(node)
            
            # Match protein IDs with sequence data
            degree_seq_pairs = []
            
            # Direct matching
            for node, degree in degrees.items():
                if node in self.protein_sequences:
                    degree_seq_pairs.append((degree, len(self.protein_sequences[node])))
                    continue
                
                # Try via aliases
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
                
                # Add regression line if there are enough points
                if len(degree_seq_pairs) > 10:
                    z = np.polyfit(degrees, seq_lengths, 1)
                    p = np.poly1d(z)
                    plt.plot(sorted(degrees), p(sorted(degrees)), "r--", alpha=0.8)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'sequence_length_vs_degree.png'), dpi=300)
                plt.close()
                logger.info(f"Saved sequence length vs. degree plot to {self.output_dir}/sequence_length_vs_degree.png")
    
    def run_visualization_pipeline(self):
        """Run the complete visualization pipeline."""
        logger.info("Starting visualization pipeline")
        
        try:
            # Load data
            self.load_data()
            
            # Run all visualizations
            self.visualize_network_overview()
            self.visualize_degree_vs_centrality()
            self.visualize_go_term_distribution()
            self.visualize_network_community_structure()
            self.visualize_network_embedding()
            self.visualize_sequence_length_distribution()
            
            logger.info("Visualization pipeline completed successfully")
            logger.info(f"All visualizations saved to {self.output_dir}/")
            
        except Exception as e:
            logger.error(f"Visualization pipeline failed: {str(e)}")
            raise


if __name__ == "__main__":
    # Create visualizer with default settings
    visualizer = PPIVisualizer(
        data_dir="processed_data",
        output_dir="visualizations"
    )
    
    # Run the visualization pipeline
    visualizer.run_visualization_pipeline()
    
    print(f"\nVisualization completed! All visualizations saved to 'visualizations/' directory.")