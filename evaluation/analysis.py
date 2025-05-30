# evaluation/analysis.py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict
import logging
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def analyze_network_topology(data, hub_percentile=90):
    """
    Analyze topological properties of the PPI network.
    
    Args:
        data (torch_geometric.data.Data): PyG graph data.
        hub_percentile (int): Percentile to define hub proteins (e.g., 90 for top 10%).
    
    Returns:
        dict: Dictionary with network statistics.
    """
    if data.num_nodes == 0:
        logging.warning("Graph has no nodes. Cannot analyze topology.")
        return {
            'num_nodes': 0, 'num_edges': 0, 'avg_degree': 0,
            'max_degree': 0, 'min_degree': 0, 'median_degree': 0,
            'hub_nodes_indices': [], 'num_hubs': 0, 'avg_clustering': 0,
            'largest_cc_size': 0, 'largest_cc_diameter': "N/A"
        }

    # Convert to NetworkX for analysis
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    
    # Add edges if they exist
    if data.edge_index is not None and data.edge_index.numel() > 0:
        edge_list = data.edge_index.cpu().numpy().T
        G.add_edges_from(edge_list)
    else:
        logging.warning("Graph has no edges. Some topology stats will be 0 or N/A.")

    stats = {}
    stats['num_nodes'] = G.number_of_nodes()
    stats['num_edges'] = G.number_of_edges()
    
    if G.number_of_nodes() > 0:
        stats['avg_degree'] = 2 * G.number_of_edges() / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        
        degree_sequence = sorted([d for _, d in G.degree()], reverse=True)
        if degree_sequence:
            stats['max_degree'] = max(degree_sequence)
            stats['min_degree'] = min(degree_sequence)
            stats['median_degree'] = np.median(degree_sequence)
            
            degree_dict = dict(G.degree())
            if degree_dict: # Ensure degree_dict is not empty before calculating percentile
                degrees_for_percentile = list(degree_dict.values()) # Use np.percentile correctly; it expects an array-like input
                if degrees_for_percentile: # Check if the list is not empty
                    hub_degree_threshold = np.percentile(degrees_for_percentile, hub_percentile)
                    hubs = [node for node, degree in degree_dict.items() if degree >= hub_degree_threshold]
                    stats['hub_nodes_indices'] = hubs
                    stats['num_hubs'] = len(hubs)
                else: # Should not happen if degree_dict was populated
                    stats['hub_nodes_indices'] = []
                    stats['num_hubs'] = 0
            else: # If no nodes have degrees (e.g. graph with nodes but no edges)
                stats['hub_nodes_indices'] = []
                stats['num_hubs'] = 0

            stats['avg_clustering'] = nx.average_clustering(G) if G.number_of_edges() > 0 else 0 # Avg clustering needs edges
        else: # No degrees (e.g. graph with nodes but no edges)
            stats['max_degree'] = 0
            stats['min_degree'] = 0
            stats['median_degree'] = 0
            stats['hub_nodes_indices'] = []
            stats['num_hubs'] = 0
            stats['avg_clustering'] = 0

        try:
            if G.number_of_edges() > 0: # Connected components require edges
                largest_cc = max(nx.connected_components(G), key=len)
                largest_cc_subgraph = G.subgraph(largest_cc)
                stats['largest_cc_size'] = len(largest_cc)
                stats['largest_cc_diameter'] = nx.diameter(largest_cc_subgraph) if largest_cc_subgraph.number_of_edges() > 0 else "N/A (component has no edges)"
            else:
                stats['largest_cc_size'] = G.number_of_nodes() if G.number_of_nodes() > 0 else 0 # Each node is a CC
                stats['largest_cc_diameter'] = "N/A (no edges)"

        except (nx.NetworkXError, ValueError) as e:
            logging.warning(f"Could not compute largest CC properties: {e}")
            stats['largest_cc_size'] = "N/A"
            stats['largest_cc_diameter'] = "N/A"
    else: # No nodes
        for key in ['avg_degree', 'max_degree', 'min_degree', 'median_degree', 'hub_nodes_indices', 'num_hubs', 'avg_clustering', 'largest_cc_size', 'largest_cc_diameter']:
            stats[key] = 0 if key not in ['largest_cc_diameter'] else "N/A"
        stats['hub_nodes_indices'] = []


    return stats


def generate_topology_report(stats, node_idx_to_protein_id=None, filepath=None, top_n_hubs_to_show=5):
    """
    Generate a report on network topology with biological interpretation.
    
    Args:
        stats (dict): Dictionary with network statistics.
        node_idx_to_protein_id (dict, optional): Mapping from node index to original protein ID/name.
        filepath (str, optional): Optional path to save the report.
        top_n_hubs_to_show (int): Number of top hub IDs/names to list if mapping is provided.
    """
    report = [
        "=== PPI NETWORK TOPOLOGY ANALYSIS ===",
        "",
        f"Number of proteins (nodes): {stats.get('num_nodes', 'N/A')}",
        f"Number of interactions (edges): {stats.get('num_edges', 'N/A')}",
        f"Average number of interactions per protein: {stats.get('avg_degree', 'N/A'):.2f}",
        f"Maximum interactions for a single protein: {stats.get('max_degree', 'N/A')}",
        f"Number of hub proteins (highly connected): {stats.get('num_hubs', 'N/A')}",
    ]
    if node_idx_to_protein_id and stats.get('hub_nodes_indices'):
        report.append("  Example Hub Protein IDs/Names (up to top N):")
        for i, hub_idx in enumerate(stats['hub_nodes_indices'][:top_n_hubs_to_show]):
            protein_name = node_idx_to_protein_id.get(hub_idx, f"Index {hub_idx}")
            report.append(f"    - {protein_name} (Node Index: {hub_idx})")

    report.extend([
        f"Network modularity (avg. clustering coefficient): {stats.get('avg_clustering', 'N/A'):.4f}",
        f"Size of largest connected component: {stats.get('largest_cc_size', 'N/A')}",
        f"Diameter of largest connected component: {stats.get('largest_cc_diameter', 'N/A')}",
        "",
        "--- Biological Interpretation ---",
        "1. Network Structure: Scale-free topology is common. Hubs are key.",
        "2. Functional Implications: High clustering suggests complexes. Diameter indicates separation.",
        "3. Hub Proteins: Often essential, conserved, coordinators. Disruptions can be severe.",
        "4. Link Prediction Context: Novel links likely within modules. Similar patterns imply shared function.",
    ])
    
    full_report = "\n".join(report)
    logging.info(full_report)
    
    if filepath:
        with open(filepath, 'w') as f:
            f.write(full_report)
        logging.info(f"Topology report saved to {filepath}")
    
    return full_report


def identify_functional_modules(data, node_embeddings, method='kmeans', k_range=None, dbscan_eps=0.5, dbscan_min_samples=5):
    """
    Use node embeddings to identify potential functional modules (protein complexes).
    
    Args:
        data (torch_geometric.data.Data): PyG graph data.
        node_embeddings (torch.Tensor): Node embeddings from GNN.
        method (str): Clustering method ('kmeans' or 'dbscan').
        k_range (range, optional): Range of k values for K-Means silhouette analysis. Default: range(2, min(11, num_nodes // 5)).
        dbscan_eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other (for DBSCAN).
        dbscan_min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point (for DBSCAN).
        
    Returns:
        tuple: (modules_dict, num_identified_modules)
               modules_dict (dict): Dictionary with module assignments (e.g., {"Module 1": [node_idx_1, node_idx_2,...]}).
               num_identified_modules (int): Number of distinct modules found (excluding noise for DBSCAN).
    """
    if node_embeddings is None or node_embeddings.numel() == 0:
        logging.warning("Node embeddings are empty. Cannot identify functional modules.")
        return {}, 0
        
    embeddings_np = node_embeddings.cpu().numpy()
    num_nodes = data.num_nodes

    if num_nodes < 2 : # Clustering requires at least 2 points
        logging.warning(f"Not enough nodes ({num_nodes}) for clustering.")
        return {"Module 1": list(range(num_nodes))}, 1 if num_nodes > 0 else 0


    cluster_labels = None
    num_clusters = 0

    if method == 'kmeans':
        if k_range is None:
            upper_k_bound = min(11, num_nodes -1 if num_nodes > 1 else 1) # K must be < n_samples
            k_range = range(2, max(3, upper_k_bound)) # Ensure k_range has at least one value if possible

        if not k_range or k_range.start >= num_nodes: # If k_range is empty or invalid
             logging.warning(f"Cannot perform K-Means: k_range {k_range} is invalid for {num_nodes} nodes. Defaulting to 1 cluster.")
             cluster_labels = np.zeros(num_nodes, dtype=int)
             num_clusters = 1
        else:
            sil_scores = []
            best_k = k_range.start
            max_sil_score = -1

            for k_idx, k in enumerate(k_range):
                if k >= num_nodes: # k must be less than n_samples for silhouette_score
                    logging.warning(f"Skipping k={k} for K-Means as it's >= num_nodes ({num_nodes}).")
                    if k_idx == 0 and len(k_range) == 1: # Only one invalid k was given
                        cluster_labels = np.zeros(num_nodes, dtype=int) # Default to one cluster
                        num_clusters = 1
                        break
                    continue

                kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
                current_labels = kmeans.fit_predict(embeddings_np)
                try:
                    score = silhouette_score(embeddings_np, current_labels)
                    sil_scores.append(score)
                    if score > max_sil_score:
                        max_sil_score = score
                        best_k = k
                except ValueError as e: # silhouette_score fails if only 1 cluster is found by KMeans for a given k
                    logging.warning(f"Silhouette score error for k={k}: {e}. Skipping this k.")
                    sil_scores.append(-1) # Penalize this k
                    continue
            
            if not sil_scores and not cluster_labels: # If all k values failed or k_range was effectively empty
                logging.warning("K-Means failed to find suitable k. Defaulting to 1 cluster.")
                cluster_labels = np.zeros(num_nodes, dtype=int)
                num_clusters = 1
            elif not cluster_labels : # If sil_scores were computed
                logging.info(f"Best k for K-Means based on silhouette score: {best_k}")
                final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init='auto')
                cluster_labels = final_kmeans.fit_predict(embeddings_np)
                num_clusters = best_k
    
    elif method == 'dbscan':
        dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        cluster_labels = dbscan.fit_predict(embeddings_np)
        num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0) # Number of clusters in labels, ignoring noise if present.
        if num_clusters == 0 and -1 in cluster_labels: # All points are noise
            logging.warning("DBSCAN found all points as noise. Treating as 1 effective cluster.")
            num_clusters = 1 # Or assign all to a single "noise" module
        elif num_clusters == 0: # No noise, but also no clusters (e.g. very few points)
             logging.warning("DBSCAN found 0 clusters. Defaulting to 1 cluster.")
             cluster_labels = np.zeros(num_nodes, dtype=int)
             num_clusters = 1
        logging.info(f"DBSCAN found {num_clusters} clusters (excluding noise points labeled -1).")

    else:
        logging.error(f"Unknown clustering method: {method}. Defaulting to 1 cluster.")
        cluster_labels = np.zeros(num_nodes, dtype=int)
        num_clusters = 1

    modules = defaultdict(list)
    for node_idx, module_idx in enumerate(cluster_labels):
        module_key = int(module_idx) if module_idx != -1 else "Noise"  # Map DBSCAN noise (-1) to a specific module name if desired, or handle separately
        modules[module_key].append(int(node_idx))
    
    modules_dict = {}
    module_counter = 1
    noise_nodes = []
    for label_key in sorted(modules.keys(), key=lambda x: (isinstance(x, str), x)): # Sort to have "Noise" last if it exists
        if label_key == "Noise":
            noise_nodes = modules[label_key]
            continue
        modules_dict[f"Module {module_counter}"] = modules[label_key]
        module_counter += 1
    
    if noise_nodes:
         modules_dict["Noise (DBSCAN)"] = noise_nodes
         num_identified_modules = num_clusters # num_identified_modules should reflect actual clusters, not the noise "cluster"
    else:
        num_identified_modules = num_clusters if num_clusters > 0 else 1 # Ensure at least 1 module if no clusters found

    if not modules_dict and num_nodes > 0 : # If somehow modules_dict is empty but there are nodes
        modules_dict["Module 1"] = list(range(num_nodes))
        num_identified_modules = 1

    return modules_dict, num_identified_modules


def visualize_functional_modules(data, node_embeddings, save_path=None, clustering_method='kmeans', **cluster_kwargs):
    """
    Visualize protein functional modules based on embeddings.
    
    Args:
        data (torch_geometric.data.Data): PyG graph data.
        node_embeddings (torch.Tensor): Node embeddings from GNN.
        save_path (str, optional): Optional path to save the figure.
        clustering_method (str): 'kmeans' or 'dbscan'.
        **cluster_kwargs: Additional arguments for the chosen clustering method.
    """
    if node_embeddings is None or node_embeddings.numel() == 0:
        logging.warning("Node embeddings are empty. Cannot visualize functional modules.")
        return {}

    modules_dict, num_effective_modules = identify_functional_modules(data, node_embeddings, method=clustering_method, **cluster_kwargs)

    if not modules_dict or num_effective_modules == 0:
        logging.warning("No modules identified or num_effective_modules is 0. Skipping visualization.")
        if data.num_nodes > 0 and not modules_dict: # Fallback: visualize all nodes as one module if modules_dict is empty but nodes exist
            modules_dict = {"Module 1": list(range(data.num_nodes))}
            num_effective_modules = 1
        else:
            return modules_dict

    num_colors_needed = max(1, num_effective_modules) # Need at least one color for cmap
    cmap = plt.cm.get_cmap('viridis', num_colors_needed) # 'viridis' or 'tab20' for more colors
    
    node_colors_by_module_idx = np.zeros(data.num_nodes) # Store module index for coloring
    
    module_idx_map = {} # map module name (e.g. "Module 1") to a numeric index for cmap
    current_color_idx = 0

    for module_name, node_indices in modules_dict.items():
        if "Noise" in module_name: # Color noise differently or skip coloring
            for node_idx in node_indices:
                node_colors_by_module_idx[node_idx] = -1 # Special index for noise
        else:
            if module_name not in module_idx_map:
                module_idx_map[module_name] = current_color_idx
                current_color_idx +=1
            
            color_idx_for_cmap = module_idx_map[module_name] % num_colors_needed # Cycle through colors if more modules than cmap distinct colors
            for node_idx in node_indices:
                 node_colors_by_module_idx[node_idx] = color_idx_for_cmap


    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    if data.edge_index is not None:
         G.add_edges_from(data.edge_index.cpu().numpy().T)
    
    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        
    for module_name, assigned_module_idx in module_idx_map.items():
        nodes_in_module = modules_dict[module_name]
        color_for_module = cmap(assigned_module_idx % num_colors_needed) # Ensure valid color index
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_module, node_size=80, 
                               node_color=[color_for_module], label=module_name, alpha=0.8)

    if "Noise (DBSCAN)" in modules_dict: # Draw noise nodes if they exist
        noise_nodes = modules_dict["Noise (DBSCAN)"]
        nx.draw_networkx_nodes(G, pos, nodelist=noise_nodes, node_size=50,
                               node_color='grey', label='Noise (DBSCAN)', alpha=0.5)

    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray')
    
    plt.title(f"Protein Functional Modules ({clustering_method.capitalize()})", fontsize=16)
    plt.axis('off')
    plt.legend(fontsize=10, title="Functional Modules", loc="upper right")
    
    plt.figtext(0.5, 0.01, 
               f"Identified {num_effective_modules} potential modules using {clustering_method}.\n"
               "Proteins in the same module may share functions.",
               ha="center", fontsize=10, bbox={"facecolor":"lightcoral", "alpha":0.2, "pad":5})
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Functional modules visualization saved to {save_path}")
    
    # plt.show()
    return modules_dict


def analyze_prediction_patterns(model, data, device, degree_percentile_threshold=50):
    """
    Analyze patterns in link predictions based on node degrees.
    
    Args:
        model: The trained GNN model.
        data (torch_geometric.data.Data): PyG graph data (should be a single graph with edge_label_index).
        device: Device to run on (cuda/cpu).
        degree_percentile_threshold (int): Percentile to differentiate low/high degree nodes.
        
    Returns:
        dict: Dictionary with prediction analysis results.
    """
    model.eval()
    data = data.to(device)
    
    if not (hasattr(data, 'edge_label_index') and data.edge_label_index is not None and
            hasattr(data, 'edge_label') and data.edge_label is not None and
            hasattr(data, 'edge_index') and data.edge_index is not None):
        logging.error("Data object is missing required attributes (edge_label_index, edge_label, edge_index) for prediction pattern analysis.")
        return {}

    with torch.no_grad():
        logits = model(data) # Assumes model takes the whole data object
        pred_scores = torch.sigmoid(logits).cpu().numpy()
    
    edge_label_index_np = data.edge_label_index.cpu().numpy()
    edge_labels_np = data.edge_label.cpu().numpy()
    
    G_msg = nx.Graph() # Build graph from message-passing edges (data.edge_index) to calculate degrees
    G_msg.add_nodes_from(range(data.num_nodes))
    G_msg.add_edges_from(data.edge_index.cpu().numpy().T)
    
    if G_msg.number_of_nodes() == 0:
        logging.warning("Graph for degree calculation has no nodes.")
        return {}

    degree_dict = dict(G_msg.degree())
    if not degree_dict: # No nodes with degrees
        actual_degree_threshold = 0
        logging.warning("No node degrees found. Using 0 as degree threshold.")
    else:
        degrees = list(degree_dict.values())
        actual_degree_threshold = np.percentile(degrees, degree_percentile_threshold) if degrees else 0
    
    logging.info(f"Degree threshold for high/low classification (at {degree_percentile_threshold}th percentile): {actual_degree_threshold:.2f}")

    results = {
        'high_to_high': {'count': 0, 'avg_score': 0.0, 'true_pos': 0, 'actual_pos':0},
        'high_to_low': {'count': 0, 'avg_score': 0.0, 'true_pos': 0, 'actual_pos':0},
        'low_to_low': {'count': 0, 'avg_score': 0.0, 'true_pos': 0, 'actual_pos':0}
    }
    
    for i in range(edge_label_index_np.shape[1]):
        src_node, dst_node = edge_label_index_np[0, i], edge_label_index_np[1, i]
        score = pred_scores[i]
        true_label = edge_labels_np[i]
        
        src_degree = degree_dict.get(src_node, 0)
        dst_degree = degree_dict.get(dst_node, 0)
        
        src_is_high = src_degree > actual_degree_threshold
        dst_is_high = dst_degree > actual_degree_threshold
        
        category = None
        if src_is_high and dst_is_high: category = 'high_to_high'
        elif (src_is_high and not dst_is_high) or (not src_is_high and dst_is_high): category = 'high_to_low'
        else: category = 'low_to_low'
        
        results[category]['count'] += 1
        results[category]['avg_score'] += score
        if true_label == 1: # Actual positive interaction
            results[category]['actual_pos'] +=1
            if score >= 0.5: # Assuming 0.5 threshold for predicted positive
                results[category]['true_pos'] +=1
                
    for category in results:
        if results[category]['count'] > 0:
            results[category]['avg_score'] /= results[category]['count']
        results[category]['precision'] = results[category]['true_pos'] / results[category]['count'] if results[category]['count'] > 0 else 0 # Add precision for this category
        results[category]['recall'] = results[category]['true_pos'] / results[category]['actual_pos'] if results[category]['actual_pos'] > 0 else 0 # Add recall for this category (TP / actual positives in this category)


    results['total_predictions_analyzed'] = edge_label_index_np.shape[1]
    results['degree_threshold_value'] = actual_degree_threshold
    return results


def generate_prediction_patterns_report(results, filepath=None):
    """Generate a report on prediction patterns."""
    if not results or 'total_predictions_analyzed' not in results:
        return "No prediction pattern results to generate report."
    
    total = results.get('total_predictions_analyzed', 1)
    report = [
        "=== PPI PREDICTION PATTERN ANALYSIS ===",
        f"Degree threshold for High/Low classification: {results.get('degree_threshold_value', 'N/A'):.2f}",
        "",
        "--- Prediction Statistics by Node Connectivity ---",
        f"Total supervised edges analyzed: {total}",
    ]

    for cat_key, cat_name in [('high_to_high', 'Hub-Hub (High-High)'), 
                              ('high_to_low', 'Hub-Peripheral (High-Low/Low-High)'), 
                              ('low_to_low', 'Peripheral-Peripheral (Low-Low)')]:
        cat_data = results.get(cat_key, {'count': 0, 'avg_score': 0, 'precision':0, 'recall':0})
        report.extend([
            f"\n{cat_name} Predictions:",
            f"  Count: {cat_data['count']} ({cat_data['count']/total*100:.1f}% of total supervised edges)",
            f"  Average confidence score: {cat_data['avg_score']:.4f}",
            f"  Precision within category: {cat_data.get('precision', 'N/A'):.4f}",
            f"  Recall within category: {cat_data.get('recall', 'N/A'):.4f}",
        ])
    
    report.extend([
        "", "--- Biological Interpretation ---",
        "1. Hub-Hub: Interactions between highly connected proteins, potentially bridging modules.",
        "2. Hub-Peripheral: Interactions connecting hubs to less connected proteins, often regulatory.",
        "3. Peripheral-Peripheral: Interactions between less connected proteins, often within specific pathways/complexes.",
        "4. Confidence/Precision Patterns: Observe if the model is more confident or precise for certain types of interactions."
    ])
    
    full_report = "\n".join(report)
    logging.info(full_report)
    if filepath:
        with open(filepath, 'w') as f:
            f.write(full_report)
        logging.info(f"Prediction patterns report saved to {filepath}")
    return full_report


