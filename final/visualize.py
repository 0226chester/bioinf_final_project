import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def visualize_graph(data, node_colors=None, node_size_attr=None, node_idx_to_name=None, 
                    label_top_n_nodes=0, title="PPI Network", node_base_size=50, save_path=None):
    """
    Visualize a PPI graph.
    
    Args:
        data (torch_geometric.data.Data): PyG graph data.
        node_colors (list/np.array/str, optional): Node colors. Can be a single color string,
                                                   or an array/list of colors/values to map via cmap.
        node_size_attr (np.array, optional): Attribute to scale node sizes (e.g., degree).
        node_idx_to_name (dict, optional): Mapping from node index to protein name/ID for labels.
        label_top_n_nodes (int): If node_idx_to_name provided, label top N nodes by degree.
        title (str): Plot title.
        node_base_size (int): Base size of nodes.
        save_path (str, optional): Path to save the figure.
    """
    if data.num_nodes == 0:
        logging.warning("Graph has no nodes. Skipping visualization.")
        return

    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    if data.edge_index is not None:
        G.add_edges_from(data.edge_index.cpu().numpy().T)
    
    pos = nx.spring_layout(G, k=0.5, iterations=50, seed=42)
    
    # Node sizes
    if node_size_attr is not None:
        sizes = np.array(node_size_attr)
        node_sizes = node_base_size + (sizes / np.max(sizes) if np.max(sizes) > 0 else sizes) * node_base_size * 2
    else:
        node_sizes = node_base_size
        
    # Node colors
    color_map_name = 'viridis' # Default cmap if node_colors is an array of values
    if isinstance(node_colors, (list, np.ndarray)) and not all(isinstance(c, str) for c in node_colors):
        # Assume node_colors is an array of values to be mapped by a colormap
        pass # nx.draw_networkx_nodes will handle this with cmap
    elif node_colors is None:
        node_colors = 'lightblue'


    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.get_cmap(color_map_name) if isinstance(node_colors, (list, np.ndarray)) else None, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray')
    
    # Node labels
    labels = {}
    if node_idx_to_name and label_top_n_nodes > 0:
        degrees = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        for i in range(min(label_top_n_nodes, len(degrees))):
            node_idx = degrees[i][0]
            labels[node_idx] = node_idx_to_name.get(node_idx, f"Idx {node_idx}")
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_color='black')

    plt.title(title, fontsize=16)
    plt.axis('off')
    plt.figtext(0.5, 0.01, "Nodes: Proteins, Edges: Interactions.", ha="center", fontsize=10,
               bbox={"facecolor":"lightcoral", "alpha":0.2, "pad":5})
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Graph visualization saved to {save_path}")
    plt.show()


def visualize_embeddings(model, data, device, perplexity=30, color_by=None, cmap_name='viridis',
                         title="Protein Embeddings (t-SNE)", save_path=None):
    """
    Visualize node embeddings using t-SNE.
    
    Args:
        model: The trained GNN model.
        data (torch_geometric.data.Data): PyG graph data.
        device: Device to run on (cuda/cpu).
        perplexity (float): t-SNE perplexity.
        color_by (np.array, optional): Array of values to color nodes by (e.g., module assignments, degrees).
        cmap_name (str): Name of the Matplotlib colormap to use if color_by is provided.
        title (str): Plot title.
        save_path (str, optional): Path to save the figure.
    """
    model.eval()
    data_on_device = data.to(device)
    
    with torch.no_grad():
        if not (hasattr(data_on_device, 'x') and hasattr(data_on_device, 'edge_index')):
            logging.error("Data object missing 'x' or 'edge_index' for embedding visualization.")
            return
        embeddings = model.get_embeddings(data_on_device.x, data_on_device.edge_index)
    
    embeddings_np = embeddings.cpu().numpy()
    if embeddings_np.shape[0] < perplexity : # tSNE perplexity should be less than n_samples -1
        perplexity = max(5, embeddings_np.shape[0] // 2 -1) # Adjust perplexity
        if perplexity < 5 : # still too low
            logging.warning(f"Too few samples ({embeddings_np.shape[0]}) for meaningful t-SNE. Skipping embedding visualization.")
            return

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    node_coords = tsne.fit_transform(embeddings_np)
    
    plt.figure(figsize=(10, 8))
    
    node_colors_for_plot = color_by if color_by is not None else np.arange(data.num_nodes)
    # Ensure color_by has the same length as num_nodes if provided
    if color_by is not None and len(color_by) != data.num_nodes:
        logging.warning(f"Length of 'color_by' ({len(color_by)}) does not match num_nodes ({data.num_nodes}). Defaulting to node index for color.")
        node_colors_for_plot = np.arange(data.num_nodes)

    scatter = plt.scatter(node_coords[:, 0], node_coords[:, 1], 
                         c=node_colors_for_plot, cmap=plt.get_cmap(cmap_name), 
                         alpha=0.7, s=50)
    
    plt.colorbar(scatter, label='Node Attribute' if color_by is not None else 'Node Index')
    plt.title(title, fontsize=16)
    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.figtext(0.5, 0.01, "Clusters may indicate similar functions or pathways.", ha="center", fontsize=10,
               bbox={"facecolor":"lightcoral", "alpha":0.2, "pad":5})
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Embeddings visualization saved to {save_path}")
    plt.show()


def visualize_predicted_links(model, data, device, threshold=0.7, node_idx_to_name=None,
                              label_top_n_predictions=5, title="PPI Network with Predicted Links", save_path=None):
    """
    Visualize a graph with predicted (novel, false positive) links highlighted.
    
    Args:
        model: The trained GNN model.
        data (torch_geometric.data.Data): PyG graph data with edge_label and edge_label_index.
        device: Device to run on (cuda/cpu).
        threshold (float): Confidence threshold for highlighting predictions.
        node_idx_to_name (dict, optional): Mapping for node labels.
        label_top_n_predictions (int): Label nodes involved in top N novel predictions.
        title (str): Plot title.
        save_path (str, optional): Path to save the figure.
    """
    model.eval()
    data_on_device = data.to(device)

    if not (hasattr(data_on_device, 'edge_label_index') and data_on_device.edge_label_index is not None and
            hasattr(data_on_device, 'edge_label') and data_on_device.edge_label is not None and
            hasattr(data_on_device, 'edge_index') and data_on_device.edge_index is not None):
        logging.error("Data object is missing required attributes for predicted links visualization.")
        return
        
    with torch.no_grad():
        logits = model(data_on_device)
        pred_scores = torch.sigmoid(logits).cpu() # Keep as tensor for now
    
    edge_label_index_np = data.edge_label_index.cpu().numpy()
    edge_labels_np = data.edge_label.cpu().numpy() # True labels for supervision edges
    
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    # Add message-passing edges (original graph structure)
    G.add_edges_from(data.edge_index.cpu().numpy().T, type='message_passing')
    
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    plt.figure(figsize=(12, 10))
    
    # Draw nodes and message-passing edges
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='lightblue', alpha=0.8)
    message_edges = [(u,v) for u,v,d in G.edges(data=True) if d['type']=='message_passing']
    nx.draw_networkx_edges(G, pos, edgelist=message_edges, width=0.5, alpha=0.3, edge_color='gray', label='Original Edges (for GNN messages)')

    # Identify and draw predicted novel links (high confidence on originally negative samples)
    novel_pred_edges = []
    novel_pred_scores_for_label = [] # For labeling
    
    # Iterate through supervised edges
    for i in range(edge_label_index_np.shape[1]):
        src, dst = edge_label_index_np[0, i], edge_label_index_np[1, i]
        is_true_positive_link = edge_labels_np[i] == 1 # This was a known positive link in the supervision set
        score = pred_scores[i].item()

        if not is_true_positive_link and score >= threshold: # Predicted as link, but was a negative sample
            novel_pred_edges.append((src, dst))
            novel_pred_scores_for_label.append({'nodes':(src,dst), 'score':score})
        elif is_true_positive_link and score >=threshold: # Correctly predicted known positive link (Optional to draw differently)
            # Could draw these as, e.g., thicker black lines if desired
            pass


    if novel_pred_edges:
        nx.draw_networkx_edges(G, pos, edgelist=novel_pred_edges, width=2.0, alpha=0.8, 
                              edge_color='red', style='dashed', label=f'Novel Predicted Links (Score >= {threshold})')

    # Node labels for top novel predictions
    labels = {}
    if node_idx_to_name and label_top_n_predictions > 0 and novel_pred_scores_for_label:
        # Sort predictions by score to label the top ones
        novel_pred_scores_for_label.sort(key=lambda x: x['score'], reverse=True)
        nodes_to_label = set()
        for item in novel_pred_scores_for_label[:label_top_n_predictions]:
            nodes_to_label.add(item['nodes'][0])
            nodes_to_label.add(item['nodes'][1])
        for node_idx in nodes_to_label:
            labels[node_idx] = node_idx_to_name.get(node_idx, f"Idx {node_idx}")
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=9, font_color='darkred')

    plt.title(title, fontsize=16)
    plt.axis('off')
    handles, plot_labels = plt.gca().get_legend_handles_labels()
    # Create proxy artists for legend if they don't exist from draw_networkx_edges
    if not any('Original Edges' in lbl for lbl in plot_labels):
        handles.append(plt.Line2D([0], [0], color='gray', lw=0.5, label='Original Edges (for GNN messages)'))
    if not any('Novel Predicted Links' in lbl for lbl in plot_labels) and novel_pred_edges:
         handles.append(plt.Line2D([0], [0], color='red', linestyle='--', lw=2, label=f'Novel Predicted Links (Score >= {threshold})'))
    plt.legend(handles=handles, loc="upper right", fontsize=10)

    plt.figtext(0.5, 0.01, "Red dashed lines: high-confidence novel predictions.", ha="center", fontsize=10,
               bbox={"facecolor":"lightcoral", "alpha":0.2, "pad":5})
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Predicted links visualization saved to {save_path}")
    plt.show()


def visualize_degree_distribution(data, title="Node Degree Distribution in PPI Network", save_path=None):
    """Visualize the degree distribution of nodes in the graph (using message-passing edges)."""
    if data.num_nodes == 0:
        logging.warning("Graph has no nodes. Skipping degree distribution.")
        return

    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    # Use data.edge_index for degree calculation, as this represents the graph structure used by GNN
    if data.edge_index is not None:
        G.add_edges_from(data.edge_index.cpu().numpy().T)
    
    degrees = [d for _, d in G.degree()]
    if not degrees: # If no edges, all degrees are 0
        degrees = [0] * data.num_nodes if data.num_nodes > 0 else [0]


    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=max(1, min(30, len(set(degrees)))), alpha=0.75, color='skyblue', edgecolor='black') # Ensure bins are reasonable
    
    plt.title(title, fontsize=16)
    plt.xlabel('Node Degree (Number of Interactions)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    mean_degree = np.mean(degrees) if degrees else 0
    median_degree = np.median(degrees) if degrees else 0
    
    plt.axvline(mean_degree, color='r', linestyle='--', label=f'Mean Degree: {mean_degree:.2f}')
    plt.axvline(median_degree, color='g', linestyle='--', label=f'Median Degree: {median_degree:.2f}')
    plt.legend(fontsize=10)
    
    plt.figtext(0.5, 0.01, "Distribution of protein connectivity. Power-law like is common.", ha="center", fontsize=10,
               bbox={"facecolor":"lightcoral", "alpha":0.2, "pad":5})
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Degree distribution plot saved to {save_path}")
    plt.show()

