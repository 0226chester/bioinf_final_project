# visualization/plots.py
import torch
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.metrics import precision_recall_curve, average_precision_score
import logging

# It's assumed that 'identify_functional_modules' would be in a module like 'evaluation.analysis'
# For this script to be runnable, you'd adjust the import path based on your final project structure.
# Example: from ..evaluation.analysis import identify_functional_modules
# For now, if identify_functional_modules is needed by visualize_functional_modules,
# it should be made available to this script's scope, or that function might need to be
# refactored or also moved here if it's purely for visualization setup.
# Given its analytical nature (clustering), it's better placed in analysis.py.
# We will proceed assuming it can be imported.


def visualize_graph(data, node_colors=None, node_size_attr=None, node_idx_to_name=None,
                    label_top_n_nodes=0, title="PPI Network", node_base_size=50, save_path=None):
    """
    Visualize a PPI graph.
    
    Args:
        data (torch_geometric.data.Data): PyG graph data.
        node_colors (list/np.array/str, optional): Node colors.
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
    
    if node_size_attr is not None: # Node sizes
        sizes = np.array(node_size_attr)
        node_sizes = node_base_size + (sizes / np.max(sizes) if np.max(sizes) > 0 else sizes) * node_base_size * 2
    else:
        node_sizes = node_base_size
        
    color_map_name = 'viridis' # Node colors
    if isinstance(node_colors, (list, np.ndarray)) and not all(isinstance(c, str) for c in node_colors):
        pass # nx.draw_networkx_nodes will handle this with cmap
    elif node_colors is None:
        node_colors = 'lightblue'


    plt.figure(figsize=(12, 10))
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, cmap=plt.get_cmap(color_map_name) if isinstance(node_colors, (list, np.ndarray)) else None, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, edge_color='gray')
    
    labels = {} # Node labels
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
        color_by (np.array, optional): Array of values to color nodes by.
        cmap_name (str): Name of the Matplotlib colormap.
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
    if embeddings_np.shape[0] < perplexity : 
        perplexity = max(5, embeddings_np.shape[0] // 2 -1) 
        if perplexity < 5 : 
            logging.warning(f"Too few samples ({embeddings_np.shape[0]}) for meaningful t-SNE. Skipping embedding visualization.")
            return

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, init='pca', learning_rate='auto')
    node_coords = tsne.fit_transform(embeddings_np)
    
    plt.figure(figsize=(10, 8))
    
    node_colors_for_plot = color_by if color_by is not None else np.arange(data.num_nodes)
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
        pred_scores = torch.sigmoid(logits).cpu() 
    
    edge_label_index_np = data.edge_label_index.cpu().numpy()
    edge_labels_np = data.edge_label.cpu().numpy() 
    
    G = nx.Graph()
    G.add_nodes_from(range(data.num_nodes))
    G.add_edges_from(data.edge_index.cpu().numpy().T, type='message_passing')
    
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    plt.figure(figsize=(12, 10))
    
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color='lightblue', alpha=0.8)
    message_edges = [(u,v) for u,v,d in G.edges(data=True) if d['type']=='message_passing']
    nx.draw_networkx_edges(G, pos, edgelist=message_edges, width=0.5, alpha=0.3, edge_color='gray', label='Original Edges (for GNN messages)')

    novel_pred_edges = []
    novel_pred_scores_for_label = [] 
    
    for i in range(edge_label_index_np.shape[1]): # Iterate through supervised edges
        src, dst = edge_label_index_np[0, i], edge_label_index_np[1, i]
        is_true_positive_link = edge_labels_np[i] == 1 
        score = pred_scores[i].item()

        if not is_true_positive_link and score >= threshold: # Predicted as link, but was a negative sample
            novel_pred_edges.append((src, dst))
            novel_pred_scores_for_label.append({'nodes':(src,dst), 'score':score})
        elif is_true_positive_link and score >=threshold: 
            pass


    if novel_pred_edges:
        nx.draw_networkx_edges(G, pos, edgelist=novel_pred_edges, width=2.0, alpha=0.8, 
                              edge_color='red', style='dashed', label=f'Novel Predicted Links (Score >= {threshold})')

    labels = {} # Node labels for top novel predictions
    if node_idx_to_name and label_top_n_predictions > 0 and novel_pred_scores_for_label:
        novel_pred_scores_for_label.sort(key=lambda x: x['score'], reverse=True) # Sort predictions by score to label the top ones
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
    if not any('Original Edges' in lbl for lbl in plot_labels): # Create proxy artists for legend if they don't exist from draw_networkx_edges
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
    if data.edge_index is not None: # Use data.edge_index for degree calculation, as this represents the graph structure used by GNN
        G.add_edges_from(data.edge_index.cpu().numpy().T)
    
    degrees = [d for _, d in G.degree()]
    if not degrees: 
        degrees = [0] * data.num_nodes if data.num_nodes > 0 else [0]


    plt.figure(figsize=(10, 6))
    plt.hist(degrees, bins=max(1, min(30, len(set(degrees)))), alpha=0.75, color='skyblue', edgecolor='black') 
    
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


def plot_training_history(history, filepath=None):
    """
    Plot training metrics history.
    
    Args:
        history: Dictionary with training metrics
        filepath: Optional path to save the figure
    """
    num_epochs_trained = len(history['train_loss'])
    if num_epochs_trained == 0:
        print("No training history to plot.")
        return
        
    epochs = range(1, num_epochs_trained + 1)
    
    plt.figure(figsize=(18, 10))
    
    plt.subplot(2, 3, 1) # Plot Training Loss
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 3, 2) # Plot Validation AUC
    plt.plot(epochs, history['val_auc'], 'r-', label='Validation AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 3, 3) # Plot Validation AP
    plt.plot(epochs, history['val_ap'], 'g-', label='Validation AP')
    plt.title('Validation AP')
    plt.xlabel('Epochs')
    plt.ylabel('AP')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(2, 3, 4) # Plot Test AUC (at best validation AP epoch)
    plt.plot(epochs, history['test_auc_at_best_val_ap'], 'c-', label='Test AUC (at best Val AP)')
    plt.title('Test AUC (at best Val AP)')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 5) # Plot Test AP (at best validation AP epoch)
    plt.plot(epochs, history['test_ap_at_best_val_ap'], 'm-', label='Test AP (at best Val AP)')
    plt.title('Test AP (at best Val AP)')
    plt.xlabel('Epochs')
    plt.ylabel('AP')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 3, 6) # Plot Learning Rate
    plt.plot(epochs, history['lr'], 'y-', label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.yscale('log') 
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    if filepath:
        try:
            plt.savefig(filepath)
            print(f"Training history plot saved to {filepath}")
        except Exception as e:
            print(f"Error saving plot: {e}")
    
    plt.show()


def plot_precision_recall_curve(model, test_graphs, device, filepath=None):
    """
    Plot precision-recall curve for the model.
    
    Args:
        model: The trained GNN model
        test_graphs: List of test graphs
        device: Device to run on (cuda/cpu)
        filepath: Optional path to save the figure
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad(): 
        for data in test_graphs:
            data = data.to(device)
            if data.edge_label_index is None or data.edge_label_index.numel() == 0:
                continue
            if data.edge_index is None or data.edge_index.numel() == 0: 
                continue

            logits = model(data)
            preds = torch.sigmoid(logits)
            
            all_preds.append(preds.detach().cpu()) 
            all_labels.append(data.edge_label.cpu().float())

    if not all_labels or not all_preds : 
        print("No data for plotting precision-recall curve.") 
        return

    final_preds = torch.cat(all_preds, dim=0).numpy()
    final_labels = torch.cat(all_labels, dim=0).numpy()

    if len(np.unique(final_labels)) < 2: 
        print("WARNING: Only one class present in labels for PR curve. Plotting may not be meaningful.")
    
    precision, recall, thresholds = precision_recall_curve(final_labels, final_preds)
    ap = average_precision_score(final_labels, final_preds)
    
    plt.figure(figsize=(10, 7))
    plt.plot(recall, precision, 'b-', linewidth=2)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title(f'Precision-Recall Curve (AP = {ap:.4f})', fontsize=16)
    
    if len(final_labels) > 0:
        baseline_precision = np.sum(final_labels == 1) / len(final_labels) if len(final_labels) > 0 else 0.5
        plt.plot([0, 1], [baseline_precision] * 2, 'r--', linewidth=2, 
                 label=f'Baseline (Prevalence = {baseline_precision:.2f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    plt.figtext(0.5, 0.01, 
                "High precision values indicate reliable PPI predictions.\n"
                "The gap between the model curve and baseline shows the biological relevance of the model.",
                ha="center", fontsize=10, bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    if filepath:
        plt.savefig(filepath)
    plt.show()


def visualize_functional_modules(data, node_embeddings, save_path=None, clustering_method='kmeans', 
                                 # Assuming identify_functional_modules is imported or available
                                 identify_functional_modules_func=None, 
                                 **cluster_kwargs):
    """
    Visualize protein functional modules based on embeddings.
    
    Args:
        data (torch_geometric.data.Data): PyG graph data.
        node_embeddings (torch.Tensor): Node embeddings from GNN.
        save_path (str, optional): Optional path to save the figure.
        clustering_method (str): 'kmeans' or 'dbscan'.
        identify_functional_modules_func (function): The actual function to identify modules.
        **cluster_kwargs: Additional arguments for the chosen clustering method.
    """
    if node_embeddings is None or node_embeddings.numel() == 0:
        logging.warning("Node embeddings are empty. Cannot visualize functional modules.")
        return {}

    if identify_functional_modules_func is None:
        logging.error("identify_functional_modules_func not provided to visualize_functional_modules.")
        return {}

    modules_dict, num_effective_modules = identify_functional_modules_func(data, node_embeddings, method=clustering_method, **cluster_kwargs)

    if not modules_dict or num_effective_modules == 0:
        logging.warning("No modules identified or num_effective_modules is 0. Skipping visualization.")
        if data.num_nodes > 0 and not modules_dict: 
            modules_dict = {"Module 1": list(range(data.num_nodes))}
            num_effective_modules = 1
        else:
            return modules_dict

    num_colors_needed = max(1, num_effective_modules) 
    cmap = plt.cm.get_cmap('viridis', num_colors_needed) 
    
    node_colors_by_module_idx = np.zeros(data.num_nodes) 
    module_idx_map = {} 
    current_color_idx = 0

    for module_name, node_indices in modules_dict.items():
        if "Noise" in module_name: 
            for node_idx in node_indices:
                node_colors_by_module_idx[node_idx] = -1 
        else:
            if module_name not in module_idx_map:
                module_idx_map[module_name] = current_color_idx
                current_color_idx +=1
            color_idx_for_cmap = module_idx_map[module_name] % num_colors_needed 
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
        color_for_module = cmap(assigned_module_idx % num_colors_needed) 
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_module, node_size=80, 
                               node_color=[color_for_module], label=module_name, alpha=0.8)

    if "Noise (DBSCAN)" in modules_dict: 
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
    plt.show()
    return modules_dict # This function is for visualization, returning dict is optional


def plot_prediction_pattern_statistics(results, save_path=None):
    """Visualize the prediction pattern statistics."""
    if not results or 'total_predictions_analyzed' not in results or results['total_predictions_analyzed'] == 0:
        logging.warning("Insufficient data for prediction pattern visualization.")
        return
    
    categories = ['Hub-Hub', 'Hub-Peripheral', 'Peripheral-Peripheral']
    map_keys = ['high_to_high', 'high_to_low', 'low_to_low']
    
    counts = [results.get(k, {}).get('count', 0) for k in map_keys]
    avg_scores = [results.get(k, {}).get('avg_score', 0) for k in map_keys]
    precisions = [results.get(k, {}).get('precision', 0) for k in map_keys]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6)) 
    
    axes[0].bar(categories, counts, color=['#ff9999', '#66b3ff', '#99ff99']) 
    axes[0].set_title('Number of Supervised Edges', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=10)
    for i, v in enumerate(counts): axes[0].text(i, v + 0.5, str(v), ha='center', va='bottom', fontsize=9)

    axes[1].bar(categories, avg_scores, color=['#ffadad', '#a0c4ff', '#b3ffb3']) 
    axes[1].set_title('Average Confidence Score', fontsize=12)
    axes[1].set_ylabel('Avg. Confidence', fontsize=10)
    axes[1].set_ylim(0, 1.0)
    for i, v in enumerate(avg_scores): axes[1].text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    axes[2].bar(categories, precisions, color=['#ffd6a5', '#caffbf', '#9bf6ff']) 
    axes[2].set_title('Precision within Category', fontsize=12)
    axes[2].set_ylabel('Precision', fontsize=10)
    axes[2].set_ylim(0, 1.0)
    for i, v in enumerate(precisions): axes[2].text(i, v + 0.01, f"{v:.3f}", ha='center', va='bottom', fontsize=9)

    for ax in axes:
        ax.tick_params(axis='x', rotation=15, labelsize=9)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.suptitle("Prediction Pattern Statistics by Node Connectivity", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95]) 
    
    plt.figtext(0.5, 0.01, 
               f"Analysis based on degree threshold at {results.get('degree_threshold_value', 'N/A'):.2f} (connectivity).",
               ha="center", fontsize=10, bbox={"facecolor":"lightcoral", "alpha":0.2, "pad":3})
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        logging.info(f"Prediction pattern plot saved to {save_path}")
    plt.show()