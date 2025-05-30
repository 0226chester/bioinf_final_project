# evaluation/metrics.py
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt


@torch.no_grad()
def evaluate_model(model, test_graphs, device, threshold=0.5):
    """
    Comprehensive evaluation of the link prediction model.
    
    Args:
        model: The trained GNN model
        test_graphs: List of test graphs
        device: Device to run on (cuda/cpu)
        threshold: Probability threshold for positive prediction
        
    Returns:
        Dictionary with various evaluation metrics
    """
    model.eval()
    all_preds = []
    all_labels = []
    all_edge_indices = []
    
    # Collect predictions and labels
    for data in test_graphs:
        data = data.to(device)
        if data.edge_label_index is None or data.edge_label_index.numel() == 0:
            continue
        if data.edge_index is None or data.edge_index.numel() == 0:
            continue

        # Get predictions
        logits = model(data)
        preds = torch.sigmoid(logits)
        
        # Store predictions, labels and edge indices
        all_preds.append(preds.cpu())
        all_labels.append(data.edge_label.cpu().float())
        all_edge_indices.append(data.edge_label_index.cpu())

    if not all_labels or not all_preds:
        print("Warning: No data for evaluation")
        return {}

    # Concatenate results
    final_preds = torch.cat(all_preds, dim=0).numpy()
    final_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Calculate metrics
    metrics = {}
    
    if len(np.unique(final_labels)) < 2:
        print("WARNING: Only one class present in labels for evaluation. Some metrics may be ill-defined or default to 0/NaN.")
    else:
        # Proceed with metric calculation
        metrics['auc'] = roc_auc_score(final_labels, final_preds)
        metrics['ap'] = average_precision_score(final_labels, final_preds)
    
    # Binary predictions using threshold
    binary_preds = (final_preds >= threshold).astype(np.int32)
    
    # Confusion matrix
    tp = np.sum((binary_preds == 1) & (final_labels == 1))
    fp = np.sum((binary_preds == 1) & (final_labels == 0))
    tn = np.sum((binary_preds == 0) & (final_labels == 0))
    fn = np.sum((binary_preds == 0) & (final_labels == 1))
    
    # Derived metrics
    metrics['accuracy'] = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0 # Added check for zero division
    metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
    metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
    metrics['f1'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    
    # Count prediction statistics
    metrics['total_edges'] = len(final_labels)
    metrics['positive_edges'] = np.sum(final_labels == 1)
    metrics['negative_edges'] = np.sum(final_labels == 0)
    metrics['predicted_positive'] = np.sum(binary_preds == 1)
    
    # Get high-confidence predictions for novel interactions
    # (predicted positive for negative examples)
    high_conf_mask = (final_preds >= 0.9) & (final_labels == 0)
    metrics['high_conf_novel'] = np.sum(high_conf_mask)
    
    return metrics


def get_novel_predictions(model, test_graphs, device, threshold=0.9, top_k=20):
    """
    Get the most confident novel link predictions.
    
    Args:
        model: The trained GNN model
        test_graphs: List of test graphs
        device: Device to run on (cuda/cpu)
        threshold: Confidence threshold
        top_k: Number of top predictions to return
        
    Returns:
        List of dictionaries with prediction details
    """
    model.eval()
    novel_predictions = []
    
    for graph_idx, data in enumerate(test_graphs):
        data = data.to(device)
        if data.edge_label_index is None or data.edge_label_index.numel() == 0:
            continue
        if data.edge_index is None or data.edge_index.numel() == 0: # Check for message passing edges
            continue

        # Get predictions
        with torch.no_grad():
            logits = model(data)
            preds = torch.sigmoid(logits)
        
        # Get indices of edges being predicted
        src_nodes = data.edge_label_index[0].cpu().numpy()
        dst_nodes = data.edge_label_index[1].cpu().numpy()
        labels = data.edge_label.cpu().numpy()
        scores = preds.cpu().numpy()
        
        # Find negative edges (labels=0) with high prediction scores
        for i in range(len(labels)):
            if labels[i] == 0 and scores[i] >= threshold:
                novel_predictions.append({
                    'graph_idx': graph_idx,
                    'source_node': int(src_nodes[i]),
                    'target_node': int(dst_nodes[i]),
                    'confidence': float(scores[i])
                })
    
    # Sort by confidence (descending)
    novel_predictions.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Return top-k predictions
    return novel_predictions[:top_k]


def generate_evaluation_report(metrics, novel_predictions=None, filepath=None):
    """
    Generate a text report with evaluation results.
    
    Args:
        metrics: Dictionary with evaluation metrics
        novel_predictions: List of novel predictions (optional)
        filepath: Optional path to save the report
    """
    report = [
        "=== PROTEIN-PROTEIN INTERACTION LINK PREDICTION EVALUATION ===",
        "",
        f"Total test edges: {metrics.get('total_edges', 'N/A')}",
        f"Positive edges: {metrics.get('positive_edges', 'N/A')}",
        f"Negative edges: {metrics.get('negative_edges', 'N/A')}",
        "",
        "--- Performance Metrics ---",
        f"ROC AUC: {metrics.get('auc', 'N/A'):.4f}",
        f"Average Precision: {metrics.get('ap', 'N/A'):.4f}",
        f"Accuracy: {metrics.get('accuracy', 'N/A'):.4f}",
        f"Precision: {metrics.get('precision', 'N/A'):.4f}",
        f"Recall: {metrics.get('recall', 'N/A'):.4f}",
        f"F1 Score: {metrics.get('f1', 'N/A'):.4f}",
        "",
        "--- Prediction Statistics ---",
        f"Predicted positive interactions: {metrics.get('predicted_positive', 'N/A')}",
        f"High-confidence novel interactions: {metrics.get('high_conf_novel', 'N/A')}",
        "",
    ]
    
    # Add novel predictions if provided
    if novel_predictions:
        report.extend([
            "--- Top Novel Protein Interaction Predictions ---",
            "These predictions represent potential undiscovered PPIs that could be",
            "prioritized for experimental validation:",
            "0-based indexing is used for protein IDs.",
            ""
        ])
        
        for i, pred in enumerate(novel_predictions[:10], 1):  # Show top 10
            report.append(f"{i}. Protein {pred['source_node']} → Protein {pred['target_node']} " # 0-based
                          f"(Confidence: {pred['confidence']:.4f})")
        
        report.extend([
            "",
            "Biological Interpretation:",
            "- High-confidence predictions may represent proteins in the same complex",
            "- These proteins likely participate in related biological processes",
            "- Consider experimental validation using co-immunoprecipitation or yeast two-hybrid"
        ])
    
    # Join report lines
    full_report = "\n".join(report)
    print(full_report)
    
    # Save to file if filepath provided
    if filepath:
        with open(filepath, 'w') as f:
            f.write(full_report)
    
    return full_report