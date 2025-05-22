import torch
import time
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt


def train(model, graphs, optimizer, criterion, device):
    """
    Standard training function for one epoch.
    
    Args:
        model: The GNN model
        graphs: List of PPI graphs
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to run on (cuda/cpu)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0
    
    for data in graphs:
        data = data.to(device)
        # Ensure edge_label_index exists and is not empty for supervision
        if not hasattr(data, 'edge_label_index') or data.edge_label_index is None or data.edge_label_index.numel() == 0:
            print(f"Skipping graph due to missing or empty edge_label_index. Graph: {data}")
            continue
        # Ensure edge_label exists for supervision
        if not hasattr(data, 'edge_label') or data.edge_label is None:
            print(f"Skipping graph due to missing edge_label. Graph: {data}")
            continue
        # Ensure edge_index exists for message passing
        if data.edge_index is None or data.edge_index.numel() == 0:
            # Depending on the model, this might be acceptable if it can handle graphs with no edges,
            # or it might be an issue. For GNNs, it's usually expected.
            # If the model requires edges, this graph should probably be skipped or handled.
            # For now, let's assume the model can process it or it's an edge case.
            # If this causes issues downstream (e.g., in model forward pass), add a continue.
            pass # Or print a warning, or continue

        optimizer.zero_grad()
        logits = model(data) # model expects data.x, data.edge_index, and for link prediction, data.edge_label_index
        
        # Ensure edge_label is not empty if edge_label_index is not
        if data.edge_label.numel() == 0 and data.edge_label_index.numel() > 0:
            print(f"Skipping graph due to empty edge_label despite non-empty edge_label_index. Graph: {data}")
            continue

        loss = criterion(logits, data.edge_label.float())
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        # Corrected: loss.item() is the loss for the current single graph.
        # data.num_graphs is not an attribute of a single Data object.
        total_loss += loss.item() 
        
    return total_loss / len(graphs) if graphs else 0


@torch.no_grad()
def test(model, graphs, device):
    """
    Evaluation function that calculates AUC and AP metrics.
    
    Args:
        model: The GNN model
        graphs: List of PPI graphs
        device: Device to run on (cuda/cpu)
        
    Returns:
        AUC and AP scores
    """
    model.eval()
    all_preds = []
    all_labels = []

    for data in graphs:
        data = data.to(device)
        # Ensure edge_label_index exists and is not empty for supervision
        if not hasattr(data, 'edge_label_index') or data.edge_label_index is None or data.edge_label_index.numel() == 0:
            continue
        # Ensure edge_label exists for supervision
        if not hasattr(data, 'edge_label') or data.edge_label is None:
            continue
        # Ensure edge_index exists for message passing (similar to train)
        if data.edge_index is None or data.edge_index.numel() == 0:
            pass # Or handle as in train

        logits = model(data)
        preds = torch.sigmoid(logits)

        # Ensure edge_label is not empty if edge_label_index is not
        if data.edge_label.numel() == 0 and data.edge_label_index.numel() > 0:
            continue
            
        all_preds.append(preds.cpu())
        all_labels.append(data.edge_label.cpu().float())

    if not all_labels or not all_preds: # Check if lists are empty
        print("Warning: No valid data found for testing, returning 0.0 for AUC and AP.")
        return 0.0, 0.0

    final_preds = torch.cat(all_preds, dim=0)
    final_labels = torch.cat(all_labels, dim=0)

    # Ensure there are both positive and negative samples for metric calculation
    if len(torch.unique(final_labels)) < 2:
        print(f"Warning: Only one class present in labels after processing all graphs. ROC AUC / AP score is not well-defined. Labels: {torch.unique(final_labels)}")
        # Depending on the case, you might return 0.0 or handle it differently.
        # For example, if all are positive and all predictions are high, AP might be 1.0.
        # However, roc_auc_score will raise an error.
        # For simplicity, returning 0.0 if metrics can't be computed robustly.
        return 0.0, 0.0
    
    # Check for NaN or Inf in predictions, which can happen with empty inputs or numerical issues
    if torch.isnan(final_preds).any() or torch.isinf(final_preds).any():
        print("Warning: NaN or Inf found in predictions. ROC AUC / AP might be incorrect.")
        return 0.0, 0.0


    auc = roc_auc_score(final_labels.numpy(), final_preds.numpy())
    ap = average_precision_score(final_labels.numpy(), final_preds.numpy())
    return auc, ap


def train_with_early_stopping(model, train_graphs, val_graphs, test_graphs, 
                             optimizer, criterion, device, 
                             patience=10, n_epochs=100, 
                             lr_scheduler=None, verbose=True):
    """
    Training procedure with early stopping, learning rate scheduling,
    and progress tracking.
    
    Args:
        model: The GNN model
        train_graphs: Training graphs
        val_graphs: Validation graphs
        test_graphs: Test graphs
        optimizer: PyTorch optimizer
        criterion: Loss function
        device: Device to run on (cuda/cpu)
        patience: Early stopping patience
        n_epochs: Maximum number of epochs
        lr_scheduler: Learning rate scheduler (optional)
        verbose: Whether to print progress
        
    Returns:
        Best model and training history
    """
    # Initialize tracking variables
    best_val_ap = 0
    best_val_auc = 0 # Keep track of AUC corresponding to best AP
    test_auc_at_best_val_ap = 0 # Test AUC when val_ap was best
    test_ap_at_best_val_ap = 0  # Test AP when val_ap was best
    best_epoch = 0
    counter = 0 # For early stopping
    best_model_state = None
    
    # History for plotting
    history = {
        'train_loss': [],
        'val_auc': [],
        'val_ap': [],
        'test_auc_at_best_val_ap': [], # To see test performance evolution when val_ap improves
        'test_ap_at_best_val_ap': [],  # To see test performance evolution when val_ap improves
        'lr': []
    }
    
    start_time = time.time()
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        epoch_start_time = time.time()
        # Train for one epoch
        loss = train(model, train_graphs, optimizer, criterion, device)
        history['train_loss'].append(loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Validate
        val_auc, val_ap = test(model, val_graphs, device)
        history['val_auc'].append(val_auc)
        history['val_ap'].append(val_ap)
        
        # Update learning rate
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(val_ap) # ReduceLROnPlateau typically monitors a metric
            else:
                lr_scheduler.step() # For other schedulers like StepLR

        epoch_duration = time.time() - epoch_start_time
        
        # Print progress
        if verbose and (epoch % 5 == 0 or epoch == 1 or epoch == n_epochs):
            print(f'Epoch: {epoch:03d}/{n_epochs}, Loss: {loss:.4f}, Val AUC: {val_auc:.4f}, Val AP: {val_ap:.4f}, '
                  f'LR: {current_lr:.6f}, Time: {epoch_duration:.2f}s')
        
        # Check if this is the best model so far based on validation AP
        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_val_auc = val_auc # Store AUC for this best AP
            best_epoch = epoch
            counter = 0 # Reset early stopping counter
            
            # Evaluate on test set with the current best model
            current_test_auc, current_test_ap = test(model, test_graphs, device)
            test_auc_at_best_val_ap = current_test_auc
            test_ap_at_best_val_ap = current_test_ap
            
            # Save best model state (on CPU to save GPU memory)
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if verbose:
                 print(f'---> New best model found at epoch {epoch:03d}: Val AP: {best_val_ap:.4f}, Val AUC: {best_val_auc:.4f} | Test AP: {test_ap_at_best_val_ap:.4f}, Test AUC: {test_auc_at_best_val_ap:.4f}')
        else:
            counter += 1
        
        history['test_auc_at_best_val_ap'].append(test_auc_at_best_val_ap if best_val_ap > 0 else 0)
        history['test_ap_at_best_val_ap'].append(test_ap_at_best_val_ap if best_val_ap > 0 else 0)
            
        # Early stopping
        if patience > 0 and counter >= patience:
            if verbose:
                print(f"Early stopping triggered at epoch {epoch} after {patience} epochs without improvement in Val AP.")
            break
    
    total_training_time = time.time() - start_time
    
    # Load best model state before returning
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.to(device) # Ensure model is on the correct device
    
    # Print summary
    if verbose:
        print(f"\nTraining completed in {total_training_time:.2f} seconds.")
        if best_model_state is not None:
            print(f"Best model found at epoch {best_epoch}:")
            print(f"  Best Validation AP: {best_val_ap:.4f}, Corresponding Val AUC: {best_val_auc:.4f}")
            print(f"  Test AP (at best Val AP epoch): {test_ap_at_best_val_ap:.4f}, Test AUC: {test_auc_at_best_val_ap:.4f}")
        else:
            print("No best model state was saved (e.g., training stopped early or no improvement).")
            # Optionally, evaluate the final model state if no best model was found
            final_test_auc, final_test_ap = test(model, test_graphs, device)
            print(f"  Final model Test AP: {final_test_ap:.4f}, Test AUC: {final_test_auc:.4f}")


    return model, history


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
    
    plt.figure(figsize=(18, 10)) # Adjusted figure size
    
    # Plot Training Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot Validation AUC
    plt.subplot(2, 3, 2)
    plt.plot(epochs, history['val_auc'], 'r-', label='Validation AUC')
    plt.title('Validation AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()
    
    # Plot Validation AP
    plt.subplot(2, 3, 3)
    plt.plot(epochs, history['val_ap'], 'g-', label='Validation AP')
    plt.title('Validation AP')
    plt.xlabel('Epochs')
    plt.ylabel('AP')
    plt.grid(True)
    plt.legend()
    
    # Plot Test AUC (at best validation AP epoch)
    plt.subplot(2, 3, 4)
    plt.plot(epochs, history['test_auc_at_best_val_ap'], 'c-', label='Test AUC (at best Val AP)')
    plt.title('Test AUC (at best Val AP)')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.grid(True)
    plt.legend()

    # Plot Test AP (at best validation AP epoch)
    plt.subplot(2, 3, 5)
    plt.plot(epochs, history['test_ap_at_best_val_ap'], 'm-', label='Test AP (at best Val AP)')
    plt.title('Test AP (at best Val AP)')
    plt.xlabel('Epochs')
    plt.ylabel('AP')
    plt.grid(True)
    plt.legend()

    # Plot Learning Rate
    plt.subplot(2, 3, 6)
    plt.plot(epochs, history['lr'], 'y-', label='Learning Rate')
    plt.title('Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('LR')
    plt.yscale('log') # Often useful for LR
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

