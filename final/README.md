# Protein-Protein Interaction Link Prediction

This project implements an improved graph neural network (GNN) model for predicting protein-protein interactions (PPIs) using PyTorch Geometric. The implementation includes model architecture enhancements, training techniques, evaluation metrics, visualization tools, and biological validation approaches.

## Project Structure

- `models.py`: Improved GNN architectures for link prediction
- `train.py`: Enhanced training functions with early stopping and learning rate scheduling
- `evaluate.py`: Evaluation metrics and functions for link prediction
- `visualize.py`: Network and embedding visualization tools
- `bio_validation.py`: Biological validation and interpretation of results
- `main.py`: Main script that ties everything together
- `README.md`: This file

## Requirements

- Python 3.7+
- PyTorch
- PyTorch Geometric
- NetworkX
- scikit-learn
- matplotlib
- numpy
- pandas

## Quick Start

1. **Install dependencies**:
   ```bash
   pip install torch torch_geometric networkx scikit-learn matplotlib numpy pandas
   ```

2. **Run the main script**:
   ```bash
   python main.py --visualize
   ```

3. **Results**:
   All results (model, plots, reports) will be saved in the `results/` directory.

## Command Line Arguments

```
usage: main.py [-h] [--hidden_channels HIDDEN_CHANNELS]
               [--embed_channels EMBED_CHANNELS] [--dropout DROPOUT]
               [--use_gat] [--use_mlp_predictor] [--lr LR]
               [--epochs EPOCHS] [--patience PATIENCE] [--seed SEED]
               [--output_dir OUTPUT_DIR] [--no_cuda] [--visualize]

PPI Link Prediction

optional arguments:
  -h, --help            show this help message and exit
  --hidden_channels HIDDEN_CHANNELS
                        Number of hidden channels
  --embed_channels EMBED_CHANNELS
                        Number of embedding channels
  --dropout DROPOUT     Dropout rate
  --use_gat             Use GAT instead of GraphSAGE
  --use_mlp_predictor   Use MLP predictor instead of dot product
  --lr LR               Learning rate
  --epochs EPOCHS       Number of epochs
  --patience PATIENCE   Early stopping patience
  --seed SEED           Random seed
  --output_dir OUTPUT_DIR
                        Output directory
  --no_cuda             Disable CUDA
  --visualize           Generate visualizations
```

## Model Architecture

The project implements an enhanced Graph Neural Network with:

1. **Multiple GNN layers**: GraphSAGE or GAT with residual connections
2. **Batch normalization**: For training stability
3. **Dropout**: To prevent overfitting
4. **Link prediction options**: 
   - Simple dot product for efficiency
   - MLP predictor for more complex interactions

## Training Improvements

- **Early stopping**: Prevents overfitting by monitoring validation performance
- **Learning rate scheduling**: Adapts learning rate during training
- **Gradient clipping**: Prevents exploding gradients

## Biological Validation and Interpretation

The code provides biological context through:

1. **Network topology analysis**: Identifies hub proteins and network properties
2. **Functional module detection**: Uses embeddings to find protein complexes
3. **Prediction pattern analysis**: Examines how predictions relate to protein connectivity
4. **Visualization tools**: Shows network structure, embeddings, and predictions

## Example Outputs

The pipeline generates several outputs:

1. **Evaluation reports**: Metrics on link prediction performance
2. **Network visualizations**: Graph structure and predicted links
3. **Embedding visualizations**: t-SNE plots of protein embeddings
4. **Biological analyses**: Reports on network topology and prediction patterns

## Extending the Project

To incorporate additional biological data:

1. Add functions in `bio_validation.py` to process external data (GO annotations, expression data, etc.)
2. Modify the model in `models.py` to accept additional features
3. Update the main script to include the new data sources

## References

- **PyTorch Geometric**: https://pytorch-geometric.readthedocs.io/
- **Protein-Protein Interaction Networks**: https://www.nature.com/articles/nrg3552
- **Graph Neural Networks**: https://arxiv.org/abs/1812.08434
