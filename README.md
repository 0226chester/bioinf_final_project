# Mouse PPI-GVAE: Protein Complex Subnetwork Synthesis

## Project Overview
This repository contains the implementation of a Graph Variational Autoencoder (GVAE) approach for synthesizing biologically meaningful protein complex subnetworks from mouse protein-protein interaction (PPI) data.

**Research Question:** Can a GVAE learn and generate biologically meaningful protein complex subnetworks from mouse PPI data, and how do these synthetic complexes compare to known experimental complexes?

## Key Features
- Graph-based generative modeling of protein complexes
- Integration of topological and biological (GO terms) node features
- Comparison against known complexes and random baseline
- Biological validation of generated complexes

## Datasets
- **STRING database**: Mouse protein-protein interactions (combined score ≥ 0.7)
- **Gene Ontology**: Functional annotations for mouse proteins
- **CORUM database**: Known protein complexes for validation

## Repository Structure
- `data/`: Raw and processed data
- `notebooks/`: Jupyter notebooks for exploration and analysis
- `src/`: Source code for data processing, models, and evaluation
- `configs/`: Configuration files
- `scripts/`: Executable scripts
- `tests/`: Unit tests
- `results/`: Saved models, figures, and predictions
- `docs/`: Project documentation

## Setup and Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/mouse-ppi-gvae.git
cd mouse-ppi-gvae

# Create and activate a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

### Data Preparation
```bash
# Download required data
python scripts/download_data.py

# Preprocess the data
python scripts/preprocess_data.py
```

## Usage

### Training the Model
```bash
python scripts/train_model.py --config configs/model_config.yaml
```

### Generating Protein Complexes
```bash
python scripts/generate_complexes.py --model results/models/gvae_model_best.pt --num_complexes 100
```

### Evaluation
```bash
python scripts/evaluate_model.py --predictions results/predictions/generated_complexes.pkl
```

## Visualization
The `notebooks/` directory contains Jupyter notebooks for visualizing:
- Network structures of real vs. generated complexes
- Functional coherence through GO term enrichment
- Topological property comparison metrics

## Project Approach
1. **Data preprocessing**: Filter STRING PPI data (score ≥ 0.7) and convert to binary interactions
2. **Feature engineering**: Combine topological metrics with GO term-based biological features
3. **Model training**: Train GVAE on known protein complexes
4. **Complex generation**: Generate novel protein complex candidates
5. **Evaluation**: Compare against known complexes and random subgraph baseline

## Citation
If you use this code in your research, please cite our work:
```
@article{your-name2025gvae,
  title={Generating Biologically Meaningful Protein Complexes with Graph Variational Autoencoders},
  author={Your Name and Contributors},
  journal={bioRxiv},
  year={2025}
}
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributors
- Your Name (@yourusername)