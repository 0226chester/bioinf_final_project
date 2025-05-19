# Mouse PPI Network Preprocessing Documentation

This document explains how to use the preprocessing modules for the GVAE Protein Complex Subnetwork Synthesis Project. The preprocessing pipeline transforms raw biological data from STRING, CORUM, and GO databases into formatted datasets suitable for graph neural network training.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Data Directory Structure](#data-directory-structure)
4. [Preprocessing Workflow](#preprocessing-workflow)
5. [Module Descriptions](#module-descriptions)
6. [Command-line Usage](#command-line-usage)
7. [Parameters Reference](#parameters-reference)
8. [Logging System](#logging-system)
9. [Troubleshooting](#troubleshooting)
10. [Examples](#examples)

## Overview

The preprocessing pipeline consists of several modules that perform the following operations:

1. Parse protein alias mappings from STRING database
2. Process protein information and annotations
3. Build the protein-protein interaction (PPI) network with confidence filtering
4. Extract protein complexes from CORUM database
5. Process Gene Ontology (GO) annotations
6. Calculate topological features for each protein in the network
7. Create GO-based feature vectors
8. Apply dimensionality reduction to GO features
9. Combine topological and GO features into node feature vectors
10. Export all processed data in appropriate formats

## Prerequisites

Before running the preprocessing pipeline, ensure you have:

- Required Python packages:
- Downloaded raw data files from:
  - STRING database (version 12.0 for Mus musculus)
  - CORUM database (all complexes)
  - MGI Gene Ontology Annotation (GAF) file

## Data Directory Structure

The expected directory structure is:

```
project_root/
├── data/
│   ├── raw/                           # Raw data files
│   │   ├── 10090.protein.aliases.v12.0.txt
│   │   ├── 10090.protein.info.v12.0.txt
│   │   ├── 10090.protein.links.v12.0.txt
│   │   ├── corum_allComplexes.txt
│   │   └── mgi.gaf
│   └── processed/                     # Output of preprocessing
│       ├── features/                  # Feature vectors
│       │   ├── node_features.npy
│       │   ├── node_ids.txt
│       │   └── feature_info.txt
│       ├── complexes/                 # Protein complexes
│       ├── ppi_network.edgelist       # Network structure
│       ├── network_stats.txt          # Network statistics
│       └── protein_info.tsv           # Protein metadata
├── logs/                              # Log files
└── src/                               # Source code
    └── data/                          # Data processing modules
        ├── preprocessing.py           # Basic data processing
        ├── feature_engineering.py     # Feature creation
        └── dataloader.py              # PyTorch data loading
```

## Preprocessing Workflow

The complete preprocessing workflow is executed by the `preprocess_data.py` script, which performs the following steps:

1. **Data Validation** - Checks that all required input files exist
2. **Protein Alias Processing** - Creates mappings between different ID systems
3. **Protein Information Processing** - Extracts metadata for each protein
4. **PPI Network Construction** - Builds graph with confidence filtering 
5. **Protein Complex Extraction** - Identifies complexes from CORUM
6. **GO Annotation Processing** - Processes biological annotations
7. **Topological Feature Calculation** - Computes network metrics
8. **GO Feature Engineering** - Creates feature vectors from GO terms
9. **Dimensionality Reduction** - Applies PCA to GO features
10. **Feature Vector Creation** - Combines all features
11. **Data Export** - Saves all processed data in appropriate formats

## Module Descriptions

### preprocessing.py

This module handles basic data processing operations:

- `process_aliases()` - Creates mapping dictionaries between different ID types
- `process_protein_info()` - Extracts protein names and metadata
- `build_ppi_network()` - Constructs protein interaction network
- `process_corum_complexes()` - Extracts protein complexes
- `process_go_annotations()` - Processes Gene Ontology annotations
- `export_network_data()` - Exports processed network data

### feature_engineering.py

This module creates node feature vectors:

- `calculate_topology_features()` - Computes centrality measures
- `create_go_feature_vectors()` - Creates GO term embeddings
- `reduce_go_dimensions()` - Applies dimensionality reduction to GO features
- `create_combined_feature_vectors()` - Combines topological and GO features
- `export_feature_data()` - Exports node feature vectors

### dataloader.py

This module provides PyTorch-compatible data loading:

- `PPINetworkDataset` - Dataset class for PPI network data
- `get_complex_dataloader()` - Creates DataLoader for protein complexes
- `load_ppi_network()` - Loads the full PPI network

## Command-line Usage

### Full Preprocessing Pipeline

To run the entire preprocessing pipeline:

```bash
python scripts/preprocess_data.py
```

With custom parameters:

```bash
python scripts/preprocess_data.py \
  --raw_dir=data/raw \
  --processed_dir=data/processed \
  --confidence_threshold=800 \
  --pca_components=15 \
  --go_aspects=PC \
  --log_dir=logs \
  --verbose \
  --log_to_file
```

### Testing the Dataloader

To test the dataloader functionality:

```bash
python -m src.data.dataloader \
  --data-dir=data/processed \
  --max-nodes=20 \
  --batch-size=16 \
  --log-file=logs/dataloader_test.log \
  --verbose
```

## Parameters Reference

### preprocess_data.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--raw_dir` | Directory containing raw data files | `data/raw` |
| `--processed_dir` | Directory to save processed data | `data/processed` |
| `--aliases_file` | STRING protein aliases file | `10090.protein.aliases.v12.0.txt` |
| `--info_file` | STRING protein info file | `10090.protein.info.v12.0.txt` |
| `--links_file` | STRING protein links file | `10090.protein.links.v12.0.txt` |
| `--corum_file` | CORUM protein complexes file | `corum_allComplexes.txt` |
| `--mgi_file` | MGI gene ontology annotation file | `mgi.gaf` |
| `--confidence_threshold` | Confidence threshold for PPI network (0-1000) | `700` |
| `--pca_components` | Number of PCA components for GO term reduction | `10` |
| `--go_aspects` | GO aspects to use (P,F,C) | `C` |
| `--log_dir` | Directory to save log files | `logs` |
| `--verbose` | Enable verbose (DEBUG) logging | `False` |
| `--log_to_file` | Save logs to file in addition to console output | `False` |

### dataloader.py

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--data-dir` | Directory containing processed data | `data/processed` |
| `--max-nodes` | Maximum number of nodes per complex | `20` |
| `--batch-size` | Batch size for dataloader | `16` |
| `--log-file` | Log file path | `None` |
| `--verbose` | Enable verbose logging | `False` |

## Logging System

The preprocessing modules use a comprehensive logging system that:

1. Logs to both console and file (if specified)
2. Includes timestamps, log levels, and component information
3. Provides detailed statistics during processing
4. Tracks timing information for each step
5. Records warnings and errors with context information

Log levels:
- `INFO` - Normal processing information
- `DEBUG` - Detailed debugging information (enabled with `--verbose`)
- `WARNING` - Potential issues that don't stop processing
- `ERROR` - Critical issues that prevent further processing

## Troubleshooting

### Common Issues

1. **Missing files**: 
   - Ensure all required input files are in the raw data directory
   - Verify file names match the expected names or use parameters to specify custom names

2. **Memory errors during network construction**: 
   - For very large networks, consider increasing the `--confidence_threshold` 
   - For betweenness centrality calculation, the approximation parameter can be adjusted in the code

3. **Empty network**:
   - Check if the confidence threshold is too high
   - Verify that the input links file follows the STRING format

4. **No protein complexes extracted**:
   - Verify the CORUM file format
   - Check if mouse complexes are present in the file
   - Ensure the mapping between UniProt and STRING IDs is correct

5. **Log files not created**:
   - Ensure the log directory exists and is writable
   - Use the `--log_to_file` flag to enable file logging

### Checking Progress

The logs provide detailed information on each processing step:

```
INFO - Starting Mouse PPI network preprocessing
INFO - Raw data directory: /path/to/data/raw
INFO - Processed data directory: /path/to/data/processed
INFO - Confidence threshold: 700
INFO - Step 1/11: Processing protein aliases
INFO - Created 52847 STRING to UniProt mappings
INFO - Completed in 5.23 seconds
...
```

## Examples

### Basic Run with Default Parameters

```bash
python scripts/preprocess_data.py
```

This will:
- Look for raw data in `data/raw`
- Save processed data to `data/processed`
- Use a confidence threshold of 700
- Use only the Cellular Component (C) GO aspect
- Reduce GO features to 10 dimensions
- Log at INFO level to console only

### Custom Run for Higher Precision Network

```bash
python scripts/preprocess_data.py \
  --confidence_threshold=900 \
  --go_aspects=PC \
  --pca_components=20 \
  --verbose \
  --log_to_file
```

This will:
- Create a higher-precision network (confidence score ≥ 900)
- Use both Process (P) and Component (C) GO aspects
- Create richer feature vectors (20 PCA components)
- Enable detailed debugging logs
- Save logs to a timestamped file in the logs directory

### Processing Custom Data Files

```bash
python scripts/preprocess_data.py \
  --raw_dir=data/custom_data \
  --aliases_file=my_aliases.txt \
  --info_file=my_info.txt \
  --links_file=my_links.txt \
  --corum_file=my_complexes.txt \
  --mgi_file=my_go.gaf
```

This will process custom data files from a different directory.