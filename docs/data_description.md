# Mouse PPI Network and Protein Complex Data Description

## Overview

This document details the protein-protein interaction (PPI) network and protein complex data processed for the GVAE Protein Complex Subnetwork Synthesis Project. It describes the original data sources, processing steps, and characteristics of the final dataset used for modeling.

## Raw Data Sources

The dataset integrates information from three major biological databases:

1. **STRING Database (v12.0, Mus musculus)**
   - Protein interaction data (links file)
   - Protein annotations and metadata (info file)
   - ID mapping between different database systems (aliases file)
   - Accessed through: https://string-db.org/
   - Files: `10090.protein.links.v12.0.txt`, `10090.protein.info.v12.0.txt`, `10090.protein.aliases.v12.0.txt`

2. **CORUM Database**
   - Experimentally verified protein complexes
   - Accessed through: https://mips.helmholtz-muenchen.de/corum/
   - File: `corum_allComplexes.txt`

3. **Gene Ontology Annotations (MGI)**
   - Functional annotations of mouse genes
   - Accessed through: http://geneontology.org/
   - File: `mgi.gaf`

## Data Processing Pipeline

The processing workflow consists of several key steps:

1. **Protein Identifier Mapping**
   - Mapped 21,788 STRING IDs to UniProt IDs
   - Mapped 20,348 STRING IDs to gene identifiers
   - Created cross-reference dictionaries for seamless ID conversion

2. **Protein Information Processing**
   - Extracted metadata for 21,840 proteins including names and descriptions

3. **PPI Network Construction**
   - Applied confidence score filtering (threshold ≥ 0.7)
   - Retained 201,860 high-confidence interactions from 12,684,354 total interactions
   - Filtered out 12,280,634 low-confidence interactions
   - Final network contains 15,971 proteins (nodes)

4. **Protein Complex Extraction**
   - Identified 1,182 mouse complexes in CORUM
   - Successfully mapped 332 complexes to the STRING network
   - Average complex size: 4.15 proteins

5. **GO Annotation Processing**
   - Processed 598,195 GO annotation entries
   - Extracted terms across three aspects:
     - Biological Process: 12,658 unique terms
     - Molecular Function: 4,683 unique terms
     - Cellular Component: 1,838 unique terms

## Feature Engineering

Features for each protein in the network were created by combining:

1. **Topological Features (3 dimensions)**
   - Degree centrality: Measures the number of connections
   - Clustering coefficient: Measures local clustering
   - Betweenness centrality: Measures the importance of a node as a bridge

2. **Functional Features (10 dimensions)**
   - Applied dimensionality reduction (PCA) to GO Cellular Component annotations
   - Reduced from original 1,838 dimensions to 10 dimensions
   - Captures 50% of the variance in the original data

3. **Combined Feature Vectors**
   - Total dimension: 13 features per protein
   - All features normalized for consistent scaling

## Network Characteristics

The processed protein interaction network has the following properties:

- **Nodes (Proteins)**: 15,971
- **Edges (Interactions)**: 201,860
- **Filtering**: Only high-confidence interactions (score ≥ 0.7)
- **Node Features**: 13-dimensional vectors (3 topological + 10 GO-derived)

## Protein Complex Dataset

The processed protein complex collection has these characteristics:

- **Total Complexes**: 332
- **Average Size**: 4.15 proteins per complex
- **Size Range**: Varies from small (3 proteins) to large complexes
- **Representation**: Each complex is represented as a connected subgraph of the main PPI network

## Processed Data Structure

The processed data is organized in the `data/processed` directory with the following files:

- **ppi_network.edgelist**: Tab-separated file with protein interactions and confidence scores
- **protein_info.tsv**: Protein annotations including names and sizes
- **complex_summary.tsv**: Summary information about each complex
- **complexes/**: Directory containing individual complex edgelists
- **node_features.npy**: NumPy array of node feature vectors (15,971 × 13)
- **node_ids.txt**: Mapping between array indices and protein IDs
- **feature_info.txt**: Information about feature dimensions
- **network_stats.txt**: Network-level statistics

## Usage Guidelines

### For Model Training

1. Use `node_features.npy` and `node_ids.txt` to access protein features
2. Use complex edgelists from the `complexes/` directory as training examples
3. Consider splitting complexes into training/validation/test sets

### For Evaluation

1. Compare generated complexes against known complexes
2. Evaluate both structural properties and biological coherence
3. Use the STRING network to assess the novelty of predicted interactions

### For Visualization

1. Use the protein names from `protein_info.tsv` for readable visualizations
2. The subgraphs in `complexes/` can be used to visualize known complexes

---

*Note: This dataset was processed on [current date] for the GVAE Protein Complex Subnetwork Synthesis Project.*