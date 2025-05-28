"""
Feature engineering module for PPI link prediction.

This module implements various feature extraction methods including:
- GO term features with propagation and information content
- Protein sequence features
- Graph topological features
- Feature combination and scaling
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict, Counter
import logging
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
from sklearn.decomposition import TruncatedSVD
import math
import pickle
import os

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Feature engineering for PPI link prediction.
    
    This class handles all feature extraction and engineering steps,
    including GO term processing, sequence features, and graph features.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize feature engineer with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.features_config = config['features']
        self.processed_dir = config['data']['processed_dir']
        
        # Feature components
        self.go_mlb = None  # MultiLabelBinarizer for GO terms
        self.go_svd = None  # SVD for GO term reduction
        self.scaler = None  # Feature scaler
        self.feature_names = []  # Track feature names
        
        # GO DAG for propagation (will be loaded if needed)
        self.go_dag = None
        
    def extract_features(self, proteins_df: pd.DataFrame, 
                        mappings: Dict) -> Tuple[np.ndarray, List[str]]:
        """
        Extract all features for proteins.
        
        Args:
            proteins_df: DataFrame with protein information
            mappings: Dictionary with protein mappings (GO, sequences, etc.)
            
        Returns:
            Tuple of (feature_matrix, feature_names)
        """
        logger.info("Starting feature extraction...")
        
        features_list = []
        names_list = []
        
        # 1. GO term features
        if self.features_config['use_go_propagation'] or self.features_config['use_ic_features']:
            go_features, go_names = self._extract_go_features(proteins_df, mappings)
            features_list.append(go_features)
            names_list.extend(go_names)
            
        # 2. Sequence features
        if self.features_config['use_sequence_features']:
            seq_features, seq_names = self._extract_sequence_features(proteins_df)
            features_list.append(seq_features)
            names_list.extend(seq_names)
            
        # 3. Graph features
        if self.features_config['use_graph_features']:
            graph_features, graph_names = self._extract_graph_features(proteins_df)
            features_list.append(graph_features)
            names_list.extend(graph_names)
            
        # 4. Binary indicators
        binary_features, binary_names = self._extract_binary_features(proteins_df)
        features_list.append(binary_features)
        names_list.extend(binary_names)
        
        # Combine all features
        feature_matrix = np.hstack(features_list)
        self.feature_names = names_list
        
        logger.info(f"Extracted {feature_matrix.shape[1]} features for "
                   f"{feature_matrix.shape[0]} proteins")
        
        # Scale features if configured
        if self.features_config['scale_features']:
            feature_matrix = self._scale_features(feature_matrix, names_list)
            
        return feature_matrix, names_list
        
    def _extract_go_features(self, proteins_df: pd.DataFrame, 
                           mappings: Dict) -> Tuple[np.ndarray, List[str]]:
        """Extract GO term based features."""
        logger.info("Extracting GO term features...")
        
        # Parse GO terms for each protein
        protein_go_sets = []
        protein_to_go = mappings.get('protein_to_go', {})
        
        for _, row in proteins_df.iterrows():
            go_terms = set()
            
            # Parse from GO_term column
            if pd.notna(row['GO_term']) and row['GO_term']:
                go_terms.update(row['GO_term'].split(';'))
                
            # Propagate to parent terms if configured
            if self.features_config['use_go_propagation'] and go_terms:
                go_terms = self._propagate_go_terms(go_terms)
                
            protein_go_sets.append(go_terms)
            
        # Create multi-hot encoding
        self.go_mlb = MultiLabelBinarizer(sparse_output=True)
        go_multi_hot = self.go_mlb.fit_transform(protein_go_sets)
        
        logger.info(f"Found {len(self.go_mlb.classes_)} unique GO terms")
        
        # Apply dimensionality reduction if needed
        if (self.features_config['go_svd_components'] > 0 and 
            len(self.go_mlb.classes_) > self.features_config['go_svd_components']):
            
            logger.info(f"Applying SVD to reduce GO features to "
                       f"{self.features_config['go_svd_components']} components")
            
            self.go_svd = TruncatedSVD(
                n_components=self.features_config['go_svd_components'],
                random_state=42
            )
            go_features = self.go_svd.fit_transform(go_multi_hot)
            feature_names = [f'GO_SVD_{i}' for i in range(go_features.shape[1])]
            
        else:
            go_features = go_multi_hot.toarray()
            feature_names = [f'GO_{term}' for term in self.go_mlb.classes_]
            
        # Add Information Content features if configured
        if self.features_config['use_ic_features']:
            ic_features, ic_names = self._calculate_ic_features(
                protein_go_sets, protein_to_go
            )
            go_features = np.hstack([go_features, ic_features])
            feature_names.extend(ic_names)
            
        return go_features, feature_names
        
    def _propagate_go_terms(self, go_terms: Set[str]) -> Set[str]:
        """
        Propagate GO terms to parent terms.
        
        Args:
            go_terms: Set of GO term IDs
            
        Returns:
            Expanded set including parent terms
        """
        # Load GO DAG if not already loaded
        if self.go_dag is None:
            self._load_go_dag()
            
        if self.go_dag is None:
            # If GO DAG still not available, return original terms
            return go_terms
            
        propagated = set(go_terms)
        
        # Add parent terms for each GO term
        for term in go_terms:
            if term in self.go_dag:
                # Get all ancestors
                ancestors = self._get_go_ancestors(term)
                propagated.update(ancestors)
                
        return propagated
        
    def _load_go_dag(self):
        """Load Gene Ontology DAG for term propagation."""
        try:
            # Try to import goatools
            from goatools.obo_parser import GODag
            
            go_obo_path = os.path.join(self.processed_dir, 'go-basic.obo')
            if os.path.exists(go_obo_path):
                logger.info(f"Loading GO DAG from {go_obo_path}")
                self.go_dag = GODag(go_obo_path, optional_attrs=['relationship'])
            else:
                logger.warning(f"GO OBO file not found at {go_obo_path}")
                self.go_dag = None
                
        except ImportError:
            logger.warning("goatools not installed. GO propagation disabled.")
            self.go_dag = None
            
    def _get_go_ancestors(self, go_term: str) -> Set[str]:
        """Get all ancestor terms for a GO term."""
        ancestors = set()
        
        if go_term not in self.go_dag:
            return ancestors
            
        # Get term object
        term_obj = self.go_dag[go_term]
        
        # Recursively get parents
        to_visit = [term_obj]
        visited = set()
        
        while to_visit:
            current = to_visit.pop()
            if current.id in visited:
                continue
                
            visited.add(current.id)
            ancestors.add(current.id)
            
            # Add parents to visit
            for parent in current.parents:
                if parent.id not in visited:
                    to_visit.append(parent)
                    
        return ancestors
        
    def _calculate_ic_features(self, protein_go_sets: List[Set[str]], 
                              protein_to_go: Dict) -> Tuple[np.ndarray, List[str]]:
        """
        Calculate Information Content features for GO terms.
        
        Args:
            protein_go_sets: List of GO term sets for each protein
            protein_to_go: Complete protein to GO mapping
            
        Returns:
            IC features and feature names
        """
        # Calculate term frequencies
        term_counts = Counter()
        total_proteins = len([s for s in protein_go_sets if s])
        
        for go_set in protein_go_sets:
            for term in go_set:
                term_counts[term] += 1
                
        # Calculate Information Content for each term
        term_ic = {}
        for term, count in term_counts.items():
            if total_proteins > 0 and count > 0:
                probability = count / total_proteins
                term_ic[term] = -math.log2(probability)
            else:
                term_ic[term] = 0
                
        # Calculate IC features for each protein
        ic_features = []
        
        for go_set in protein_go_sets:
            if go_set:
                ic_values = [term_ic.get(term, 0) for term in go_set]
                ic_sum = sum(ic_values)
                ic_mean = np.mean(ic_values) if ic_values else 0
            else:
                ic_sum = ic_mean = 0
                
            ic_features.append([ic_sum, ic_mean])
            
        feature_names = ['GO_IC_Sum', 'GO_IC_Mean']
        
        return np.array(ic_features), feature_names
        
    def _extract_sequence_features(self, proteins_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract protein sequence based features."""
        logger.info("Extracting sequence features...")
        
        features = []
        
        # Amino acid composition
        amino_acids = self.features_config['amino_acids']
        
        for _, row in proteins_df.iterrows():
            sequence = row['sequence'] if pd.notna(row['sequence']) else ''
            
            # Calculate features
            seq_length = len(sequence)
            
            # Amino acid composition
            aa_counts = Counter(sequence.upper())
            aa_comp = [aa_counts.get(aa, 0) / seq_length if seq_length > 0 else 0 
                      for aa in amino_acids]
            
            # Combine features
            features.append([np.log1p(seq_length)] + aa_comp)
            
        # Feature names
        feature_names = ['sequence_length_log'] + [f'AA_{aa}' for aa in amino_acids] 
        
        return np.array(features), feature_names
        
    def _extract_graph_features(self, proteins_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract graph topological features."""
        logger.info("Extracting graph features...")
        
        # Get specified features
        feature_cols = self.features_config['graph_features']
        
        # Extract and transform features
        features = []
        for col in feature_cols:
            if col in proteins_df.columns:
                # Apply log transformation to make features more normal
                values = proteins_df[col].values
                if col in ['degree', 'betweenness', 'closeness']:
                    values = np.log1p(values)
                features.append(values)
            else:
                logger.warning(f"Graph feature {col} not found in dataframe")
                
        if features:
            features = np.column_stack(features)
            feature_names = [f'{col}_log' if col in ['degree', 'betweenness', 'closeness'] 
                           else col for col in feature_cols if col in proteins_df.columns]
        else:
            features = np.zeros((len(proteins_df), 0))
            feature_names = []
            
        return features, feature_names
        
    def _extract_binary_features(self, proteins_df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """Extract binary indicator features."""
        features = []
        
        # Has GO term
        has_go = proteins_df['has_GO_term'].values.reshape(-1, 1)
        features.append(has_go)
        
        # Has sequence
        has_seq = proteins_df['has_sequence'].values.reshape(-1, 1)
        features.append(has_seq)
        
        features = np.hstack(features)
        feature_names = ['has_GO_term', 'has_sequence']
        
        return features, feature_names
        
    def _scale_features(self, features: np.ndarray, 
                       feature_names: List[str]) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names
            
        Returns:
            Scaled feature matrix
        """
        logger.info("Scaling features...")
        
        # Identify features to scale (exclude binary features)
        binary_features = {'has_GO_term', 'has_sequence'}
        scale_mask = np.array([name not in binary_features for name in feature_names])
        
        if np.any(scale_mask):
            self.scaler = StandardScaler()
            features_to_scale = features[:, scale_mask]
            
            # Only scale features with variance
            variances = np.var(features_to_scale, axis=0)
            valid_features = variances > 1e-10
            
            if np.any(valid_features):
                features_to_scale[:, valid_features] = self.scaler.fit_transform(
                    features_to_scale[:, valid_features]
                )
                features[:, scale_mask] = features_to_scale
                
                logger.info(f"Scaled {np.sum(valid_features)} features")
            else:
                logger.warning("No features with sufficient variance for scaling")
                
        return features
        
    def save_feature_info(self, save_dir: str):
        """
        Save feature engineering components for later use.
        
        Args:
            save_dir: Directory to save feature info
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save feature names
        with open(os.path.join(save_dir, 'feature_names.pkl'), 'wb') as f:
            pickle.dump(self.feature_names, f)
            
        # Save fitted components
        components = {
            'go_mlb': self.go_mlb,
            'go_svd': self.go_svd,
            'scaler': self.scaler
        }
        
        with open(os.path.join(save_dir, 'feature_components.pkl'), 'wb') as f:
            pickle.dump(components, f)
            
        logger.info(f"Saved feature info to {save_dir}")
        
    def load_feature_info(self, save_dir: str):
        """
        Load saved feature engineering components.
        
        Args:
            save_dir: Directory containing saved feature info
        """
        # Load feature names
        with open(os.path.join(save_dir, 'feature_names.pkl'), 'rb') as f:
            self.feature_names = pickle.load(f)
            
        # Load fitted components
        with open(os.path.join(save_dir, 'feature_components.pkl'), 'rb') as f:
            components = pickle.load(f)
            
        self.go_mlb = components['go_mlb']
        self.go_svd = components['go_svd']
        self.scaler = components['scaler']
        
        logger.info(f"Loaded feature info from {save_dir}")


def engineer_features(proteins_df: pd.DataFrame, mappings: Dict, 
                     config: Dict) -> Tuple[np.ndarray, List[str]]:
    """
    Main function to engineer features for proteins.
    
    Args:
        proteins_df: DataFrame with protein information
        mappings: Dictionary with protein mappings
        config: Configuration dictionary
        
    Returns:
        Tuple of (feature_matrix, feature_names)
    """
    engineer = FeatureEngineer(config)
    features, names = engineer.extract_features(proteins_df, mappings)
    
    # Save feature info for reproducibility
    save_dir = os.path.join(config['data']['processed_dir'], 'feature_info')
    engineer.save_feature_info(save_dir)
    
    return features, names