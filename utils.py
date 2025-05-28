"""
Utility functions for PPI link prediction project.

This module contains helper functions for configuration loading,
logging setup, and other common utilities.
"""

import os
import yaml
import json
import random
import numpy as np
import torch
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import hashlib


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    # Validate configuration
    _validate_config(config)
    
    # Set up experiment directory
    config = _setup_experiment_dir(config)
    
    return config


def _validate_config(config: Dict[str, Any]):
    """Validate configuration parameters."""
    required_sections = ['data', 'features', 'model', 'training']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate specific parameters
    if config['data']['confidence_threshold'] < 0 or config['data']['confidence_threshold'] > 1000:
        raise ValueError("Confidence threshold must be between 0 and 1000")
    
    if config['model']['type'] not in ['GraphSAGE', 'GAT']:
        raise ValueError(f"Unknown model type: {config['model']['type']}")
    
    if config['training']['val_ratio'] + config['training']['test_ratio'] >= 1.0:
        raise ValueError("Sum of validation and test ratios must be less than 1.0")


# In utils.py
def _setup_experiment_dir(config: Dict[str, Any]) -> Dict[str, Any]:
    """Set up experiment directory based on configuration."""
    
    # 如果 config 中已經有一個完整的 'experiment.dir' 並且它存在，
    # (這通常發生在從實驗目錄下的 config.yaml 載入時)
    # 則直接使用它，並確保子目錄存在，但不創建新的時間戳目錄。
    if 'dir' in config.get('experiment', {}) and os.path.exists(config['experiment']['dir']):
        exp_dir = config['experiment']['dir']
        print(f"Using existing experiment directory: {exp_dir}")
        # 確保必要的子目錄存在
        os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
        # 不再重新儲存 config.yaml，因為我們就是從這裡載入的
        return config

    # 原始邏輯：創建新的實驗目錄
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.get('experiment', {}).get('name', 'default_experiment') # 確保有預設值
    # 確保 base_experiments_dir 存在，例如 "experiments"
    base_experiments_dir = "experiments"
    os.makedirs(base_experiments_dir, exist_ok=True)
    exp_dir = os.path.join(base_experiments_dir, f"{exp_name}_{timestamp}")
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
    
    # Add experiment directory to config
    if 'experiment' not in config:
        config['experiment'] = {}
    config['experiment']['dir'] = exp_dir
    
    # Save configuration to new experiment directory
    config_save_path = os.path.join(exp_dir, "config.yaml")
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"New experiment directory created: {exp_dir}. Configuration saved.")
    
    return config


def setup_logging(config: Dict[str, Any], name: str = "ppi_link_prediction") -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        config: Configuration dictionary
        name: Logger name
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(config['experiment']['dir'], 'logs', f'{name}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    return logger


def set_seed(seed: int):
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(cuda_preference: bool = True) -> torch.device:
    """
    Get the device to use for computation.
    
    Args:
        cuda_preference: Whether to prefer CUDA if available
        
    Returns:
        torch.device object
    """
    if cuda_preference and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        logging.info("Using CPU")
    
    return device


def save_results(results: Dict[str, Any], config: Dict[str, Any], filename: str = "results.json"):
    """
    Save experiment results to JSON file.
    
    Args:
        results: Results dictionary
        config: Configuration dictionary
        filename: Output filename
    """
    output_path = os.path.join(config['experiment']['dir'], filename)
    
    # Add configuration hash for tracking
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    
    results['config_hash'] = config_hash
    results['timestamp'] = datetime.now().isoformat()
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logging.info(f"Results saved to {output_path}")


def create_edge_index_from_adjacency(adj_matrix: np.ndarray) -> torch.Tensor:
    """
    Create edge index from adjacency matrix.
    
    Args:
        adj_matrix: Adjacency matrix
        
    Returns:
        Edge index tensor of shape [2, num_edges]
    """
    rows, cols = np.where(adj_matrix > 0)
    edge_index = torch.tensor(np.stack([rows, cols], axis=0), dtype=torch.long)
    return edge_index


def calculate_class_weights(labels: torch.Tensor) -> torch.Tensor:
    """
    Calculate class weights for imbalanced data.
    
    Args:
        labels: Binary labels
        
    Returns:
        Class weights tensor
    """
    num_pos = labels.sum().item()
    num_neg = len(labels) - num_pos
    
    # Calculate weights inversely proportional to class frequencies
    pos_weight = len(labels) / (2 * num_pos) if num_pos > 0 else 1.0
    neg_weight = len(labels) / (2 * num_neg) if num_neg > 0 else 1.0
    
    weights = torch.ones_like(labels, dtype=torch.float)
    weights[labels == 1] = pos_weight
    weights[labels == 0] = neg_weight
    
    return weights