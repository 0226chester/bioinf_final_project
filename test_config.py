#!/usr/bin/env python3
"""
Test script to validate your updated config.yaml
Run this after updating your config to ensure it loads correctly.
"""

import sys
import os
import yaml

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_yaml_syntax(config_path='config.yaml'):
    """Test if YAML syntax is valid."""
    print("1. Testing YAML syntax...")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("   ✅ YAML syntax is valid")
        return config
    except yaml.YAMLError as e:
        print(f"   ❌ YAML syntax error: {e}")
        return None
    except FileNotFoundError:
        print(f"   ❌ Config file not found: {config_path}")
        return None

def test_config_loading(config_path='config.yaml'):
    """Test if config loads with your utils function."""
    print("2. Testing config loading with utils.load_config()...")
    try:
        from utils import load_config
        config = load_config(config_path)
        print("   ✅ Config loads successfully with load_config()")
        return config
    except ImportError:
        print("   ⚠️  utils.load_config() not available, using yaml.safe_load()")
        return test_yaml_syntax(config_path)
    except Exception as e:
        print(f"   ❌ Error loading config: {e}")
        return None

def validate_required_fields(config):
    """Check if all required fields are present."""
    print("3. Validating required configuration fields...")
    
    required_structure = {
        'data': ['raw_dir', 'processed_dir', 'confidence_threshold'],
        'features': ['use_go_propagation', 'use_sequence_features', 'use_graph_features'],
        'model': ['type', 'hidden_dim', 'embed_dim', 'dropout', 'predictor_type'],
        'training': ['epochs', 'lr', 'patience', 'val_ratio', 'test_ratio', 
                    'data_split_seed', 'model_seed'],
        'experiment': ['name', 'dir'],
        'evaluation': ['run_advanced_analysis', 'novel_pred_threshold']
    }
    
    missing_fields = []
    
    for section, fields in required_structure.items():
        if section not in config:
            missing_fields.append(f"Missing section: {section}")
            continue
            
        for field in fields:
            if field not in config[section]:
                missing_fields.append(f"Missing {section}.{field}")
    
    if missing_fields:
        print("   ❌ Missing required fields:")
        for field in missing_fields:
            print(f"      - {field}")
        return False
    else:
        print("   ✅ All required fields present")
        return True

def validate_seed_configuration(config):
    """Validate seed configuration specifically."""
    print("4. Validating seed configuration...")
    
    if 'training' not in config:
        print("   ❌ No training section found")
        return False
    
    training = config['training']
    seed_fields = ['data_split_seed', 'model_seed']
    
    missing_seeds = [field for field in seed_fields if field not in training]
    
    if missing_seeds:
        print(f"   ❌ Missing seed fields: {missing_seeds}")
        print("   Add these to your training section:")
        for field in missing_seeds:
            print(f"      {field}: 42")
        return False
    
    print("   ✅ Seed configuration looks good")
    print(f"      data_split_seed: {training.get('data_split_seed')}")
    print(f"      model_seed: {training.get('model_seed')}")
    
    if 'feature_seed' in training:
        print(f"      feature_seed: {training.get('feature_seed')}")
    
    return True

def validate_experiment_directory(config):
    """Check experiment directory configuration."""
    print("5. Validating experiment directory...")
    
    if 'experiment' not in config or 'dir' not in config['experiment']:
        print("   ❌ experiment.dir not configured")
        return False
    
    exp_dir = config['experiment']['dir']
    print(f"   Experiment directory: {exp_dir}")
    
    # Try to create the directory structure
    try:
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
        os.makedirs(os.path.join(exp_dir, 'plots'), exist_ok=True)
        print("   ✅ Experiment directories created successfully")
        return True
    except Exception as e:
        print(f"   ❌ Cannot create experiment directories: {e}")
        return False

def display_config_summary(config):
    """Display a summary of the configuration."""
    print("\n📋 CONFIGURATION SUMMARY")
    print("=" * 40)
    
    if 'experiment' in config:
        print(f"Experiment: {config['experiment'].get('name', 'unnamed')}")
        print(f"Directory:  {config['experiment'].get('dir', 'not set')}")
    
    if 'data' in config:
        print(f"Data dir:   {config['data'].get('processed_dir', 'not set')}")
    
    if 'training' in config:
        training = config['training']
        print(f"Seeds:      split={training.get('data_split_seed', '?')}, "
              f"model={training.get('model_seed', '?')}")
        print(f"Training:   epochs={training.get('epochs', '?')}, "
              f"lr={training.get('lr', '?')}")
        print(f"Split:      val={training.get('val_ratio', '?')}, "
              f"test={training.get('test_ratio', '?')}")
    
    if 'model' in config:
        model = config['model']
        print(f"Model:      {model.get('type', '?')} "
              f"({model.get('hidden_dim', '?')}→{model.get('embed_dim', '?')})")
        print(f"Predictor:  {model.get('predictor_type', '?')}")

def main():
    """Run all configuration tests."""
    print("⚙️  Configuration Validation Test")
    print("=" * 50)
    
    config_path = 'config.yaml'
    
    # Basic loading test
    config = test_config_loading(config_path)
    if config is None:
        print("\n❌ Cannot load config. Fix the errors above first.")
        return False
    
    # Structure validation
    fields_ok = validate_required_fields(config)
    seeds_ok = validate_seed_configuration(config)
    exp_dir_ok = validate_experiment_directory(config)
    
    # Summary
    display_config_summary(config)
    
    print("\n" + "=" * 50)
    print("🎯 VALIDATION RESULTS")
    print("=" * 50)
    
    all_good = fields_ok and seeds_ok and exp_dir_ok
    
    if all_good:
        print("✅ Configuration is valid and ready to use!")
        print("\nNext steps:")
        print("1. Run: python simple_split_test.py")
        print("2. Run: python verify_splits.py (after preprocessing)")
        print("3. Run: python main.py --mode full_pipeline")
    else:
        print("❌ Configuration has issues. Please fix the errors above.")
        print("\nCommon fixes:")
        print("- Add missing fields to config.yaml")
        print("- Check YAML indentation (use spaces, not tabs)")
        print("- Ensure all string values are quoted if they contain special characters")
    
    return all_good

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)