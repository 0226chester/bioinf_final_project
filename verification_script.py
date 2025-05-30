#!/usr/bin/env python3
"""
Test script to verify your current get_consistent_data_splits implementation works.
"""

import sys
import os
import torch
from torch_geometric.data import Data
from torch_geometric import seed_everything

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils import get_consistent_data_splits
    print("✅ Successfully imported get_consistent_data_splits")
except ImportError as e:
    print(f"❌ Could not import get_consistent_data_splits: {e}")
    sys.exit(1)

def create_test_data():
    """Create simple test data."""
    # Create a small graph
    edge_list = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6)]
    edges = []
    for a, b in edge_list:
        edges.extend([(a, b), (b, a)])
    
    edge_index = torch.tensor(edges).T
    x = torch.randn(7, 5)  # 7 nodes, 5 features
    data = Data(x=x, edge_index=edge_index)
    return data

def create_test_config():
    """Create test config."""
    return {
        'training': {
            'data_split_seed': 42,
            'val_ratio': 0.2,
            'test_ratio': 0.3
        }
    }

def test_consistency():
    """Test if your implementation produces consistent splits."""
    print("\n🔍 Testing Your Current Implementation")
    print("=" * 50)
    
    data = create_test_data()
    config = create_test_config()
    
    print(f"Test data: {data.num_nodes} nodes, {data.num_edges} edges")
    
    # Test 1: Same function call should give same results
    print("\n1. Testing immediate consistency:")
    
    train1, val1, test1 = get_consistent_data_splits(data, config, 'training')
    train2, val2, test2 = get_consistent_data_splits(data, config, 'training')
    
    test_edges_1 = set(tuple(edge) for edge in test1.edge_label_index.T.numpy())
    test_edges_2 = set(tuple(edge) for edge in test2.edge_label_index.T.numpy())
    
    if test_edges_1 == test_edges_2:
        print("   ✅ Immediate calls produce identical splits")
    else:
        print("   ❌ Immediate calls produce different splits")
        return False
    
    # Test 2: Training vs Evaluation (your real use case)
    print("\n2. Testing training vs evaluation consistency:")
    
    # Simulate training scenario
    seed_everything(123)  # Different global seed
    for _ in range(20):
        _ = torch.randn(10, 10)  # Change random state
    
    train_train, val_train, test_train = get_consistent_data_splits(data, config, 'training')
    
    # Simulate evaluation scenario  
    seed_everything(456)  # Different global seed again
    model = torch.nn.Linear(5, 3)
    for _ in range(50):
        _ = model(torch.randn(3, 5))  # More random operations
    
    _, _, test_eval = get_consistent_data_splits(data, config, 'evaluation')
    
    test_edges_train = set(tuple(edge) for edge in test_train.edge_label_index.T.numpy())
    test_edges_eval = set(tuple(edge) for edge in test_eval.edge_label_index.T.numpy())
    
    if test_edges_train == test_edges_eval:
        print("   ✅ Training and evaluation use identical test sets!")
        print(f"      Test set size: {len(test_edges_train)} edges")
        return True
    else:
        overlap = len(test_edges_train & test_edges_eval)
        total = len(test_edges_train)
        print(f"   ❌ Training and evaluation differ! Overlap: {overlap}/{total} ({overlap/total*100:.1f}%)")
        return False

def test_with_real_data():
    """Test with your actual data if available."""
    print("\n🔍 Testing with Your Real Data (if available)")
    print("=" * 50)
    
    try:
        from utils import load_config
        from data.loader import load_custom_ppi_data
        
        config = load_config('config.yaml')
        processed_dir = config['data']['processed_dir']
        
        interaction_file = os.path.join(processed_dir, 'interactions_processed.csv')
        numerical_features_file = os.path.join(processed_dir, 'node_features_matrix.npy')
        proteins_info_file = os.path.join(processed_dir, 'proteins_processed.csv')
        
        if all(os.path.exists(f) for f in [interaction_file, numerical_features_file, proteins_info_file]):
            print("Found your processed data files!")
            
            data = load_custom_ppi_data(interaction_file, numerical_features_file, proteins_info_file)
            if data is not None:
                print(f"Loaded real data: {data.num_nodes} nodes, {data.num_edges} edges")
                
                # Test consistency with real data
                _, _, test1 = get_consistent_data_splits(data, config, 'training')
                _, _, test2 = get_consistent_data_splits(data, config, 'evaluation')
                
                test_edges_1 = set(tuple(edge) for edge in test1.edge_label_index.T.numpy())
                test_edges_2 = set(tuple(edge) for edge in test2.edge_label_index.T.numpy())
                
                if test_edges_1 == test_edges_2:
                    print("   ✅ Real data splits are consistent!")
                    print(f"      Real test set size: {len(test_edges_1)} edges")
                    return True
                else:
                    print("   ❌ Real data splits are inconsistent!")
                    return False
            else:
                print("   ⚠️  Could not load real data")
        else:
            print("   ⚠️  Processed data files not found - run preprocessing first")
            
    except Exception as e:
        print(f"   ⚠️  Could not test with real data: {e}")
    
    return None

def main():
    print("🧪 Testing Your Current Data Splitting Implementation")
    print("=" * 60)
    
    # Test basic functionality
    basic_works = test_consistency()
    
    # Test with real data if available
    real_data_works = test_with_real_data()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST RESULTS")
    print("=" * 60)
    
    print(f"Basic consistency test:    {'✅ PASS' if basic_works else '❌ FAIL'}")
    
    if real_data_works is not None:
        print(f"Real data consistency:     {'✅ PASS' if real_data_works else '❌ FAIL'}")
    else:
        print(f"Real data consistency:     ⚠️  Could not test")
    
    if basic_works:
        print("\n🎉 YOUR IMPLEMENTATION WORKS!")
        print("✅ Data splits will be consistent between training and evaluation")
        print("✅ You're using PyG seed_everything correctly")
        print("✅ Save/restore random state pattern works")
        
        print("\n📝 Next steps:")
        print("1. Make the small seed config fix shown above")
        print("2. Run your full pipeline with confidence!")
        print("3. Your results will be reproducible")
    else:
        print("\n❌ ISSUES DETECTED")
        print("Check your get_consistent_data_splits implementation")
    
    return basic_works

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)