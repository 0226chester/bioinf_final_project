#!/usr/bin/env python3
"""
Simple test to verify RandomLinkSplit behavior without requiring processed data.
Run this to understand the splitting issue before running the full verification.
"""

import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.transforms import RandomLinkSplit

def create_dummy_data():
    """Create a simple test graph."""
    # Create a small graph with 20 nodes and some edges
    num_nodes = 20
    
    # Create some edges (undirected)
    edge_list = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Square
        (4, 5), (5, 6), (6, 7), (7, 4),  # Another square
        (0, 4), (1, 5), (2, 6), (3, 7),  # Connect the squares
        (8, 9), (9, 10), (10, 11), (11, 8),  # Third component
        (12, 13), (13, 14),  # Small component
        (15, 16), (16, 17), (17, 18), (18, 19)  # Chain
    ]
    
    # Convert to tensor (both directions for undirected)
    edges = []
    for a, b in edge_list:
        edges.extend([(a, b), (b, a)])
    
    edge_index = torch.tensor(edges).T
    x = torch.randn(num_nodes, 10)  # Random node features
    
    data = Data(x=x, edge_index=edge_index)
    print(f"Created test graph: {data.num_nodes} nodes, {data.num_edges} edges")
    return data

def test_basic_splitting():
    """Test basic RandomLinkSplit behavior."""
    print("\n=== BASIC SPLITTING TEST ===")
    
    data = create_dummy_data()
    
    # Test 1: Same seed, immediate re-call
    print("\n1. Testing same seed, immediate re-call:")
    
    torch.manual_seed(42)
    transform1 = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True)
    train1, val1, test1 = transform1(data)
    
    torch.manual_seed(42)  # Reset to same seed
    transform2 = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True)
    train2, val2, test2 = transform2(data)
    
    # Compare test edges
    test_edges_1 = set(tuple(edge) for edge in test1.edge_label_index.T.numpy())
    test_edges_2 = set(tuple(edge) for edge in test2.edge_label_index.T.numpy())
    
    if test_edges_1 == test_edges_2:
        print("   ✅ Same seed produces identical splits")
    else:
        print("   ❌ Same seed produces different splits!")
    
    # Test 2: Different seeds
    print("\n2. Testing different seeds:")
    
    torch.manual_seed(42)
    _, _, test_seed42 = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True)(data)
    
    torch.manual_seed(123)
    _, _, test_seed123 = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True)(data)
    
    test_edges_42 = set(tuple(edge) for edge in test_seed42.edge_label_index.T.numpy())
    test_edges_123 = set(tuple(edge) for edge in test_seed123.edge_label_index.T.numpy())
    
    if test_edges_42 != test_edges_123:
        print("   ✅ Different seeds produce different splits")
    else:
        print("   ❌ Different seeds produce same splits (unexpected)")
    
    # Test 3: Contaminated random state (the real problem)
    print("\n3. Testing contaminated random state (your issue):")
    
    torch.manual_seed(42)
    _, _, test_clean = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True)(data)
    
    torch.manual_seed(42)
    # Simulate operations that change random state (like your training)
    dummy_model = torch.nn.Linear(10, 5)
    for _ in range(10):
        _ = dummy_model(torch.randn(5, 10))
        _ = torch.nn.functional.dropout(torch.randn(20, 10), 0.5, training=True)
    
    # Now split again - random state has changed!
    _, _, test_contaminated = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True)(data)
    
    test_edges_clean = set(tuple(edge) for edge in test_clean.edge_label_index.T.numpy())
    test_edges_contaminated = set(tuple(edge) for edge in test_contaminated.edge_label_index.T.numpy())
    
    overlap = len(test_edges_clean & test_edges_contaminated)
    total = len(test_edges_clean)
    
    if test_edges_clean == test_edges_contaminated:
        print("   ✅ Random state contamination has no effect (unexpected)")
    else:
        print(f"   ❌ Random state contamination changes splits!")
        print(f"       Overlap: {overlap}/{total} ({overlap/total*100:.1f}%)")
        print("       This is exactly your problem!")

def test_fixes():
    """Test the proposed fixes."""
    print("\n=== TESTING FIXES ===")
    
    data = create_dummy_data()
    
    def get_consistent_split_v1(data, seed=42):
        """Quick fix: Save and restore random state."""
        current_state = torch.get_rng_state()
        torch.manual_seed(seed)
        
        transform = RandomLinkSplit(num_val=0.2, num_test=0.2, is_undirected=True)
        train, val, test = transform(data)
        
        torch.set_rng_state(current_state)
        return train, val, test
    
    print("\n1. Testing quick fix (save/restore random state):")
    
    # First call with random operations
    torch.manual_seed(42)
    for _ in range(20):
        _ = torch.randn(10, 10)  # Change random state
    _, _, test_fix1 = get_consistent_split_v1(data, seed=99)
    
    # Second call with different random operations
    torch.manual_seed(42)
    model = torch.nn.Linear(10, 5)
    for _ in range(50):
        _ = model(torch.randn(3, 10))  # Different operations
    _, _, test_fix2 = get_consistent_split_v1(data, seed=99)  # Same seed
    
    test_edges_fix1 = set(tuple(edge) for edge in test_fix1.edge_label_index.T.numpy())
    test_edges_fix2 = set(tuple(edge) for edge in test_fix2.edge_label_index.T.numpy())
    
    if test_edges_fix1 == test_edges_fix2:
        print("   ✅ Quick fix works! Splits are consistent despite random operations")
    else:
        print("   ❌ Quick fix failed")
    
    print(f"       Test set size: {len(test_edges_fix1)} edges")

def main():
    """Run all basic tests."""
    print("🧪 Basic RandomLinkSplit Behavior Test")
    print("=" * 50)
    print("This tests the core splitting issue without requiring your processed data.")
    
    test_basic_splitting()
    test_fixes()
    
    print("\n" + "=" * 50)
    print("📋 SUMMARY")
    print("=" * 50)
    print("If you see:")
    print("   ✅ Same seed produces identical splits")
    print("   ❌ Random state contamination changes splits!")
    print("   ✅ Quick fix works!")
    print("")
    print("Then this confirms:")
    print("   1. RandomLinkSplit is deterministic with same seed")
    print("   2. BUT random state changes between train/eval cause different splits")
    print("   3. The proposed fixes will work for your pipeline")
    print("")
    print("Next step: Run the full verification with your actual data:")
    print("   python verify_splits.py")

if __name__ == "__main__":
    main()