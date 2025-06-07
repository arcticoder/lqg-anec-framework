#!/usr/bin/env python3
"""
Test Coherent State to <T00> Pipeline

This script validates the coherent-state -> stress-energy pipeline by:
1. Building a flat spin network graph
2. Creating a coherent state peaked on flat geometry
3. Computing T00 expectation values at each node
4. Analyzing the energy density distribution

Author: LQG-ANEC Framework Development Team
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spin_network_utils import build_flat_graph
from coherent_states import CoherentState
from stress_tensor_operator import LocalT00

def main():
    print("=== Coherent State -> <T00> Pipeline Test ===\n")
    
    # 1. Build a 5×5×5 cubic lattice (125 nodes)
    print("1. Building flat spin network graph...")
    graph = build_flat_graph(n_nodes=125, connectivity="cubic")
    print(f"   Created graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges")
    
    # 2. Create and peak a coherent state (spread α=0.1)
    print("\n2. Creating coherent state...")
    coh = CoherentState(graph, alpha=0.1)
    print(f"   Coherent state with spread parameter α={coh.alpha}")
    print(f"   Coherence length: {coh.coherence_length():.3f}")
    
    # Peak the state on flat geometry
    print("   Peaking state on flat geometry...")
    peaked_graph = coh.peak_on_flat()
    
    # Assign coherent state amplitudes to the graph
    weave_amplitudes = coh.weave_state()
    peaked_graph.assign_amplitudes(weave_amplitudes)
    print(f"   Assigned amplitudes to {len(weave_amplitudes)} nodes")
      # 3. Compute T00 on every node
    print("\n3. Computing stress-energy tensor T00...")
    op = LocalT00()
    T00_values = op.apply(peaked_graph)
      # 4. Analyze and summarize results
    print("\n4. Analysis Results:")
    vals = np.array(list(T00_values.values()))
    
    print(f"   Total nodes analyzed: {len(vals)}")
    print(f"   T00 statistics:")
    print(f"     • Minimum:  {vals.min():.6e}")
    print(f"     • Maximum:  {vals.max():.6e}")
    print(f"     • Mean:     {vals.mean():.6e}")
    print(f"     • Std Dev:  {vals.std():.6e}")
    print(f"     • Median:   {np.median(vals):.6e}")
    
    # Check for energy density features
    negative_nodes = np.sum(vals < 0)
    positive_nodes = np.sum(vals > 0)
    zero_nodes = np.sum(np.abs(vals) < 1e-12)
    
    print(f"\n   Energy density distribution:")
    print(f"     • Negative energy nodes: {negative_nodes} ({100*negative_nodes/len(vals):.1f}%)")
    print(f"     • Positive energy nodes: {positive_nodes} ({100*positive_nodes/len(vals):.1f}%)")
    print(f"     • Near-zero nodes: {zero_nodes} ({100*zero_nodes/len(vals):.1f}%)")
    
    # Check for non-trivial structure
    if vals.std() > 1e-10:
        print(f"   ✓ Non-trivial energy density variation detected")
    else:
        print(f"   ⚠ Energy density appears uniform (may indicate issue)")
        
    # Identify extreme values
    if negative_nodes > 0:
        min_idx = np.argmin(vals)
        min_node = list(T00_values.keys())[min_idx]
        print(f"   • Most negative energy at node {min_node}: {vals[min_idx]:.6e}")
        
    if positive_nodes > 0:
        max_idx = np.argmax(vals)
        max_node = list(T00_values.keys())[max_idx]
        print(f"   • Most positive energy at node {max_node}: {vals[max_idx]:.6e}")
    
    # Test coherent state overlap and classical limit
    print("\n5. Additional coherent state tests:")
    
    # Test overlap with itself (should be 1)
    self_overlap = coh.overlap(coh)
    print(f"   • Self-overlap: {abs(self_overlap):.6f} (should be ≈ 1.0)")
    
    # Test classical limit
    classical_state = coh.classical_limit(scaling_factor=1e-3)
    print(f"   • Classical limit coherence length: {classical_state.coherence_length():.6f}")
    
    print(f"\n=== Pipeline Test Complete ===")
    
    return {
        'graph_nodes': len(graph.nodes),
        'graph_edges': len(graph.edges),
        'T00_min': vals.min(),
        'T00_max': vals.max(),
        'T00_mean': vals.mean(),
        'T00_std': vals.std(),
        'negative_fraction': negative_nodes / len(vals),
        'coherence_length': coh.coherence_length(),
        'self_overlap': abs(self_overlap)
    }

if __name__ == "__main__":
    try:
        results = main()
        print(f"\nTest completed successfully!")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
