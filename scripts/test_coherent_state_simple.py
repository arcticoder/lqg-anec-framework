#!/usr/bin/env python3
"""
Simple version that matches the user's requested format exactly.
"""
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from spin_network_utils import build_flat_graph
from coherent_states import CoherentState
from stress_tensor_operator import LocalT00

def main():
    # 1. Build a 5×5×5 cubic lattice
    graph = build_flat_graph(n_nodes=125, connectivity="cubic")
    # 2. Create and peak a coherent state (spread α=0.1)
    coh = CoherentState(graph, alpha=0.1)
    graph = coh.peak_on_flat()
    # 3. Compute T₀₀ on every node
    op = LocalT00()
    T00 = op.apply(graph)
    # 4. Summarize
    vals = np.array(list(T00.values()))
    print(f"T00 stats → min: {vals.min():.3e}, max: {vals.max():.3e}, mean: {vals.mean():.3e}")

if __name__ == "__main__":
    main()
