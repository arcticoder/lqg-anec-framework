#!/usr/bin/env python3
"""
Debug script for metamaterial source issues.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.energy_source_interface import MetamaterialCasimirSource
from src.warp_bubble_solver import WarpBubbleSolver

def debug_metamaterial():
    """Debug metamaterial source behavior."""
    
    print("Creating metamaterial source...")
    source = MetamaterialCasimirSource(
        epsilon=-2.0,
        mu=-1.5,
        cell_size=50e-9,
        n_layers=100,
        R0=5.0,
        shell_thickness=0.5
    )
    
    print(f"Source parameters: {source.parameters}")
    print(f"Validation: {source.validate_parameters()}")
    
    # Create test coordinates
    print("\nCreating test mesh...")
    solver = WarpBubbleSolver()
    coords, _ = solver.generate_mesh(radius=10.0, resolution=20)
    print(f"Generated {len(coords)} mesh points")
    
    # Compute energy profile
    print("\nComputing energy profile...")
    energy_profile = source.energy_density(coords[:, 0], coords[:, 1], coords[:, 2])
    print(f"Energy profile shape: {energy_profile.shape}")
    print(f"Energy range: [{np.min(energy_profile):.2e}, {np.max(energy_profile):.2e}]")
    print(f"Negative values: {np.sum(energy_profile < 0)} / {len(energy_profile)}")
    print(f"Zero values: {np.sum(energy_profile == 0)} / {len(energy_profile)}")
    
    # Check distances to shell
    r = np.sqrt(np.sum(coords**2, axis=1))
    r_shell = np.abs(r - source.R0)
    in_shell = r_shell <= source.shell_thickness
    print(f"\nShell analysis:")
    print(f"Points in shell: {np.sum(in_shell)} / {len(r)}")
    print(f"Shell radius: {source.R0}")
    print(f"Shell thickness: {source.shell_thickness}")
    print(f"Shell range: [{source.R0 - source.shell_thickness}, {source.R0 + source.shell_thickness}]")
    
    # Show some sample points and their energies
    print(f"\nSample points and energies:")
    for i in range(min(10, len(coords))):
        print(f"  r={r[i]:.2f}, in_shell={in_shell[i]}, energy={energy_profile[i]:.2e}")
    
    # Test stability analysis
    print(f"\nTesting stability analysis...")
    stability = solver.stability_analysis(energy_profile, coords)
    print(f"Stability: {stability:.6f}")
    
    # Manual stability calculation for debugging
    print(f"\nManual stability debug:")
    valid_mask = np.isfinite(energy_profile)
    valid_energy = energy_profile[valid_mask]
    negative_fraction = np.sum(valid_energy < 0) / len(valid_energy)
    print(f"Negative fraction: {negative_fraction:.6f}")
    
    if negative_fraction < 0.1:
        print("Detected as shell-like profile")
        if np.any(valid_energy < 0):
            negative_energy = valid_energy[valid_energy < 0]
            energy_std = np.std(negative_energy)
            energy_mean = np.abs(np.mean(negative_energy))
            print(f"Negative energy mean: {energy_mean:.2e}")
            print(f"Negative energy std: {energy_std:.2e}")
            if energy_mean > 0:
                manual_stability = 1.0 / (1.0 + energy_std / energy_mean)
                print(f"Manual stability calculation: {manual_stability:.6f}")
    
if __name__ == "__main__":
    debug_metamaterial()
