#!/usr/bin/env python3
"""
Focused parameter scan around optimal ghost EFT configuration.
"""

import sys
import os
sys.path.append('.')
sys.path.append('./src')

import numpy as np
import json
import time
from pathlib import Path

from src.ghost_condensate_eft import GhostCondensateEFT

class GaussianSmear:
    """Gaussian smearing function for ANEC calculations."""
    
    def __init__(self, tau0):
        self.tau0 = tau0
        
    def kernel(self, t):
        """Gaussian smearing kernel."""
        return np.exp(-t**2 / (2 * self.tau0**2)) / np.sqrt(2 * np.pi * self.tau0**2)

def focused_scan_robust(target_anec=-1e-12):
    """Robust focused scan around optimal parameters."""
    
    print(f"Starting focused scan targeting ANEC ≤ {target_anec:.1e} W")
    
    # Setup
    grid = np.linspace(-1e6, 1e6, 2000)  # High resolution
    smear = GaussianSmear(tau0=7*24*3600)
    
    # Focused ranges around known optimum (M=1e3, α=0.01, β=0.1)
    M_values = np.logspace(2.5, 3.5, 15)        # Around 10³ 
    alpha_values = np.linspace(0.005, 0.1, 15)   # Around 0.01
    beta_values = np.logspace(-2, 0, 15)         # Around 0.1
    
    candidates = []
    total = len(M_values) * len(alpha_values) * len(beta_values)
    
    print(f"Scanning {total} parameter combinations...")
    
    start_time = time.time()
    progress = 0
    
    for M in M_values:
        for alpha in alpha_values:
            for beta in beta_values:
                try:
                    eft = GhostCondensateEFT(M=M, alpha=alpha, beta=beta, grid=grid)
                    anec_value = eft.compute_anec(smear.kernel)
                    
                    if anec_value <= target_anec:
                        candidate = {
                            'M': float(M),
                            'alpha': float(alpha),
                            'beta': float(beta),
                            'anec_value': float(anec_value),
                            'enhancement_factor': float(-anec_value / target_anec)
                        }
                        candidates.append(candidate)
                        print(f"✓ Found candidate: ANEC={anec_value:.3e} W "
                              f"(M={M:.1e}, α={alpha:.3f}, β={beta:.3f})")
                        
                except Exception as e:
                    pass  # Skip failed configurations
                    
                progress += 1
                if progress % 100 == 0:
                    print(f"Progress: {progress}/{total} ({100*progress/total:.1f}%)")
    
    scan_time = time.time() - start_time
    
    if candidates:
        # Sort by strongest violation (most negative ANEC)
        candidates.sort(key=lambda x: x['anec_value'])
        
        print(f"\n=== FOCUSED SCAN RESULTS ===")
        print(f"Scan time: {scan_time:.2f} seconds")
        print(f"Found {len(candidates)} candidates exceeding target")
        print(f"Best ANEC violation: {candidates[0]['anec_value']:.3e} W")
        print(f"Best parameters: M={candidates[0]['M']:.1e}, α={candidates[0]['alpha']:.3f}, β={candidates[0]['beta']:.3f}")
        
        # Save results
        output_path = Path("results") / "ghost_eft_focused_scan_results.json"
        output_path.parent.mkdir(exist_ok=True)
        
        result_data = {
            'target_anec': target_anec,
            'scan_metadata': {
                'scan_time_seconds': scan_time,
                'total_combinations': total,
                'candidates_found': len(candidates)
            },
            'best_candidate': candidates[0],
            'top_10_candidates': candidates[:10],
            'all_candidates': candidates
        }
        
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=2)
            
        print(f"Results saved to {output_path}")
        
        # Print top candidates
        print(f"\nTop 5 candidates:")
        for i, candidate in enumerate(candidates[:5]):
            print(f"{i+1}. M={candidate['M']:.1e}, α={candidate['alpha']:.3f}, β={candidate['beta']:.3f}")
            print(f"   ANEC = {candidate['anec_value']:.3e} W")
            print(f"   Enhancement = {candidate['enhancement_factor']:.1f}×")
        
        return candidates
        
    else:
        print(f"\nNo candidates found exceeding target {target_anec:.1e} W")
        print("Consider relaxing the target or expanding parameter ranges")
        return []

if __name__ == "__main__":
    # Run focused scan targeting -1e-12 W
    candidates = focused_scan_robust(target_anec=-1e-12)
    
    if candidates:
        print(f"\n✓ SUCCESS: Found {len(candidates)} robust candidates")
    else:
        print("\n⚠ No candidates found - trying relaxed target...")
        # Try with relaxed target
        candidates = focused_scan_robust(target_anec=-1e-15)
