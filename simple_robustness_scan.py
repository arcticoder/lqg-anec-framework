#!/usr/bin/env python3
"""
Simple Robustness Scan for Ghost EFT

A streamlined robustness validation around the optimal Ghost EFT configuration.
"""

import sys
import json
import time
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append('src')

try:
    from ghost_condensate_eft import GhostCondensateEFT
    print("✓ Ghost EFT module imported successfully")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def simple_robustness_check():
    """Simple robustness check around optimal parameters."""
    print("=== Ghost EFT Robustness Validation ===")
    
    # Optimal parameters
    optimal = {
        'M': 1000.0,
        'alpha': 0.01,
        'beta': 0.1,
        'target': -1.418e-12
    }
    
    print(f"Optimal: M={optimal['M']}, α={optimal['alpha']}, β={optimal['beta']}")
    print(f"Target ANEC: {optimal['target']:.2e} W")
    
    # Grid setup
    grid = np.linspace(-1e6, 1e6, 2000)
    tau0 = 7 * 24 * 3600  # Week-scale
    
    def gaussian_kernel(tau):
        return (1 / np.sqrt(2 * np.pi * tau0**2)) * np.exp(-tau**2 / (2 * tau0**2))
    
    # Test parameter variations (±10% around optimal)
    variations = [
        {'M': 900, 'alpha': 0.009, 'beta': 0.09},   # -10%
        {'M': 1000, 'alpha': 0.01, 'beta': 0.1},    # Optimal
        {'M': 1100, 'alpha': 0.011, 'beta': 0.11},  # +10%
        {'M': 950, 'alpha': 0.01, 'beta': 0.1},     # M variation
        {'M': 1000, 'alpha': 0.008, 'beta': 0.1},   # α variation
        {'M': 1000, 'alpha': 0.01, 'beta': 0.12},   # β variation
    ]
    
    results = []
    violations = 0
    
    print(f"\nTesting {len(variations)} parameter configurations...")
    
    for i, params in enumerate(variations, 1):
        try:
            print(f"\n{i}. M={params['M']}, α={params['alpha']:.3f}, β={params['beta']:.2f}")
            
            eft = GhostCondensateEFT(
                M=params['M'],
                alpha=params['alpha'],
                beta=params['beta'],
                grid=grid
            )
            
            anec = eft.compute_anec(gaussian_kernel)
            
            result = {
                'config': i,
                'parameters': params,
                'anec_value': float(anec),
                'qi_violation': anec < 0,
                'target_ratio': float(anec / optimal['target']) if anec < 0 else 0
            }
            
            results.append(result)
            
            if anec < 0:
                violations += 1
                print(f"   ✓ ANEC: {anec:.2e} W (violation confirmed)")
                print(f"   ✓ Ratio: {anec/optimal['target']:.2f}× target strength")
            else:
                print(f"   ✗ ANEC: {anec:.2e} W (no violation)")
                
        except Exception as e:
            print(f"   ✗ Error: {e}")
            continue
    
    # Summary
    success_rate = violations / len(results) * 100 if results else 0
    
    print(f"\n=== Robustness Summary ===")
    print(f"Configurations tested: {len(results)}")
    print(f"Successful violations: {violations}")
    print(f"Success rate: {success_rate:.1f}%")
    
    if violations >= len(results) * 0.8:  # 80% threshold
        print(f"✓ ROBUSTNESS CONFIRMED: Parameter variations maintain ANEC violations")
        status = "VALIDATED"
    else:
        print(f"⚠ ROBUSTNESS UNCERTAIN: Lower success rate detected")
        status = "NEEDS_REVIEW"
    
    # Save results
    output = {
        'validation_type': 'parameter_robustness',
        'optimal_config': optimal,
        'test_results': results,
        'summary': {
            'total_tests': len(results),
            'violations': violations,
            'success_rate_percent': success_rate,
            'robustness_status': status
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Save to JSON
    output_file = Path("results") / "ghost_eft_robustness_validation.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return output


if __name__ == "__main__":
    try:
        results = simple_robustness_check()
        print(f"\n🎯 Robustness validation completed successfully!")
    except Exception as e:
        print(f"\n✗ Validation failed: {e}")
        sys.exit(1)
