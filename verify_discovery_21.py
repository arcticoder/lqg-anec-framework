#!/usr/bin/env python3
"""
Example Code to Verify Discovery 21: Ghost EFT ANEC Violation

This script reproduces the optimal Ghost EFT ANEC violation result
from Discovery 21 and can form the basis for automated batch scans
and dashboard updates.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from ghost_condensate_eft import GhostCondensateEFT
    from utils.smearing import GaussianSmear
    IMPORTS_OK = True
except ImportError as e:
    print(f"Warning: Missing imports - {e}")
    print("Using fallback implementation...")
    IMPORTS_OK = False

def gaussian_smear_kernel(timescale=7*24*3600):
    """Fallback Gaussian smearing kernel implementation."""
    def kernel(tau):
        import numpy as np
        return (1 / np.sqrt(2 * np.pi * timescale**2)) * np.exp(-tau**2 / (2 * timescale**2))
    return kernel

def verify_discovery_21_result():
    """Reproduce Discovery 21 result: Optimal Ghost EFT ANEC violation."""
    
    print("=== Discovery 21: Ghost EFT ANEC Verification ===")
    print("Reproducing optimal parameters: M=1000, α=0.01, β=0.1")
    print("Expected ANEC: -1.418×10⁻¹² W")
    
    if not IMPORTS_OK:
        print("⚠ Using fallback implementation (imports not available)")
        return None
    
    try:
        # Reproduce Discovery 21 result
        eft = GhostCondensateEFT(M=1000, alpha=0.01, beta=0.1)
        
        # Use week-scale smearing (as in Discovery 21)
        if IMPORTS_OK:
            try:
                smear = GaussianSmear(timescale=7*24*3600)
                anec_value = eft.compute_anec(kernel=smear.kernel)
            except:
                # Fallback to direct kernel
                kernel = gaussian_smear_kernel(timescale=7*24*3600)
                anec_value = eft.compute_anec(kernel)
        else:
            kernel = gaussian_smear_kernel(timescale=7*24*3600)
            anec_value = eft.compute_anec(kernel)
        
        print(f"Optimal Ghost EFT ANEC = {anec_value:.3e} W")
        
        # Verify against expected result
        expected = -1.418e-12
        relative_error = abs(anec_value - expected) / abs(expected)
        
        print(f"Expected ANEC        = {expected:.3e} W")
        print(f"Relative error       = {relative_error:.2%}")
        
        if relative_error < 0.1:  # 10% tolerance
            print("✓ Discovery 21 result VERIFIED")
            status = "CONFIRMED"
        else:
            print("⚠ Result differs from expected value")
            status = "DEVIATION"
        
        return {
            'computed_anec': anec_value,
            'expected_anec': expected,
            'relative_error': relative_error,
            'verification_status': status,
            'parameters': {'M': 1000, 'alpha': 0.01, 'beta': 0.1}
        }
        
    except Exception as e:
        print(f"✗ Verification failed: {e}")
        return None

def batch_scan_demo(num_configs=10):
    """Demo of automated batch scan for dashboard updates."""
    
    print(f"\n=== Batch Scan Demo ({num_configs} configurations) ===")
    
    if not IMPORTS_OK:
        print("⚠ Batch scan requires full Ghost EFT implementation")
        return []
    
    import numpy as np
    
    # Parameter variations around optimal
    np.random.seed(42)  # Reproducible results
    
    results = []
    
    for i in range(num_configs):
        # Small variations around optimal (±5%)
        M = 1000 * (0.95 + 0.1 * np.random.random())
        alpha = 0.01 * (0.95 + 0.1 * np.random.random())  
        beta = 0.1 * (0.95 + 0.1 * np.random.random())
        
        try:
            eft = GhostCondensateEFT(M=M, alpha=alpha, beta=beta)
            kernel = gaussian_smear_kernel()
            anec = eft.compute_anec(kernel)
            
            result = {
                'config_id': i+1,
                'M': M,
                'alpha': alpha,
                'beta': beta,
                'anec_value': anec,
                'qi_violation': anec < 0
            }
            
            results.append(result)
            print(f"Config {i+1:2d}: ANEC={anec:.2e} W (M={M:.0f}, α={alpha:.3f}, β={beta:.3f})")
            
        except Exception as e:
            print(f"Config {i+1:2d}: Failed - {e}")
    
    # Summary
    violations = [r for r in results if r['qi_violation']]
    violation_rate = len(violations) / len(results) * 100 if results else 0
    
    print(f"\nBatch Scan Summary:")
    print(f"Total configs: {len(results)}")
    print(f"QI violations: {len(violations)} ({violation_rate:.1f}%)")
    
    if violations:
        anec_values = [v['anec_value'] for v in violations]
        print(f"ANEC range: [{min(anec_values):.2e}, {max(anec_values):.2e}] W")
    
    return results

def main():
    """Main demonstration function."""
    print("Discovery 21: Ghost EFT Verification & Batch Scan Demo")
    print("=" * 60)
    
    # 1. Verify Discovery 21 result
    verification = verify_discovery_21_result()
    
    if verification:
        print(f"\n✓ Discovery 21 verification completed")
        
        # 2. Demo batch scanning for dashboard updates
        batch_results = batch_scan_demo(num_configs=5)
        
        print(f"\n✓ Batch scan demo completed")
        print(f"Ready for dashboard integration and automated scanning")
        
        return {
            'verification': verification,
            'batch_demo': batch_results,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        print(f"\n✗ Verification failed - check Ghost EFT implementation")
        return None

if __name__ == "__main__":
    results = main()
    
    if results:
        # Save results for integration
        import json
        from pathlib import Path
        
        output_file = Path("results") / "discovery_21_verification.json"
        output_file.parent.mkdir(exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")
        print("Ready for computational pipeline and dashboard integration!")
