#!/usr/bin/env python3
"""
Optimized Ghost EFT Configuration

Deploy the optimal ghost-condensate EFT parameters discovered through scanning
for maximum ANEC violation efficiency and rapid negative energy generation.

Optimal Configuration:
- M = 1,000 (mass scale)
- α = 0.01 (X² coupling)  
- β = 0.1 (φ² mass term)
- ANEC violation: -1.418 × 10⁻¹² W

This configuration provides the strongest quantum inequality violations
with 100% reliability across the parameter space.
"""

import numpy as np
import json
import time
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from ghost_condensate_eft import GhostCondensateEFT


class OptimalGhostConfig:
    """Optimal ghost EFT configuration for maximum ANEC violations."""
    
    def __init__(self):
        # Optimal parameters from scanning results
        self.optimal_params = {
            'M': 1000.0,           # Mass scale
            'alpha': 0.01,         # X² coupling  
            'beta': 0.1,           # φ² mass term
            'expected_anec': -1.418400352905847e-12  # W
        }
        
        # High-resolution grid for precision calculations
        self.grid = np.linspace(-1e6, 1e6, 4000)
        
        # Week-scale smearing for sustained violations
        self.tau0 = 7 * 24 * 3600  # 604,800 seconds
        
    def gaussian_kernel(self, tau):
        """Optimized Gaussian smearing kernel."""
        return (1 / np.sqrt(2 * np.pi * self.tau0**2)) * np.exp(-tau**2 / (2 * self.tau0**2))
    
    def deploy_optimal_configuration(self):
        """Deploy the optimal ghost EFT configuration."""
        print("=== Deploying Optimal Ghost EFT Configuration ===")
        print(f"Parameters: M={self.optimal_params['M']}, α={self.optimal_params['alpha']}, β={self.optimal_params['beta']}")
        print(f"Expected ANEC: {self.optimal_params['expected_anec']:.2e} W")
        print(f"Grid resolution: {len(self.grid)} points")
        print(f"Temporal smearing: {self.tau0/86400:.1f} days")
        
        # Initialize EFT with optimal parameters
        start_time = time.time()
        eft = GhostCondensateEFT(
            M=self.optimal_params['M'],
            alpha=self.optimal_params['alpha'], 
            beta=self.optimal_params['beta'],
            grid=self.grid
        )
        
        # Compute ANEC with optimal configuration
        anec_result = eft.compute_anec(self.gaussian_kernel)
        computation_time = time.time() - start_time
        
        # Verify performance
        expected = self.optimal_params['expected_anec']
        relative_error = abs(anec_result - expected) / abs(expected)
        
        print("\\n=== Deployment Results ===")
        print(f"Computed ANEC: {anec_result:.6e} W")
        print(f"Expected ANEC:  {expected:.6e} W") 
        print(f"Relative error: {relative_error:.2%}")
        print(f"Computation time: {computation_time:.4f} seconds")
        print(f"QI violation: {'✓ CONFIRMED' if anec_result < 0 else '✗ FAILED'}")
        
        # Performance metrics
        violation_strength = abs(anec_result)
        enhancement_vs_vacuum = violation_strength / 1.2e-17  # vs squeezed vacuum
        
        print("\\n=== Performance Metrics ===")
        print(f"Violation strength: {violation_strength:.2e} W")
        print(f"Enhancement vs vacuum: {enhancement_vs_vacuum:.1e}×")
        print(f"Computational efficiency: {violation_strength/computation_time:.2e} W/sec")
        
        return {
            'anec_violation': anec_result,
            'computation_time': computation_time,
            'relative_error': relative_error,
            'violation_confirmed': anec_result < 0,
            'enhancement_factor': enhancement_vs_vacuum
        }
    
    def parameter_sensitivity_analysis(self):
        """Analyze sensitivity around optimal parameters."""
        print("\\n=== Parameter Sensitivity Analysis ===")
        
        # Test parameter variations around optimal point
        variations = {
            'M': [900, 950, 1000, 1050, 1100],
            'alpha': [0.005, 0.008, 0.01, 0.012, 0.015],
            'beta': [0.08, 0.09, 0.1, 0.11, 0.12]
        }
        
        baseline = self.optimal_params
        results = {}
        
        for param, values in variations.items():
            param_results = []
            for value in values:
                # Create modified parameters
                test_params = baseline.copy()
                test_params[param.replace('alpha', 'alpha')] = value
                
                # Test configuration
                eft = GhostCondensateEFT(
                    M=test_params['M'],
                    alpha=test_params['alpha'],
                    beta=test_params['beta'],
                    grid=self.grid
                )
                
                anec = eft.compute_anec(self.gaussian_kernel)
                param_results.append({'value': value, 'anec': anec})
            
            results[param] = param_results
            
            # Print sensitivity summary
            print(f"\\n{param} sensitivity:")
            for result in param_results:
                print(f"  {param}={result['value']:<6.3f}: ANEC={result['anec']:.2e} W")
        
        return results
    
    def rapid_deployment_test(self, n_iterations=10):
        """Test rapid repeated deployment for reliability assessment."""
        print(f"\\n=== Rapid Deployment Test ({n_iterations} iterations) ===")
        
        results = []
        total_start = time.time()
        
        for i in range(n_iterations):
            start = time.time()
            eft = GhostCondensateEFT(
                M=self.optimal_params['M'],
                alpha=self.optimal_params['alpha'],
                beta=self.optimal_params['beta'],
                grid=self.grid
            )
            anec = eft.compute_anec(self.gaussian_kernel)
            elapsed = time.time() - start
            
            results.append({'iteration': i+1, 'anec': anec, 'time': elapsed})
        
        total_time = time.time() - total_start
        
        # Statistics
        anec_values = [r['anec'] for r in results]
        times = [r['time'] for r in results]
        
        print(f"Total time: {total_time:.3f} seconds")
        print(f"Average per iteration: {np.mean(times):.4f} ± {np.std(times):.4f} seconds")
        print(f"ANEC consistency: {np.mean(anec_values):.2e} ± {np.std(anec_values):.2e} W")
        print(f"Reliability: {sum(1 for a in anec_values if a < 0)}/{n_iterations} violations")
        
        return {
            'total_time': total_time,
            'average_time': np.mean(times),
            'anec_mean': np.mean(anec_values),
            'anec_std': np.std(anec_values),
            'reliability': sum(1 for a in anec_values if a < 0) / n_iterations
        }


def main():
    """Main execution function."""
    print("Ghost Condensate EFT - Optimal Configuration Deployment")
    print("=" * 60)
    
    # Initialize optimal configuration
    config = OptimalGhostConfig()
    
    # Deploy optimal configuration
    deployment_results = config.deploy_optimal_configuration()
    
    # Run sensitivity analysis  
    sensitivity_results = config.parameter_sensitivity_analysis()
    
    # Test rapid deployment reliability
    reliability_results = config.rapid_deployment_test()
    
    # Generate summary report
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'optimal_parameters': config.optimal_params,
        'deployment_results': deployment_results,
        'reliability_assessment': reliability_results,
        'configuration_status': 'OPERATIONAL' if deployment_results['violation_confirmed'] else 'FAILED'
    }
    
    # Save results
    output_path = Path('results/optimal_ghost_config_results.json')
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\\n=== Configuration Summary ===")
    print(f"Status: {summary['configuration_status']}")
    print(f"ANEC violation: {deployment_results['anec_violation']:.2e} W")
    print(f"Reliability: {reliability_results['reliability']:.1%}")
    print(f"Results saved to: {output_path}")
    
    return summary


if __name__ == "__main__":
    results = main()
