#!/usr/bin/env python3
"""
Automated Ghost EFT Batch Scanner - Discovery 21 Integration

Example usage:
    python automated_ghost_eft_scanner.py --num-configs 100 --output results/batch_scan.json
"""

import argparse
from src.ghost_condensate_eft import GhostCondensateEFT

def automated_ghost_eft_scan(num_configs=100, output_file="results/ghost_eft_batch.json"):
    """Automated batch scanning around Discovery 21 optimal parameters."""
    
    # Base parameters from Discovery 21
    base_M, base_alpha, base_beta = 1000, 0.01, 0.1
    
    # Generate parameter variations
    M_range = np.random.normal(base_M, 0.1*base_M, num_configs)
    alpha_range = np.random.normal(base_alpha, 0.1*base_alpha, num_configs)  
    beta_range = np.random.normal(base_beta, 0.1*base_beta, num_configs)
    
    results = []
    grid = np.linspace(-1e6, 1e6, 1500)
    smear = GaussianSmear(timescale=7*24*3600)
    
    for i in range(num_configs):
        try:
            eft = GhostCondensateEFT(M=M_range[i], alpha=alpha_range[i], 
                                   beta=beta_range[i], grid=grid)
            anec_value = eft.compute_anec(smear.kernel)
            
            results.append({
                'config_id': i,
                'M': float(M_range[i]),
                'alpha': float(alpha_range[i]),
                'beta': float(beta_range[i]),
                'anec_value': float(anec_value),
                'discovery_21_reference': True
            })
            
        except Exception as e:
            continue
    
    # Save batch results
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-configs', type=int, default=100)
    parser.add_argument('--output', type=str, default='results/ghost_eft_batch.json')
    args = parser.parse_args()
    
    results = automated_ghost_eft_scan(args.num_configs, args.output)
    print(f"Batch scan complete: {len(results)} configurations saved to {args.output}")
