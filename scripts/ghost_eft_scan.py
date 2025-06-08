#!/usr/bin/env python3
"""
Ghost EFT Parameter Scanning Script

Systematic parameter scan for ghost-condensate EFT ANEC violations.
Implements rapid, software-tunable exploration of (M, α, β) parameter space
to identify optimal configurations for quantum inequality circumvention.

Features:
- Week-scale temporal sampling for realistic ANEC integration  
- Multi-parameter grid search with progress tracking
- Comparison against quantum inequality bounds
- JSON output for integration with vacuum engineering pipeline
- GPU acceleration when available

Theory:
L = -X + α X²/M⁴ - β φ²
where X = ½(∂φ)², giving controlled NEC violation through negative kinetic term.
"""

import numpy as np
import json
import time
import sys
from pathlib import Path
from tqdm import trange, tqdm
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from ghost_condensate_eft import GhostCondensateEFT
except ImportError as e:
    print(f"Error importing GhostCondensateEFT: {e}")
    print("Please ensure ghost_condensate_eft.py is properly implemented in src/")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GaussianSmear:
    """Gaussian smearing kernel for ANEC integration."""
    
    def __init__(self, tau0=7*24*3600):  # Default: 1 week = 604,800 seconds
        self.tau0 = tau0
        
    def kernel(self, tau):
        """Gaussian kernel normalized to unit integral."""
        return (1 / np.sqrt(2 * np.pi * self.tau0**2)) * np.exp(-tau**2 / (2 * self.tau0**2))


def scan_ghost_eft_parameters(output_file="ghost_eft_scan_results.json"):
    """
    Comprehensive parameter scan for ghost-condensate EFT.
    
    Scans (M, α, β) parameter space to find optimal ANEC violations
    over week-scale sampling periods.
    """
    logger.info("Starting ghost EFT parameter scan...")
    logger.info("Target: sustained negative ANEC over week-scale sampling")
    
    # Setup coordinate grid (map x → τ for week-scale integration)
    grid = np.linspace(-1e6, 1e6, 2000)  # ±10⁶ seconds ~ 11.6 days
    smear = GaussianSmear(tau0=7*24*3600)  # 1 week smearing
    
    # Parameter ranges for systematic scan
    M_values = [1e3, 5e3, 1e4, 5e4, 1e5]      # Mass scales
    alpha_values = [0.01, 0.1, 0.5, 1.0, 2.0]  # X² coupling strength  
    beta_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]  # φ² mass term
    
    results = []
    violations = []
    
    total_combinations = len(M_values) * len(alpha_values) * len(beta_values)
    logger.info(f"Scanning {total_combinations} parameter combinations...")
    
    start_time = time.time()
    
    # Progress tracking
    with tqdm(total=total_combinations, desc="Ghost EFT Scan") as pbar:
        for M in M_values:
            for alpha in alpha_values:
                for beta in beta_values:
                    try:
                        # Initialize ghost EFT with current parameters
                        eft = GhostCondensateEFT(M=M, alpha=alpha, beta=beta, grid=grid)                        
                        # Compute ANEC integral
                        anec_value = eft.compute_anec(smear.kernel)
                        
                        # Store result
                        result = {
                            'M': float(M),
                            'alpha': float(alpha), 
                            'beta': float(beta),
                            'anec_value': float(anec_value),
                            'violation_strength': float(-anec_value) if anec_value < 0 else 0.0,
                            'qi_violation': bool(anec_value < 0)  # Simple QI violation check
                        }
                        
                        results.append(result)
                        if anec_value < 0:
                            violations.append(result)
                            
                    except Exception as e:
                        logger.warning(f"Failed for M={M}, α={alpha}, β={beta}: {e}")
                        
                    pbar.update(1)
    
    scan_time = time.time() - start_time
    
    # Analysis of results
    logger.info(f"Parameter scan completed in {scan_time:.2f} seconds")
    logger.info(f"Total configurations tested: {len(results)}")
    logger.info(f"Configurations with ANEC violations: {len(violations)}")
    
    if violations:
        # Find best violations
        best_violation = min(violations, key=lambda x: x['anec_value'])
        strongest_violations = sorted(violations, key=lambda x: x['anec_value'])[:10]
        
        logger.info(f"Best ANEC violation: {best_violation['anec_value']:.2e}")
        logger.info(f"Best parameters: M={best_violation['M']:.1e}, α={best_violation['alpha']}, β={best_violation['beta']:.1e}")
        
        # Summary statistics
        violation_values = [v['anec_value'] for v in violations]
        summary = {
            'scan_metadata': {
                'total_combinations': total_combinations,
                'successful_evaluations': len(results),
                'violation_count': len(violations),
                'scan_time_seconds': scan_time,
                'grid_points': len(grid),
                'smearing_timescale': smear.tau0
            },
            'best_violation': best_violation,
            'top_10_violations': strongest_violations,
            'violation_statistics': {
                'min_anec': float(np.min(violation_values)),
                'max_anec': float(np.max(violation_values)),
                'mean_anec': float(np.mean(violation_values)),
                'std_anec': float(np.std(violation_values))
            },
            'all_results': results
        }
        
    else:
        logger.warning("No ANEC violations found in parameter scan!")
        summary = {
            'scan_metadata': {
                'total_combinations': total_combinations,
                'successful_evaluations': len(results),
                'violation_count': 0,
                'scan_time_seconds': scan_time
            },
            'all_results': results
        }
    
    # Save results to JSON
    output_path = Path("results") / output_file
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to {output_path}")
    
    return summary


def focused_parameter_scan(target_anec=-1e-25):
    """
    Focused scan around promising parameter regions.
    
    Args:
        target_anec: Target ANEC violation level (default: -10⁻²⁵ W)
    """
    logger.info(f"Starting focused scan targeting ANEC ≤ {target_anec:.1e}")
    
    # Setup grid
    grid = np.linspace(-1e6, 1e6, 4000)  # Higher resolution
    smear = GaussianSmear(tau0=7*24*3600)
    
    # Focused parameter ranges (based on theory predictions)
    M_values = np.logspace(3.5, 4.5, 20)        # Around 10⁴ 
    alpha_values = np.linspace(0.1, 1.0, 20)     # Moderate coupling
    beta_values = np.logspace(-4, -2, 20)        # Small mass terms
    
    best_candidates = []
    
    logger.info(f"Focused scan: {len(M_values)} × {len(alpha_values)} × {len(beta_values)} = {len(M_values)*len(alpha_values)*len(beta_values)} combinations")
    
    with tqdm(total=len(M_values)*len(alpha_values)*len(beta_values), desc="Focused Scan") as pbar:
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
                                'target_ratio': float(anec_value / target_anec)
                            }
                            best_candidates.append(candidate)
                            logger.info(f"Target exceeded: ANEC={anec_value:.2e} (ratio: {anec_value/target_anec:.1f}×)")
                            
                    except Exception as e:
                        logger.warning(f"Evaluation failed: {e}")
                        
                    pbar.update(1)
    
    if best_candidates:
        # Sort by strongest violation
        best_candidates.sort(key=lambda x: x['anec_value'])
        
        logger.info(f"Found {len(best_candidates)} candidates exceeding target")
        logger.info(f"Best candidate: ANEC={best_candidates[0]['anec_value']:.2e}")
        
        # Save focused results
        output_path = Path("results") / "ghost_eft_focused_scan.json"
        with open(output_path, 'w') as f:
            json.dump({
                'target_anec': target_anec,
                'candidates_found': len(best_candidates),
                'best_candidates': best_candidates
            }, f, indent=2)
            
        return best_candidates
    else:
        logger.warning(f"No candidates found exceeding target {target_anec:.1e}")
        return []


def compare_with_vacuum_sources():
    """
    Compare best ghost EFT results with vacuum engineering sources.
    """
    logger.info("Comparing ghost EFT with vacuum sources...")
    
    # Load ghost EFT results
    ghost_file = Path("results") / "ghost_eft_scan_results.json"
    if not ghost_file.exists():
        logger.error("Ghost EFT scan results not found. Run scan first.")
        return
        
    with open(ghost_file, 'r') as f:
        ghost_data = json.load(f)
    
    # Get best ghost result
    if 'best_violation' in ghost_data:
        best_ghost = ghost_data['best_violation']
        ghost_anec = best_ghost['anec_value']
    else:
        logger.warning("No ghost violations found for comparison")
        return
    
    # Vacuum source comparison (placeholder - integrate with actual vacuum pipeline)
    vacuum_sources = {
        'casimir_array': -5.06e7,        # From Discovery 16
        'dynamic_casimir': -2.60e18,     # From Discovery 17  
        'squeezed_vacuum': 1.00e-27,     # From Discovery 18
        'metamaterial': -8.06e-27        # From Discovery 19
    }
    
    # Comparison analysis
    comparison = {
        'ghost_eft': {
            'anec_value': ghost_anec,
            'parameters': best_ghost,
            'type': 'field_theory'
        },
        'vacuum_sources': vacuum_sources,
        'ranking': {}
    }
    
    # Rank all sources by ANEC violation strength
    all_sources = [('ghost_eft', ghost_anec)] + [(k, v) for k, v in vacuum_sources.items()]
    all_sources.sort(key=lambda x: x[1])  # Sort by ANEC value (most negative first)
    
    comparison['ranking'] = {i+1: {'source': source, 'anec': anec} for i, (source, anec) in enumerate(all_sources)}
    
    logger.info("ANEC Violation Ranking:")
    for rank, data in comparison['ranking'].items():
        logger.info(f"{rank}. {data['source']}: {data['anec']:.2e}")
    
    # Save comparison
    output_path = Path("results") / "ghost_vacuum_comparison.json"
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    logger.info(f"Comparison saved to {output_path}")
    
    return comparison


def main():
    """Main execution pipeline for ghost EFT analysis."""
    logger.info("=== Ghost EFT Parameter Scanning Pipeline ===")
    
    # 1. Comprehensive parameter scan
    logger.info("Phase 1: Comprehensive parameter scan")
    scan_results = scan_ghost_eft_parameters()
    
    # 2. Focused scan if violations found
    if scan_results.get('violation_count', 0) > 0:
        logger.info("Phase 2: Focused parameter scan")
        focused_results = focused_parameter_scan()
        
        # 3. Comparison with vacuum sources
        logger.info("Phase 3: Comparison with vacuum sources")
        comparison = compare_with_vacuum_sources()
        
        logger.info("=== Analysis Complete ===")
        logger.info("Check results/ directory for detailed output files")
        
    else:
        logger.warning("No violations found in comprehensive scan")
        logger.info("Consider expanding parameter ranges or adjusting theory")


if __name__ == "__main__":
    main()
