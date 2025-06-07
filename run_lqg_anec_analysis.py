#!/usr/bin/env python3
"""
run_lqg_anec_analysis.py

Command-line interface for comprehensive LQG-ANEC violation analysis.
Integrates coherent states, polymer corrections, effective field theory,
and warp bubble metrics for systematic ANEC violation studies.
"""

import argparse
import sys
import os
import json
import numpy as np
from pathlib import Path
import subprocess

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.anec_violation_analysis import (
    coherent_state_anec_violation,
    scan_anec_violation_parameters, 
    save_anec_analysis_plots,
    effective_action_anec_analysis
)
from src.spin_network_utils import build_flat_graph
from src.coherent_states import CoherentState
from src.polymer_quantization import polymer_scale_hierarchy

# Optional imports (graceful fallback if not available)
try:
    from src.warp_bubble_analysis import WarpBubbleAnalysis
except ImportError:
    WarpBubbleAnalysis = None
    
try:
    from src.negative_energy import compute_negative_energy_region
except ImportError:
    compute_negative_energy_region = None

def setup_output_directory(output_dir: str) -> Path:
    """Create output directory and return Path object."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path

def make_json_serializable(obj):
    """Convert objects to JSON-serializable format."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def run_single_point_analysis(args) -> dict:
    """Run single-point ANEC violation analysis."""
    print("Running single-point ANEC analysis...")
    
    result = coherent_state_anec_violation(
        n_nodes=args.n_nodes,
        alpha=args.alpha,
        mu=args.mu,
        tau=args.tau,
        field_amplitude=args.field_amplitude
    )
    
    print(f"Results:")
    print(f"  ANEC Integral (Classical): {result['anec_integral_classical']:.3e}")
    print(f"  ANEC Integral (Midisuperspace): {result['anec_integral_midisuperspace']:.3e}")
    print(f"  ANEC Bound: {result['anec_bound']:.3e}")
    print(f"  Classical Violation: {result['classical_violation']}")
    print(f"  Violation Magnitude: {result['violation_magnitude_classical']:.3f}")
    
    return result

def run_parameter_scan(args) -> dict:
    """Run systematic parameter space scan."""
    print("Running parameter space scan...")
    
    # Generate parameter ranges
    if args.mu_range:
        mu_min, mu_max, mu_steps = map(float, args.mu_range.split(','))
        mu_range = np.linspace(mu_min, mu_max, int(mu_steps))
    else:
        mu_range = np.linspace(0.01, 0.5, 20)
    
    if args.tau_range:
        tau_min, tau_max, tau_steps = map(float, args.tau_range.split(','))
        tau_range = np.logspace(np.log10(tau_min), np.log10(tau_max), int(tau_steps))
    else:
        tau_range = np.logspace(-1, 1, 20)
    
    print(f"Scanning μ ∈ [{mu_range.min():.3f}, {mu_range.max():.3f}] ({len(mu_range)} points)")
    print(f"Scanning τ ∈ [{tau_range.min():.3f}, {tau_range.max():.3f}] ({len(tau_range)} points)")
    
    results = scan_anec_violation_parameters(
        mu_range=mu_range,
        tau_range=tau_range,
        n_nodes=args.n_nodes,
        alpha=args.alpha
    )
    
    # Find optimal parameters
    max_violation_idx = np.unravel_index(
        np.nanargmax(results['violation_grid']), 
        results['violation_grid'].shape
    )
    optimal_mu = mu_range[max_violation_idx[0]]
    optimal_tau = tau_range[max_violation_idx[1]]
    max_violation = results['violation_grid'][max_violation_idx]
    
    print(f"Maximum violation: {max_violation:.3f}")
    print(f"Optimal parameters: μ = {optimal_mu:.3f}, τ = {optimal_tau:.3f}")
    
    return results

def run_eft_analysis(args) -> dict:
    """Run effective field theory analysis."""
    print("Running EFT-based ANEC analysis...")
    
    # Build spin network and coherent state
    graph = build_flat_graph(args.n_nodes, connectivity="cubic")
    coherent_state = CoherentState(graph, args.alpha).peak_on_flat()
    
    # EFT analysis
    eft_results = effective_action_anec_analysis(graph, coherent_state)
    
    print(f"EFT ANEC Violation Estimate: {eft_results['anec_violation_eft']:.3e}")
    print(f"Non-zero EFT Coefficients:")
    for term, coeff in eft_results['eft_coefficients'].items():
        if abs(coeff) > 1e-10:
            print(f"  {term}: {coeff:.3e}")
    
    return eft_results

def run_warp_bubble_comparison(args) -> dict:
    """Compare LQG results with warp bubble analysis."""
    print("Running warp bubble comparison...")
    
    # Check if warp bubble modules are available
    if WarpBubbleAnalysis is None or compute_negative_energy_region is None:
        print("Warning: Warp bubble analysis modules not available, skipping...")
        return {'error': 'Warp bubble modules not available'}
    
    try:
        # Initialize warp bubble analysis
        warp_analysis = WarpBubbleAnalysis()
        
        # Compute negative energy for comparison
        warp_result = compute_negative_energy_region(
            lattice_size=args.n_nodes,
            polymer_scale=args.mu,
            field_amplitude=args.field_amplitude
        )
        
        print(f"Warp Bubble Analysis:")
        print(f"  Total Negative Energy: {warp_result['total_negative_energy']:.3e}")
        print(f"  Negative Energy Sites: {warp_result['negative_sites']}")
        print(f"  Polymer Enhancement: {warp_result['polymer_enhancement']}")
        
        return warp_result
        
    except Exception as e:
        print(f"Warning: Warp bubble analysis failed: {e}")
        return {'error': f'Warp bubble analysis failed: {e}'}

def save_results(results: dict, output_path: Path, args):
    """Save all results to files."""
    print(f"Saving results to {output_path}/")
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save JSON summary
    summary = {
        'parameters': {
            'n_nodes': args.n_nodes,
            'alpha': args.alpha,
            'mu': args.mu,
            'tau': args.tau,
            'field_amplitude': args.field_amplitude
        },        'analysis_type': args.analysis_type,
        'results': make_json_serializable(results)
    }
    
    with open(output_path / "analysis_summary.json", 'w') as f:
        json.dump(make_json_serializable(summary), f, indent=2)
    
    # Save plots if parameter scan was performed
    if 'violation_grid' in results:
        save_anec_analysis_plots(results, str(output_path))
    
    print(f"Results saved to {output_path}/analysis_summary.json")

def main():
    parser = argparse.ArgumentParser(
        description="LQG-ANEC Framework: Comprehensive ANEC Violation Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single point analysis
  python run_lqg_anec_analysis.py --analysis-type single --mu 0.1 --tau 1.0
  
  # Parameter scan
  python run_lqg_anec_analysis.py --analysis-type scan --mu-range "0.01,0.5,20" --tau-range "0.1,10.0,20"
  
  # Full analysis suite
  python run_lqg_anec_analysis.py --analysis-type full --output-dir results_full
  
  # EFT analysis only
  python run_lqg_anec_analysis.py --analysis-type eft --n-nodes 64
  
  # Coherent state pipeline test
  python run_lqg_anec_analysis.py --analysis-type coherent-test
  
  # Quantum inequality bound comparison
  python run_lqg_anec_analysis.py --analysis-type qi-comparison
"""
    )
      # Analysis type
    parser.add_argument('--analysis-type', type=str, default='single',
                       choices=['single', 'scan', 'eft', 'warp', 'full', 'coherent-test', 'qi-comparison'],
                       help='Type of analysis to perform')
    
    # Physical parameters
    parser.add_argument('--n-nodes', type=int, default=64,
                       help='Number of spin network nodes')
    parser.add_argument('--alpha', type=float, default=0.05,
                       help='Coherent state spread parameter')
    parser.add_argument('--mu', type=float, default=0.1,
                       help='Polymer parameter')
    parser.add_argument('--tau', type=float, default=1.0,
                       help='Sampling timescale')
    parser.add_argument('--field-amplitude', type=float, default=1.0,
                       help='Matter field amplitude')
    
    # Parameter ranges for scanning
    parser.add_argument('--mu-range', type=str, default=None,
                       help='Polymer parameter range: "min,max,steps"')
    parser.add_argument('--tau-range', type=str, default=None,
                       help='Timescale range: "min,max,steps"')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--save-plots', action='store_true',
                       help='Save analysis plots')
    
    args = parser.parse_args()
    
    # Setup output directory
    output_path = setup_output_directory(args.output_dir)
    
    print("LQG-ANEC Framework: Comprehensive ANEC Violation Analysis")
    print("=" * 60)
    print(f"Analysis Type: {args.analysis_type}")
    print(f"Parameters: n_nodes={args.n_nodes}, α={args.alpha}, μ={args.mu}, τ={args.tau}")
    print(f"Output Directory: {output_path}")
    print()
    
    results = {}
    
    # Run analysis based on type
    if args.analysis_type == 'single':
        results.update(run_single_point_analysis(args))
        
    elif args.analysis_type == 'scan':
        results.update(run_parameter_scan(args))
        
    elif args.analysis_type == 'eft':
        results.update(run_eft_analysis(args))
        
    elif args.analysis_type == 'warp':
        results.update(run_warp_bubble_comparison(args))
        
    elif args.analysis_type == 'full':
        print("Running comprehensive analysis suite...")
        print("\n" + "="*40)
        results['single_point'] = run_single_point_analysis(args)
        print("\n" + "="*40)
        results['parameter_scan'] = run_parameter_scan(args)
        print("\n" + "="*40)
        results['eft_analysis'] = run_eft_analysis(args)
        print("\n" + "="*40)
        results['warp_comparison'] = run_warp_bubble_comparison(args)
        
        # For full analysis, save each component separately
        for analysis_name, analysis_results in results.items():
            component_results = {'parameters': vars(args), analysis_name: analysis_results}
            save_results(component_results, output_path / analysis_name, args)
    
    # Save results
    if args.analysis_type != 'full':  # Full analysis saves components separately
        save_results(results, output_path, args)
    
    print(f"\nAnalysis complete!")
    print(f"Results saved to: {output_path.absolute()}")
    
    # Print polymer scale hierarchy for reference
    if args.analysis_type in ['single', 'full']:
        print(f"\nPolymer Scale Hierarchy (for reference):")
        hierarchy = polymer_scale_hierarchy()
        for scale_name, scale_value in hierarchy.items():
            print(f"  {scale_name}: μ = {scale_value:.3e}")

if __name__ == "__main__":
    main()
