#!/usr/bin/env python3
"""
3D Mesh-Based Warp Bubble Validation Runner

This script runs the complete 3D mesh validation pipeline for
Discovery 21 Ghost/Phantom EFT and metamaterial Casimir sources.
"""

import numpy as np
import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import time
import matplotlib.pyplot as plt

# Import our framework components
from src.energy_source_interface import (
    GhostCondensateEFT, 
    MetamaterialCasimirSource,
    create_energy_source
)
from src.warp_bubble_solver import (
    WarpBubbleSolver, 
    WarpBubbleResult,
    compare_energy_sources
)

def create_discovery_21_source() -> GhostCondensateEFT:
    """
    Create Ghost EFT source with Discovery 21 optimal parameters.
    
    Returns:
        Configured Ghost EFT source
    """
    return GhostCondensateEFT(
        M=1000,     # Optimal mass scale from Discovery 21
        alpha=0.01, # Optimal coupling α from Discovery 21  
        beta=0.1,   # Optimal coupling β from Discovery 21
        R0=5.0,     # 5m bubble radius
        sigma=0.2   # 20cm shell width
    )

def create_metamaterial_source() -> MetamaterialCasimirSource:
    """
    Create metamaterial Casimir source with optimized parameters.
    
    Returns:
        Configured metamaterial source
    """
    return MetamaterialCasimirSource(
        epsilon=-2.0,           # Negative permittivity
        mu=-1.5,               # Negative permeability  
        cell_size=50e-9,       # 50 nm unit cells
        n_layers=100,          # 100 layer stack
        R0=5.0,                # 5m bubble radius
        shell_thickness=0.5    # 50cm shell thickness (thicker for better mesh sampling)
    )

def run_parameter_sweep(source_type: str, 
                       param_ranges: Dict[str, List[float]],
                       radius: float = 10.0,
                       resolution: int = 30) -> Dict[str, List[WarpBubbleResult]]:
    """
    Run parameter sweep for given source type.
    
    Args:
        source_type: 'ghost' or 'metamaterial'
        param_ranges: Dictionary of parameter names to value lists
        radius: Simulation radius
        resolution: Mesh resolution
        
    Returns:
        Dictionary of parameter combinations to results
    """
    solver = WarpBubbleSolver()
    results = {}
    
    param_names = list(param_ranges.keys())
    param_values = [param_ranges[name] for name in param_names]
    
    print(f"Running parameter sweep for {source_type}...")
    print(f"Parameters: {param_names}")
    print(f"Total combinations: {np.prod([len(vals) for vals in param_values])}")
    
    # Generate all combinations
    import itertools
    for i, combo in enumerate(itertools.product(*param_values)):
        params = dict(zip(param_names, combo))
        
        # Create source with these parameters
        try:
            source = create_energy_source(source_type, **params)
            result = solver.simulate(source, radius, resolution)
            
            key = f"{source_type}_{i:03d}"
            results[key] = result
            
            print(f"  {key}: {params} → Success: {result.success}, "
                  f"Energy: {result.energy_total:.2e} J, "
                  f"Stability: {result.stability:.3f}")
                  
        except Exception as e:
            print(f"  Failed combination {combo}: {e}")
    
    return results

def analyze_mesh_convergence(source_type: str = 'ghost',
                            resolutions: List[int] = [20, 30, 50, 70, 100]) -> Dict[int, WarpBubbleResult]:
    """
    Analyze mesh convergence for given source.
    
    Args:
        source_type: Type of energy source
        resolutions: List of mesh resolutions to test
        
    Returns:
        Dictionary mapping resolution to results
    """
    if source_type == 'ghost':
        source = create_discovery_21_source()
    else:
        source = create_metamaterial_source()
    
    solver = WarpBubbleSolver()
    results = {}
    
    print(f"Analyzing mesh convergence for {source.name}...")
    
    for res in resolutions:
        print(f"  Testing resolution {res}...")
        result = solver.simulate(source, radius=10.0, resolution=res)
        results[res] = result
        
        print(f"    Nodes: {result.mesh_nodes}, "
              f"Total Energy: {result.energy_total:.2e} J, "
              f"Time: {result.execution_time:.3f} s")
    
    return results

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

def generate_comparison_report(results: Dict[str, WarpBubbleResult],
                              output_dir: str = "results") -> str:
    """
    Generate comprehensive comparison report.
    
    Args:
        results: Dictionary of source names to results
        output_dir: Output directory for report
        
    Returns:
        Path to generated report
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary data
    summary = {
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'sources_tested': len(results),
        'successful_sources': sum(1 for r in results.values() if r.success),
        'results': {}
    }
    
    # Analyze each result
    for name, result in results.items():
        summary['results'][name] = {
            'success': result.success,
            'total_energy_J': result.energy_total,
            'stability': result.stability,
            'max_negative_density_J_per_m3': result.max_negative_density,
            'execution_time_s': result.execution_time,
            'mesh_nodes': result.mesh_nodes,
            'parameters': result.parameters
        }
      # Save JSON report
    json_path = os.path.join(output_dir, 'warp_bubble_comparison_report.json')
    with open(json_path, 'w') as f:
        json.dump(convert_numpy_types(summary), f, indent=2)
    
    # Generate markdown report
    md_path = os.path.join(output_dir, 'warp_bubble_comparison_report.md')
    with open(md_path, 'w') as f:
        f.write("# 3D Mesh-Based Warp Bubble Validation Report\n\n")
        f.write(f"Generated: {summary['timestamp']}\n\n")
        
        f.write("## Summary\n\n")
        f.write(f"- Sources tested: {summary['sources_tested']}\n")
        f.write(f"- Successful sources: {summary['successful_sources']}\n")
        f.write(f"- Success rate: {summary['successful_sources']/summary['sources_tested']*100:.1f}%\n\n")
        
        f.write("## Results by Source\n\n")
        
        # Sort by total energy (most negative first)
        sorted_results = sorted(results.items(), 
                               key=lambda x: x[1].energy_total)
        
        for name, result in sorted_results:
            f.write(f"### {name}\n\n")
            f.write(f"- **Success**: {result.success}\n")
            f.write(f"- **Total Energy**: {result.energy_total:.2e} J\n")
            f.write(f"- **Stability**: {result.stability:.3f}\n")
            f.write(f"- **Max Negative Density**: {result.max_negative_density:.2e} J/m³\n")
            f.write(f"- **Execution Time**: {result.execution_time:.3f} s\n")
            f.write(f"- **Mesh Nodes**: {result.mesh_nodes}\n")
            f.write(f"- **Parameters**: {result.parameters}\n\n")
    
    print(f"Report generated: {md_path}")
    return md_path

def create_visualization_comparison(results: Dict[str, WarpBubbleResult],
                                  output_dir: str = "results") -> str:
    """
    Create comparative visualizations.
    
    Args:
        results: Dictionary of results to visualize
        output_dir: Output directory
        
    Returns:
        Path to generated visualization
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Energy comparison
    ax = axes[0, 0]
    names = list(results.keys())
    energies = [results[name].energy_total for name in names]
    colors = ['red' if e < 0 else 'blue' for e in energies]
    
    bars = ax.bar(range(len(names)), energies, color=colors, alpha=0.7)
    ax.set_xlabel('Energy Source')
    ax.set_ylabel('Total Energy (J)')
    ax.set_title('Total Energy Comparison')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([name[:15] for name in names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, energy in zip(bars, energies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{energy:.1e}', ha='center', va='bottom' if height > 0 else 'top')
    
    # Stability comparison
    ax = axes[0, 1]
    stabilities = [results[name].stability for name in names]
    ax.bar(range(len(names)), stabilities, alpha=0.7, color='green')
    ax.set_xlabel('Energy Source')
    ax.set_ylabel('Stability Metric')
    ax.set_title('Stability Comparison')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([name[:15] for name in names], rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Performance comparison (execution time vs mesh nodes)
    ax = axes[1, 0]
    exec_times = [results[name].execution_time for name in names]
    mesh_nodes = [results[name].mesh_nodes for name in names]
    
    scatter = ax.scatter(mesh_nodes, exec_times, c=stabilities, 
                        cmap='viridis', s=100, alpha=0.7)
    ax.set_xlabel('Mesh Nodes')
    ax.set_ylabel('Execution Time (s)')
    ax.set_title('Performance: Time vs Mesh Size')
    ax.grid(True, alpha=0.3)
    
    # Add colorbar for stability
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Stability')
    
    # Success rate summary
    ax = axes[1, 1]
    successful = sum(1 for r in results.values() if r.success)
    failed = len(results) - successful
    
    wedges, texts, autotexts = ax.pie([successful, failed], 
                                     labels=['Successful', 'Failed'],
                                     colors=['green', 'red'],
                                     autopct='%1.1f%%',
                                     startangle=90)
    ax.set_title('Success Rate')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = os.path.join(output_dir, 'warp_bubble_comparison.png')
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    
    print(f"Visualization saved: {viz_path}")
    return viz_path

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='3D Warp Bubble Validation')
    parser.add_argument('--source', choices=['ghost', 'metamaterial', 'both'],
                       default='both', help='Energy source to test')
    parser.add_argument('--radius', type=float, default=10.0,
                       help='Simulation radius (m)')
    parser.add_argument('--resolution', type=int, default=30,
                       help='Mesh resolution')
    parser.add_argument('--output-dir', default='results',
                       help='Output directory')
    parser.add_argument('--sweep', action='store_true',
                       help='Run parameter sweep')
    parser.add_argument('--convergence', action='store_true',
                       help='Run mesh convergence study')
    
    args = parser.parse_args()
    
    print("3D Mesh-Based Warp Bubble Validation")
    print("=" * 50)
    print(f"Domain radius: {args.radius} m")
    print(f"Mesh resolution: {args.resolution}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    results = {}
    
    if args.convergence:
        print("Running mesh convergence analysis...")
        conv_results = analyze_mesh_convergence('ghost')
        # Save convergence results
        conv_path = os.path.join(args.output_dir, 'convergence_analysis.json')
        conv_data = {
            str(res): {
                'total_energy': result.energy_total,
                'stability': result.stability,
                'execution_time': result.execution_time,
                'mesh_nodes': result.mesh_nodes
            }
            for res, result in conv_results.items()
        }
        with open(conv_path, 'w') as f:
            json.dump(conv_data, f, indent=2)
        print(f"Convergence analysis saved: {conv_path}")
        return
    
    if args.sweep:
        print("Running parameter sweeps...")
        
        if args.source in ['ghost', 'both']:
            ghost_ranges = {
                'M': [500, 1000, 1500],
                'alpha': [0.005, 0.01, 0.02],
                'beta': [0.05, 0.1, 0.15]
            }
            ghost_sweep = run_parameter_sweep('ghost', ghost_ranges, 
                                            args.radius, args.resolution)
            results.update(ghost_sweep)
        
        if args.source in ['metamaterial', 'both']:
            meta_ranges = {
                'epsilon': [-3.0, -2.0, -1.5],
                'mu': [-2.5, -1.5, -1.0],
                'n_layers': [50, 100, 150]
            }
            meta_sweep = run_parameter_sweep('metamaterial', meta_ranges,
                                           args.radius, args.resolution)
            results.update(meta_sweep)
            
    else:
        # Standard comparison
        sources = []
        
        if args.source in ['ghost', 'both']:
            sources.append(create_discovery_21_source())
        
        if args.source in ['metamaterial', 'both']:
            sources.append(create_metamaterial_source())
        
        print("Running standard comparison...")
        results = compare_energy_sources(sources, args.radius, args.resolution)
    
    if results:
        # Generate reports and visualizations
        report_path = generate_comparison_report(results, args.output_dir)
        viz_path = create_visualization_comparison(results, args.output_dir)
        
        print("\nValidation Complete!")
        print(f"Report: {report_path}")
        print(f"Visualization: {viz_path}")
        
        # Print best result
        best_source = max(results.keys(), 
                         key=lambda k: results[k].stability if results[k].success else -1)
        if results[best_source].success:
            print(f"\nBest performing source: {best_source}")
            print(f"  Total Energy: {results[best_source].energy_total:.2e} J")
            print(f"  Stability: {results[best_source].stability:.3f}")
            print(f"  Max Negative Density: {results[best_source].max_negative_density:.2e} J/m³")
        else:
            print("\nNo successful configurations found.")

if __name__ == "__main__":
    main()
