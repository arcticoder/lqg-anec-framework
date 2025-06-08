#!/usr/bin/env python3
"""
Example Script: Programmatic 3D Validation and Best Source Analysis

This script demonstrates how to programmatically invoke the 3D mesh validation
and automatically identify the best performing energy source configuration.
"""

import sys
import json
import os
from pathlib import Path

def run_validation_and_analyze():
    """
    Run 3D validation on both sources and analyze results to find the best configuration.
    """
    print("=" * 60)
    print("Programmatic 3D Validation and Analysis")
    print("=" * 60)
    
    # Import the validation module
    try:
        from run_3d_mesh_validation import main as validate
    except ImportError:
        print("Error: Could not import run_3d_mesh_validation module")
        return None
    
    # Set up arguments for both source validation
    original_argv = sys.argv.copy()
    sys.argv = [
        'run_3d_mesh_validation.py', 
        '--source', 'both', 
        '--radius', '10', 
        '--resolution', '30'
    ]
    
    print("Running 3D validation on both Ghost EFT and Metamaterial Casimir sources...")
    print(f"Command: {' '.join(sys.argv)}")
    print()
    
    try:
        # Run the validation
        validate()
        
        # Restore original argv
        sys.argv = original_argv
        
        # Load and analyze the results
        results_path = 'results/warp_bubble_comparison_report.json'
        if not os.path.exists(results_path):
            print(f"Error: Results file not found at {results_path}")
            return None
            
        print("Loading analysis results...")
        with open(results_path, 'r') as f:
            report = json.load(f)
        
        # Find the best source based on success and stability
        print("\nAnalyzing performance metrics...")
        best_source = None
        best_score = -1
        
        results_summary = []
        
        for source_name, result in report['results'].items():
            # Calculate composite score: success (binary) + stability (0-1) + energy efficiency
            success_score = 1.0 if result['success'] else 0.0
            stability_score = result['stability']
            # Energy efficiency: prefer lower absolute energy (more efficient)
            energy_efficiency = 1.0 / (1.0 + abs(result['total_energy_J']) * 1e12)  # Scale for comparison
            
            composite_score = success_score + stability_score + energy_efficiency
            
            results_summary.append({
                'source': source_name,
                'success': result['success'],
                'total_energy_J': result['total_energy_J'],
                'stability': result['stability'],
                'max_negative_density': result['max_negative_density_J_per_m3'],
                'execution_time_s': result['execution_time_s'],
                'mesh_nodes': result['mesh_nodes'],
                'composite_score': composite_score,
                'parameters': result.get('parameters', {})
            })
            
            if composite_score > best_score:
                best_score = composite_score
                best_source = source_name
        
        # Sort results by composite score
        results_summary.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Display results
        print(f"\nTested {len(results_summary)} source configurations")
        print(f"All sources successful: {report['successful_sources'] == report['sources_tested']}")
        print()
        
        print("Performance Summary (ranked by composite score):")
        print("-" * 80)
        for i, result in enumerate(results_summary[:5]):  # Show top 5
            print(f"{i+1}. {result['source']}")
            print(f"   Success: {result['success']}")
            print(f"   Total Energy: {result['total_energy_J']:.2e} J")
            print(f"   Stability: {result['stability']:.3f}")
            print(f"   Max Negative Density: {result['max_negative_density']:.2e} J/m³")
            print(f"   Execution Time: {result['execution_time_s']:.3f} s")
            print(f"   Composite Score: {result['composite_score']:.3f}")
            if result['parameters']:
                print(f"   Parameters: {result['parameters']}")
            print()
        
        # Highlight the best source
        best_result = next(r for r in results_summary if r['source'] == best_source)
        
        print("=" * 60)
        print("OPTIMAL SOURCE IDENTIFICATION")
        print("=" * 60)
        print(f"Best performing source: {best_source}")
        print(f"Composite score: {best_score:.3f}")
        print()
        print("Key metrics:")
        print(f"  • Total Energy: {best_result['total_energy_J']:.2e} J")
        print(f"  • Stability: {best_result['stability']:.3f} ({best_result['stability']*100:.1f}%)")
        print(f"  • Max Negative Density: {best_result['max_negative_density']:.2e} J/m³")
        print(f"  • Execution Time: {best_result['execution_time_s']:.3f} s")
        print(f"  • Mesh Nodes: {best_result['mesh_nodes']:,}")
        
        if best_result['parameters']:
            print("\nOptimal parameters:")
            for param, value in best_result['parameters'].items():
                print(f"  • {param}: {value}")
        
        print()
        print("Recommendation:")
        if 'ghost' in best_source.lower():
            print("→ Ghost/Phantom EFT is the optimal choice for warp bubble applications")
            print("→ Proceed with Ghost EFT parameter optimization for experimental design")
        else:
            print("→ Alternative energy source shows optimal performance")
            print("→ Consider this configuration for further development")
        
        return best_result
        
    except Exception as e:
        print(f"Error during validation: {e}")
        sys.argv = original_argv
        return None

def demonstrate_parameter_optimization():
    """
    Demonstrate parameter space exploration for the optimal source.
    """
    print("\n" + "=" * 60)
    print("PARAMETER OPTIMIZATION DEMONSTRATION")
    print("=" * 60)
    
    # Import the validation module
    try:
        from run_3d_mesh_validation import main as validate
    except ImportError:
        print("Error: Could not import run_3d_mesh_validation module")
        return
    
    # Run parameter sweep for Ghost EFT
    original_argv = sys.argv.copy()
    sys.argv = [
        'run_3d_mesh_validation.py', 
        '--source', 'ghost', 
        '--sweep'
    ]
    
    print("Running parameter sweep for Ghost EFT...")
    print(f"Command: {' '.join(sys.argv)}")
    
    try:
        validate()
        sys.argv = original_argv
        
        # Load and analyze sweep results
        results_path = 'results/warp_bubble_comparison_report.json'
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                sweep_report = json.load(f)
            
            # Find the best configuration from the sweep
            best_config = None
            best_stability = 0
            
            for config_name, result in sweep_report['results'].items():
                if result['success'] and result['stability'] > best_stability:
                    best_stability = result['stability']
                    best_config = (config_name, result)
            
            if best_config:
                config_name, config_result = best_config
                print(f"\nOptimal configuration from sweep: {config_name}")
                print(f"Stability: {config_result['stability']:.4f}")
                print(f"Total Energy: {config_result['total_energy_J']:.2e} J")
                print("Parameters:")
                for param, value in config_result['parameters'].items():
                    print(f"  • {param}: {value}")
        
    except Exception as e:
        print(f"Error during parameter sweep: {e}")
        sys.argv = original_argv

if __name__ == "__main__":
    # Run the complete analysis pipeline
    print("Starting automated 3D validation and analysis pipeline...")
    
    # Step 1: Validate both sources and find the best
    best_result = run_validation_and_analyze()
    
    if best_result:
        # Step 2: Demonstrate parameter optimization
        demonstrate_parameter_optimization()
        
        print("\n" + "=" * 60)
        print("ANALYSIS COMPLETE")
        print("=" * 60)
        print("Results available in: results/warp_bubble_comparison_report.json")
        print("Visualizations saved in: results/warp_bubble_comparison.png")
        print("\nNext steps:")
        print("1. Use optimal parameters for experimental design")
        print("2. Run mesh convergence studies for precision requirements")
        print("3. Proceed with metric-ansatz optimizations")
    else:
        print("Analysis failed. Please check the validation pipeline.")
