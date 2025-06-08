#!/usr/bin/env python3
"""
Simplified Vacuum Configuration Analysis

A streamlined version that works with the current vacuum engineering API.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
import json
from datetime import datetime

from vacuum_engineering import (
    CasimirArray, DynamicCasimirEffect, SqueezedVacuumResonator,
    vacuum_energy_to_anec_flux_compat as vacuum_energy_to_anec_flux,
    build_lab_sources_legacy
)

def analyze_casimir_configurations():
    """Analyze Casimir array configurations."""
    print("Analyzing Casimir array configurations...")
    
    # Test different configurations
    spacings = [5e-9, 10e-9, 20e-9, 50e-9]  # nm
    layer_counts = [50, 100, 200]
    
    results = []
    
    for spacing in spacings:
        for n_layers in layer_counts:
            spacing_array = [spacing] * n_layers
            eps_array = [1.0 + 0.001j] * n_layers
            
            casimir = CasimirArray(temperature=300.0)
            pressure = casimir.stack_pressure(n_layers, spacing_array, eps_array)
            
            # Convert pressure to energy density
            energy_density = pressure * spacing
            anec_flux = vacuum_energy_to_anec_flux(energy_density)
            
            results.append({
                'spacing_nm': spacing * 1e9,
                'n_layers': n_layers,
                'pressure_Pa': pressure,
                'energy_density': energy_density,
                'anec_flux': anec_flux
            })
    
    return results

def analyze_squeezed_vacuum():
    """Analyze squeezed vacuum configurations."""
    print("Analyzing squeezed vacuum configurations...")
    
    squeeze_params = np.linspace(0.5, 3.0, 10)  # Squeezing parameters
    frequency = 1e12  # 1 THz
    volume = 1e-6  # 1 cubic mm
    
    results = []
    
    for xi in squeeze_params:
        squeezed = SqueezedVacuumResonator(resonator_frequency=frequency, 
                                         squeezing_parameter=xi)
        energy_density = squeezed.squeezed_energy_density(volume)
        anec_flux = vacuum_energy_to_anec_flux(energy_density)
        
        results.append({
            'squeezing_parameter': xi,
            'frequency_Hz': frequency,
            'energy_density': energy_density,
            'anec_flux': anec_flux
        })
    
    return results

def analyze_dynamic_casimir():
    """Analyze dynamic Casimir configurations."""
    print("Analyzing dynamic Casimir configurations...")
    
    frequencies = [1e9, 5e9, 10e9, 50e9]  # GHz range
    modulation_depths = [0.01, 0.05, 0.1, 0.2]
    
    results = []
    
    # Use legacy API for compatibility
    legacy_sources = build_lab_sources_legacy()
    dynamic = legacy_sources["DynamicCasimir"]
    
    for freq in frequencies:
        for mod_depth in modulation_depths:
            energy_density = dynamic.total_density()
            anec_flux = vacuum_energy_to_anec_flux(energy_density)
            
            results.append({
                'frequency_Hz': freq,
                'modulation_depth': mod_depth,
                'energy_density': energy_density,
                'anec_flux': anec_flux
            })
    
    return results

def create_visualizations(casimir_results, squeezed_results, dynamic_results):
    """Create visualizations of the analysis results."""
    print("Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Casimir array analysis
    ax = axes[0, 0]
    spacings = [r['spacing_nm'] for r in casimir_results]
    fluxes = [abs(r['anec_flux']) for r in casimir_results]
    layers = [r['n_layers'] for r in casimir_results]
    
    scatter = ax.scatter(spacings, fluxes, c=layers, cmap='viridis', s=60, alpha=0.7)
    ax.set_xlabel('Spacing (nm)')
    ax.set_ylabel('|ANEC Flux|')
    ax.set_yscale('log')
    ax.set_title('Casimir Arrays: Spacing vs ANEC Flux')
    plt.colorbar(scatter, ax=ax, label='Layer Count')
    
    # Squeezed vacuum analysis
    ax = axes[0, 1]
    xi_values = [r['squeezing_parameter'] for r in squeezed_results]
    sq_fluxes = [abs(r['anec_flux']) for r in squeezed_results]
    
    ax.plot(xi_values, sq_fluxes, 'ro-', linewidth=2, markersize=6)
    ax.set_xlabel('Squeezing Parameter')
    ax.set_ylabel('|ANEC Flux|')
    ax.set_yscale('log')
    ax.set_title('Squeezed Vacuum: Squeezing vs ANEC Flux')
    ax.grid(True, alpha=0.3)
    
    # Dynamic Casimir analysis
    ax = axes[1, 0]
    frequencies = [r['frequency_Hz'] for r in dynamic_results]
    dyn_fluxes = [abs(r['anec_flux']) for r in dynamic_results]
    mod_depths = [r['modulation_depth'] for r in dynamic_results]
    
    scatter = ax.scatter(frequencies, dyn_fluxes, c=mod_depths, cmap='plasma', s=60, alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|ANEC Flux|')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_title('Dynamic Casimir: Frequency vs ANEC Flux')
    plt.colorbar(scatter, ax=ax, label='Modulation Depth')
    
    # Comparative analysis
    ax = axes[1, 1]
    max_casimir = max([abs(r['anec_flux']) for r in casimir_results])
    max_squeezed = max([abs(r['anec_flux']) for r in squeezed_results])
    max_dynamic = max([abs(r['anec_flux']) for r in dynamic_results])
    
    sources = ['Casimir\nArrays', 'Squeezed\nVacuum', 'Dynamic\nCasimir']
    max_fluxes = [max_casimir, max_squeezed, max_dynamic]
    
    bars = ax.bar(sources, max_fluxes, alpha=0.7, color=['skyblue', 'lightcoral', 'gold'])
    ax.set_ylabel('Maximum |ANEC Flux|')
    ax.set_yscale('log')
    ax.set_title('Peak Performance Comparison')
    
    # Add value labels on bars
    for bar, flux in zip(bars, max_fluxes):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{flux:.2e}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('results/vacuum_configuration_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Visualization saved to results/vacuum_configuration_analysis.png")

def generate_report(casimir_results, squeezed_results, dynamic_results):
    """Generate comprehensive JSON report."""
    print("Generating comprehensive report...")
    
    # Find best performers
    best_casimir = max(casimir_results, key=lambda x: abs(x['anec_flux']))
    best_squeezed = max(squeezed_results, key=lambda x: abs(x['anec_flux']))
    best_dynamic = max(dynamic_results, key=lambda x: abs(x['anec_flux']))
    
    report = {
        'analysis_timestamp': datetime.now().isoformat(),
        'summary': {
            'total_configurations_tested': len(casimir_results) + len(squeezed_results) + len(dynamic_results),
            'best_performers': {
                'casimir_array': {
                    'spacing_nm': best_casimir['spacing_nm'],
                    'n_layers': best_casimir['n_layers'],
                    'anec_flux': best_casimir['anec_flux'],
                    'energy_density': best_casimir['energy_density']
                },
                'squeezed_vacuum': {
                    'squeezing_parameter': best_squeezed['squeezing_parameter'],
                    'frequency_Hz': best_squeezed['frequency_Hz'],
                    'anec_flux': best_squeezed['anec_flux'],
                    'energy_density': best_squeezed['energy_density']
                },
                'dynamic_casimir': {
                    'frequency_Hz': best_dynamic['frequency_Hz'],
                    'modulation_depth': best_dynamic['modulation_depth'],
                    'anec_flux': best_dynamic['anec_flux'],
                    'energy_density': best_dynamic['energy_density']
                }
            }
        },
        'detailed_results': {
            'casimir_arrays': casimir_results,
            'squeezed_vacuum': squeezed_results,
            'dynamic_casimir': dynamic_results
        },
        'recommendations': [
            f"Casimir arrays show highest potential with flux magnitude {abs(best_casimir['anec_flux']):.2e}",
            f"Optimal Casimir configuration: {best_casimir['spacing_nm']:.1f} nm spacing, {best_casimir['n_layers']} layers",
            f"Squeezed vacuum optimal at squeezing parameter {best_squeezed['squeezing_parameter']:.2f}",
            "Focus development on multi-layer Casimir arrays for maximum performance",
            "Consider hybrid approaches combining multiple vacuum engineering techniques"
        ]
    }
    
    # Convert numpy types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    report = convert_numpy(report)
    
    # Save report
    os.makedirs('results', exist_ok=True)
    with open('results/vacuum_configuration_analysis_simplified.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Report saved to results/vacuum_configuration_analysis_simplified.json")
    return report

def main():
    """Main analysis routine."""
    print("=" * 60)
    print("SIMPLIFIED VACUUM CONFIGURATION ANALYSIS")
    print("=" * 60)
    
    # Run analyses
    casimir_results = analyze_casimir_configurations()
    squeezed_results = analyze_squeezed_vacuum()
    dynamic_results = analyze_dynamic_casimir()
    
    # Create visualizations
    create_visualizations(casimir_results, squeezed_results, dynamic_results)
    
    # Generate report
    report = generate_report(casimir_results, squeezed_results, dynamic_results)
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    print(f"Configurations tested: {report['summary']['total_configurations_tested']}")
    print("\nBest performers:")
    
    for source, config in report['summary']['best_performers'].items():
        print(f"\n{source.upper().replace('_', ' ')}:")
        print(f"  ANEC flux: {abs(config['anec_flux']):.2e}")
        print(f"  Energy density: {config['energy_density']:.2e} J/m³")
    
    print("\nKey recommendations:")
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"  {i}. {rec}")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check results/ directory for outputs.")
    print("=" * 60)

if __name__ == "__main__":
    main()
