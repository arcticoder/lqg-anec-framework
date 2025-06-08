#!/usr/bin/env python3
"""
Vacuum Engineering Driver Script

Comprehensive driver for testing realistic materials and layer configurations
in Casimir arrays, dynamic Casimir circuits, and squeezed vacuum systems.

This script:
1. Scans parameter spaces for optimal configurations
2. Compares different vacuum engineering approaches  
3. Evaluates feasibility for target ANEC violation fluxes
4. Generates plots and analysis reports

Author: LQG-ANEC Framework - Vacuum Engineering Team
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import sys
import os
from pathlib import Path

# Add src to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vacuum_engineering import (
    CasimirArray, DynamicCasimirEffect, SqueezedVacuumResonator,
    MetamaterialCasimir, comprehensive_vacuum_analysis,
    vacuum_energy_to_anec_flux, MATERIAL_DATABASE
)

def scan_casimir_materials():
    """
    Scan realistic materials for optimal Casimir configurations.
    """
    print("Scanning Casimir Materials...")
    print("=" * 40)
    
    # Initialize Casimir system at cryogenic temperature  
    casimir = CasimirArray(temperature=4.0)
    
    # Test materials: common lab materials + metamaterials
    materials = ['SiO2', 'Au', 'Si', 'Al', 'metamaterial']
    layer_counts = [1, 5, 10, 20, 50]
    spacing_range = np.logspace(-8, -5, 20)  # 10 nm to 10 μm
    
    results = {}
    
    for material in materials:
        mat_props = MATERIAL_DATABASE[material]
        perm = mat_props['permittivity']
        mu = mat_props['permeability']
        
        material_results = []
        
        for n_layers in layer_counts:
            layer_data = []
            
            for spacing in spacing_range:
                # Single spacing for all layers (simplified)
                spacing_list = [spacing] * n_layers
                perm_list = [perm] * n_layers
                mu_list = [mu] * n_layers
                
                try:
                    pressure = casimir.stack_pressure(n_layers, spacing_list, perm_list, mu_list)
                    
                    # Convert to energy density
                    typical_area = (1e-3)**2  # 1 mm²
                    volume = typical_area * (spacing * n_layers)
                    energy_density = pressure * spacing / volume
                    
                    layer_data.append({
                        'spacing': spacing,
                        'pressure': pressure,
                        'energy_density': energy_density,
                        'volume': volume
                    })
                    
                except Exception as e:
                    continue
            
            if layer_data:
                material_results.append({
                    'n_layers': n_layers,
                    'data': layer_data,
                    'max_pressure': min([d['pressure'] for d in layer_data]),
                    'optimal_spacing': min(layer_data, key=lambda x: x['pressure'])['spacing']
                })
        
        results[material] = material_results
        
        # Print summary for this material
        if material_results:
            best_config = min(material_results, key=lambda x: x['max_pressure'])
            print(f"\n{material}:")
            print(f"  Best pressure: {best_config['max_pressure']:.2e} Pa")
            print(f"  Optimal layers: {best_config['n_layers']}")
            print(f"  Optimal spacing: {best_config['optimal_spacing']:.2e} m")
    
    return results

def optimize_metamaterial_array():
    """
    Optimize metamaterial properties for maximum Casimir enhancement.
    """
    print("\nOptimizing Metamaterial Arrays...")
    print("=" * 40)
    
    metamaterial = MetamaterialCasimir(unit_cell_size=50e-9)
    
    # Target enhancements to test
    target_enhancements = [-10, -5, -2, 2, 5, 10]  # Negative = repulsive
    
    optimization_results = []
    
    for target in target_enhancements:
        print(f"\nTargeting enhancement factor: {target}")
        
        optimal = metamaterial.design_optimal_metamaterial(target)
        
        print(f"  Epsilon: {optimal['epsilon']:.2f}")
        print(f"  Mu: {optimal['mu']:.2f}")
        print(f"  Achieved enhancement: {optimal['enhancement']:.2f}")
        print(f"  Feasible: {optimal['feasible']}")
        
        optimization_results.append({
            'target': target,
            'result': optimal
        })
    
    return optimization_results

def dynamic_casimir_parameter_sweep():
    """
    Parameter sweep for dynamic Casimir effect optimization.
    """
    print("\nDynamic Casimir Parameter Sweep...")
    print("=" * 40)
    
    # Test parameters
    circuit_frequencies = np.logspace(9, 12, 20)  # 1 GHz to 1 THz
    drive_amplitudes = np.linspace(0.01, 0.5, 20)
    quality_factors = [100, 1000, 10000, 100000]
    
    best_configs = []
    
    for Q in quality_factors:
        print(f"\nQuality Factor: {Q}")
        
        best_energy_density = 0.0
        best_config = None
        
        for f_circuit in circuit_frequencies:
            dynamic = DynamicCasimirEffect(circuit_frequency=f_circuit, drive_amplitude=0.1)
            
            for amp in drive_amplitudes:
                dynamic.drive_amp = amp
                
                # Optimal drive frequency is typically 2×circuit frequency
                f_drive = 2 * f_circuit
                circuit_volume = (1e-3)**3  # 1 mm³
                
                energy_density = dynamic.negative_energy_density(f_drive, circuit_volume, Q)
                
                if abs(energy_density) > abs(best_energy_density):
                    best_energy_density = energy_density
                    best_config = {
                        'circuit_frequency': f_circuit,
                        'drive_frequency': f_drive,
                        'drive_amplitude': amp,
                        'quality_factor': Q,
                        'energy_density': energy_density
                    }
        
        if best_config:
            print(f"  Best energy density: {best_config['energy_density']:.2e} J/m³")
            print(f"  Circuit frequency: {best_config['circuit_frequency']:.2e} Hz")
            print(f"  Drive amplitude: {best_config['drive_amplitude']:.3f}")
            
            best_configs.append(best_config)
    
    return best_configs

def squeezed_vacuum_optimization():
    """
    Optimize squeezed vacuum parameters for maximum negative energy.
    """
    print("\nSqueezed Vacuum Optimization...")
    print("=" * 40)
    
    # Parameter ranges
    frequencies = np.logspace(12, 15, 20)  # THz to PHz range
    squeezing_params = np.linspace(0.1, 5.0, 20)
    
    optimization_results = []
    
    for freq in frequencies:
        best_energy = 0.0
        best_xi = 0.0
        
        for xi in squeezing_params:
            squeezed = SqueezedVacuumResonator(resonator_frequency=freq, squeezing_parameter=xi)
            
            # Test volume (optical fiber mode volume)
            volume = np.pi * (10e-6)**2 * 1e-3  # 10 μm radius, 1 mm length
            
            energy_density = squeezed.squeezed_energy_density(volume)
            stabilization_power = squeezed.stabilization_power()
            
            # Optimization criterion: maximize |negative energy| while keeping power reasonable
            if energy_density < 0 and stabilization_power < 1e-2:  # < 10 mW
                if abs(energy_density) > abs(best_energy):
                    best_energy = energy_density
                    best_xi = xi
        
        if best_energy < 0:
            optimization_results.append({
                'frequency': freq,
                'squeezing_parameter': best_xi,
                'energy_density': best_energy,
                'wavelength': 3e8 / freq
            })
            
            print(f"Frequency: {freq:.2e} Hz (λ = {3e8/freq:.2e} m)")
            print(f"  Best squeezing: {best_xi:.2f}")
            print(f"  Energy density: {best_energy:.2e} J/m³")
    
    return optimization_results

def compare_all_approaches():
    """
    Comprehensive comparison of all vacuum engineering approaches.
    """
    print("\nComprehensive Approach Comparison...")
    print("=" * 50)
    
    # Run full analysis
    analysis = comprehensive_vacuum_analysis(target_flux=1e-25)
    
    # Extract key metrics
    approaches = list(analysis.keys())
    energy_densities = [analysis[approach]['energy_density'] for approach in approaches]
    volumes = [analysis[approach]['volume'] for approach in approaches]
    anec_fluxes = [analysis[approach]['anec_flux'] for approach in approaches]
    target_ratios = [analysis[approach]['target_ratio'] for approach in approaches]
    
    # Create comparison plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Energy densities
    ax1.bar(approaches, np.abs(energy_densities))
    ax1.set_yscale('log')
    ax1.set_ylabel('|Energy Density| (J/m³)')
    ax1.set_title('Negative Energy Densities')
    ax1.tick_params(axis='x', rotation=45)
    
    # Volumes
    ax2.bar(approaches, volumes)
    ax2.set_yscale('log') 
    ax2.set_ylabel('Volume (m³)')
    ax2.set_title('System Volumes')
    ax2.tick_params(axis='x', rotation=45)
    
    # ANEC fluxes
    ax3.bar(approaches, np.abs(anec_fluxes))
    ax3.set_yscale('log')
    ax3.set_ylabel('|ANEC Flux| (W)')
    ax3.set_title('ANEC Violation Fluxes')
    ax3.axhline(y=1e-25, color='r', linestyle='--', label='Target')
    ax3.legend()
    ax3.tick_params(axis='x', rotation=45)
    
    # Target ratios
    ax4.bar(approaches, target_ratios)
    ax4.set_yscale('log')
    ax4.set_ylabel('Target Ratio')
    ax4.set_title('Target Achievement Ratio')
    ax4.axhline(y=1.0, color='r', linestyle='--', label='Target')
    ax4.legend()
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "vacuum_engineering_comparison.png", dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_dir / 'vacuum_engineering_comparison.png'}")
    
    # Print detailed results
    print("\nDetailed Results:")
    print("-" * 30)
    
    for approach in approaches:
        data = analysis[approach]
        print(f"\n{approach.replace('_', ' ').title()}:")
        print(f"  Energy density: {data['energy_density']:.2e} J/m³")
        print(f"  Volume: {data['volume']:.2e} m³")
        print(f"  ANEC flux: {data['anec_flux']:.2e} W")
        print(f"  Target ratio: {data['target_ratio']:.2e}")
        print(f"  Feasible: {data['feasible']}")
        
        if 'configuration' in data:
            print(f"  Configuration: {data['configuration']}")
    
    return analysis

def generate_analysis_report():
    """
    Generate comprehensive analysis report with all results.
    """
    print("\nGenerating Analysis Report...")
    print("=" * 40)
    
    # Run all analyses
    casimir_scan = scan_casimir_materials()
    metamaterial_opt = optimize_metamaterial_array()
    dynamic_sweep = dynamic_casimir_parameter_sweep()
    squeezed_opt = squeezed_vacuum_optimization()
    comparison = compare_all_approaches()
    
    # Compile results
    report = {
        'casimir_materials': casimir_scan,
        'metamaterial_optimization': metamaterial_opt,
        'dynamic_casimir_sweep': dynamic_sweep,
        'squeezed_vacuum_optimization': squeezed_opt,
        'comprehensive_comparison': comparison,
        'summary': {
            'best_approach': max(comparison.keys(), key=lambda k: comparison[k]['target_ratio']),
            'target_flux': 1e-25,
            'analysis_date': '2025-06-07'
        }
    }
    
    # Save report
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "vacuum_engineering_report.json", 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating, np.complexfloating)):
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            else:
                return obj
        
        json.dump(convert_numpy(report), f, indent=2)
    
    print(f"Analysis report saved to: {output_dir / 'vacuum_engineering_report.json'}")
    
    # Print executive summary
    print("\nExecutive Summary:")
    print("-" * 20)
    best_approach = report['summary']['best_approach']
    best_ratio = comparison[best_approach]['target_ratio']
    
    print(f"Best approach: {best_approach.replace('_', ' ').title()}")
    print(f"Target achievement: {best_ratio:.2e} × target flux")
    print(f"Gap to target: {1/best_ratio:.2e} × improvement needed")
    
    if best_ratio > 0.1:
        print("✓ Promising approach - within order of magnitude of target")
    elif best_ratio > 0.01:
        print("⚠ Moderate potential - requires significant optimization")
    else:
        print("✗ Limited potential - fundamental improvements needed")
    
    return report

if __name__ == "__main__":
    print("Vacuum Engineering Driver Script")
    print("=" * 50)
    print("Testing realistic materials and configurations...")
    
    # Generate comprehensive analysis
    try:
        report = generate_analysis_report()
        print("\n✓ Analysis completed successfully!")
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
