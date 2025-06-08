#!/usr/bin/env python3
"""
Vacuum-ANEC Integration Script

This script runs each laboratory vacuum source through the ANEC integrator
with quantum inequality (QI) smearing to assess experimental feasibility
for ANEC violation.

Features:
- Connects vacuum engineering sources to ANEC analysis framework
- Applies QI smearing kernels to negative energy distributions
- Generates comprehensive feasibility reports
- Identifies optimal configurations for laboratory implementation

Usage:
    python scripts/vacuum_anec_integration.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.constants import hbar, c, pi
import json
from datetime import datetime

from vacuum_engineering import (
    build_lab_sources, vacuum_energy_to_anec_flux, 
    MATERIAL_DATABASE, comprehensive_vacuum_analysis
)

class VacuumANECIntegrator:
    """
    Integrates vacuum engineering sources with ANEC violation analysis.
    
    Provides QI smearing, temporal integration, and feasibility assessment
    for laboratory negative energy sources.
    """
    
    def __init__(self, temporal_scale: float = 1e-6):
        """
        Initialize ANEC integrator.
        
        Args:
            temporal_scale: Characteristic temporal scale for QI smearing (seconds)
        """
        self.tau = temporal_scale
        self.results = {}
        
    def qi_smearing_kernel(self, t: float, kernel_type: str = 'gaussian') -> float:
        """
        Quantum inequality smearing kernel.
        
        Args:
            t: Time coordinate
            kernel_type: Type of smearing ('gaussian', 'exponential', 'lorentzian')
            
        Returns:
            Smearing kernel value
        """
        if kernel_type == 'gaussian':
            return np.exp(-t**2 / (2*self.tau**2)) / np.sqrt(2*pi*self.tau**2)
        elif kernel_type == 'exponential':
            return np.exp(-np.abs(t) / self.tau) / (2*self.tau)
        elif kernel_type == 'lorentzian':
            return (self.tau / pi) / (t**2 + self.tau**2)
        else:
            raise ValueError(f"Unknown kernel type: {kernel_type}")
    
    def compute_anec_violation(self, energy_density: float, volume: float, 
                              duration: float, kernel_type: str = 'gaussian') -> dict:
        """
        Compute ANEC violation from vacuum source parameters.
        
        Args:
            energy_density: Negative energy density (J/m³)
            volume: Source volume (m³)
            duration: Source operation duration (s)
            kernel_type: QI smearing kernel type
            
        Returns:
            Dictionary with ANEC violation analysis
        """
        total_energy = energy_density * volume
        
        # Temporal integration with QI smearing
        def integrand(t):
            # Source profile (assume uniform over duration)
            if abs(t) <= duration / 2:
                source_profile = 1.0
            else:
                source_profile = 0.0
                
            return self.qi_smearing_kernel(t, kernel_type) * source_profile * total_energy
        
        # Integrate over extended time window
        integration_window = max(duration, 5*self.tau)
        flux, integration_error = quad(integrand, -integration_window, integration_window)
        
        # Normalize by characteristic time
        anec_flux = flux / self.tau
        
        return {
            'anec_flux': anec_flux,
            'total_energy': total_energy,
            'integration_error': integration_error,
            'kernel_type': kernel_type,
            'temporal_scale': self.tau,
            'violation_strength': abs(anec_flux) if anec_flux < 0 else 0.0
        }
    
    def analyze_casimir_source(self, source_config: dict) -> dict:
        """
        Analyze Casimir array source for ANEC violation.
        
        Args:
            source_config: Configuration dictionary from build_lab_sources
            
        Returns:
            ANEC analysis results
        """
        casimir_source = source_config['source']
        params = source_config['params']
        
        # Test optimal configuration
        spacing = params['optimal_spacing']
        material = 'SiO2'  # High-performance dielectric
        
        # Compute Casimir pressure and energy density
        pressure = casimir_source.casimir_pressure(
            spacing, 
            MATERIAL_DATABASE[material]['permittivity']
        )
        
        # Multi-layer enhancement
        n_layers = params['n_layers']
        total_pressure = pressure * n_layers
        
        # Convert to energy density
        energy_density = total_pressure * spacing / params['volume']
        
        # Assume quasi-static operation (long duration)
        duration = 1e-3  # 1 ms operation time
        
        anec_result = self.compute_anec_violation(energy_density, params['volume'], duration)
        
        # Add Casimir-specific metrics
        anec_result.update({
            'source_type': 'casimir_array',
            'casimir_pressure': pressure,
            'total_pressure': total_pressure,
            'layer_count': n_layers,
            'plate_spacing': spacing,
            'material': material,
            'enhancement_factor': n_layers
        })
        
        return anec_result
    
    def analyze_dynamic_casimir_source(self, source_config: dict) -> dict:
        """
        Analyze dynamic Casimir source for ANEC violation.
        
        Args:
            source_config: Configuration dictionary from build_lab_sources
            
        Returns:
            ANEC analysis results
        """
        dynamic_source = source_config['source']
        params = source_config['params']
        
        # Compute negative energy density
        energy_density = dynamic_source.negative_energy_density(
            params['drive_frequency'],
            params['volume'],
            params['quality_factor']
        )
        
        # Dynamic Casimir is inherently time-dependent
        # Duration limited by drive pulse length
        duration = 1.0 / params['drive_frequency']  # Single drive cycle
        
        anec_result = self.compute_anec_violation(energy_density, params['volume'], duration)
        
        # Add dynamic Casimir-specific metrics
        photon_rate = dynamic_source.photon_creation_rate(
            params['drive_frequency'], 
            params['quality_factor']
        )
        
        anec_result.update({
            'source_type': 'dynamic_casimir',
            'drive_frequency': params['drive_frequency'],
            'quality_factor': params['quality_factor'],
            'photon_creation_rate': photon_rate,
            'drive_cycle_duration': duration
        })
        
        return anec_result
    
    def analyze_squeezed_vacuum_source(self, source_config: dict) -> dict:
        """
        Analyze squeezed vacuum source for ANEC violation.
        
        Args:
            source_config: Configuration dictionary from build_lab_sources
            
        Returns:
            ANEC analysis results
        """
        squeezed_source = source_config['source']
        params = source_config['params']
        
        # Compute squeezed vacuum energy density
        energy_density = squeezed_source.squeezed_energy_density(params['volume'])
        
        # Squeezed states can be maintained continuously with active stabilization
        duration = 1e-3  # 1 ms coherence time
        
        anec_result = self.compute_anec_violation(energy_density, params['volume'], duration)
        
        # Add squeezed vacuum-specific metrics
        stabilization_power = squeezed_source.stabilization_power()
        
        anec_result.update({
            'source_type': 'squeezed_vacuum',
            'squeezing_parameter': squeezed_source.xi,
            'resonator_frequency': squeezed_source.omega_res / (2*pi),
            'stabilization_power': stabilization_power,
            'coherence_time': duration
        })
        
        return anec_result
    
    def run_comprehensive_analysis(self, target_anec_flux: float = 1e-25) -> dict:
        """
        Run comprehensive ANEC integration analysis for all vacuum sources.
        
        Args:
            target_anec_flux: Target ANEC violation flux for comparison (W)
            
        Returns:
            Complete analysis results
        """
        print("Running Vacuum-ANEC Integration Analysis...")
        print("=" * 50)
        
        # Build laboratory sources
        sources = build_lab_sources('comprehensive')
        
        analysis_results = {
            'timestamp': datetime.now().isoformat(),
            'target_anec_flux': target_anec_flux,
            'temporal_scale': self.tau,
            'sources': {}
        }
        
        # Analyze each source type
        for source_name, source_config in sources.items():
            print(f"\nAnalyzing {source_name} source...")
            
            if source_name == 'casimir':
                result = self.analyze_casimir_source(source_config)
            elif source_name == 'dynamic':
                result = self.analyze_dynamic_casimir_source(source_config)
            elif source_name == 'squeezed':
                result = self.analyze_squeezed_vacuum_source(source_config)
            else:
                continue
            
            # Compute target ratios and feasibility
            result['target_ratio'] = abs(result['anec_flux'] / target_anec_flux) if result['anec_flux'] < 0 else 0.0
            result['feasible'] = result['target_ratio'] > 0.1  # Within order of magnitude
            result['experimentally_accessible'] = (
                result['violation_strength'] > 1e-30 and  # Detectable threshold
                result['total_energy'] > -1e-18  # Femtojoule scale
            )
            
            analysis_results['sources'][source_name] = result
            
            # Print key results
            print(f"  ANEC flux: {result['anec_flux']:.2e} W")
            print(f"  Target ratio: {result['target_ratio']:.2e}")
            print(f"  Feasible: {result['feasible']}")
            print(f"  Experimentally accessible: {result['experimentally_accessible']}")
        
        # Find best performing source
        viable_sources = {k: v for k, v in analysis_results['sources'].items() 
                         if v['target_ratio'] > 0}
        
        if viable_sources:
            best_source = max(viable_sources.keys(), 
                            key=lambda k: viable_sources[k]['target_ratio'])
            analysis_results['best_source'] = best_source
            analysis_results['best_target_ratio'] = viable_sources[best_source]['target_ratio']
        else:
            analysis_results['best_source'] = None
            analysis_results['best_target_ratio'] = 0.0
        
        self.results = analysis_results
        return analysis_results
    
    def generate_report(self, output_path: str = 'results/vacuum_anec_integration_report.json'):
        """
        Generate comprehensive JSON report of vacuum-ANEC integration analysis.
        
        Args:
            output_path: Path to save the report
        """
        if not self.results:
            raise ValueError("No analysis results available. Run analysis first.")
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
          # Convert numpy types to JSON serializable
        def convert_types(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating, np.complexfloating)):
                return obj.item()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_types(v) for v in obj]
            else:
                return obj
        
        serializable_results = convert_types(self.results)
        
        with open(output_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        print(f"\nReport saved to: {output_path}")
        
        return output_path
    
    def create_visualization(self, save_path: str = 'results/vacuum_anec_comparison.png'):
        """
        Create visualization comparing vacuum sources for ANEC violation.
        
        Args:
            save_path: Path to save the plot
        """
        if not self.results:
            raise ValueError("No analysis results available. Run analysis first.")
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        sources = list(self.results['sources'].keys())
        colors = ['blue', 'red', 'green', 'orange'][:len(sources)]
        
        # 1. ANEC Flux Comparison
        anec_fluxes = [abs(self.results['sources'][s]['anec_flux']) for s in sources]
        target_flux = self.results['target_anec_flux']
        
        bars1 = ax1.bar(sources, anec_fluxes, color=colors, alpha=0.7)
        ax1.axhline(y=target_flux, color='black', linestyle='--', 
                   label=f'Target: {target_flux:.1e} W')
        ax1.set_ylabel('|ANEC Flux| (W)')
        ax1.set_title('ANEC Violation Flux Comparison')
        ax1.set_yscale('log')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, flux in zip(bars1, anec_fluxes):
            if flux > 0:
                ax1.text(bar.get_x() + bar.get_width()/2, flux,
                        f'{flux:.1e}', ha='center', va='bottom', fontsize=8)
        
        # 2. Target Ratio Comparison
        target_ratios = [self.results['sources'][s]['target_ratio'] for s in sources]
        
        bars2 = ax2.bar(sources, target_ratios, color=colors, alpha=0.7)
        ax2.axhline(y=1.0, color='black', linestyle='--', label='Target = 1.0')
        ax2.set_ylabel('Target Ratio')
        ax2.set_title('Target Achievement Ratio')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        for bar, ratio in zip(bars2, target_ratios):
            if ratio > 0:
                ax2.text(bar.get_x() + bar.get_width()/2, ratio,
                        f'{ratio:.1e}', ha='center', va='bottom', fontsize=8)
        
        # 3. Energy Density vs Volume
        energy_densities = [self.results['sources'][s]['total_energy'] / 
                           (1e-15 if s == 'dynamic' else 1e-12) for s in sources]  # Normalize by typical volumes
        volumes = [1e-15 if s == 'dynamic' else 1e-12 for s in sources]  # Typical volumes
        
        scatter = ax3.scatter(volumes, energy_densities, c=colors, s=100, alpha=0.7)
        for i, source in enumerate(sources):
            ax3.annotate(source, (volumes[i], energy_densities[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=9)
        
        ax3.set_xlabel('Volume (m³)')
        ax3.set_ylabel('Energy Density (J/m³)')
        ax3.set_title('Energy Density vs Source Volume')
        ax3.set_xscale('log')
        ax3.grid(True, alpha=0.3)
        
        # 4. Feasibility Assessment
        feasible = [self.results['sources'][s]['feasible'] for s in sources]
        accessible = [self.results['sources'][s]['experimentally_accessible'] for s in sources]
        
        x_pos = np.arange(len(sources))
        width = 0.35
        
        bars4a = ax4.bar(x_pos - width/2, feasible, width, label='Feasible', 
                        color='lightblue', alpha=0.7)
        bars4b = ax4.bar(x_pos + width/2, accessible, width, label='Experimentally Accessible',
                        color='lightgreen', alpha=0.7)
        
        ax4.set_xlabel('Source')
        ax4.set_ylabel('Assessment')
        ax4.set_title('Feasibility Assessment')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(sources)
        ax4.set_ylim(0, 1.2)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Visualization saved to: {save_path}")

def main():
    """Main execution function."""
    print("Vacuum-ANEC Integration Analysis")
    print("=" * 40)
    
    # Initialize integrator
    integrator = VacuumANECIntegrator(temporal_scale=1e-6)  # Microsecond QI scale
    
    # Run comprehensive analysis
    target_flux = 1e-25  # Target ANEC violation flux in W
    results = integrator.run_comprehensive_analysis(target_flux)
    
    # Generate report
    report_path = integrator.generate_report()
    
    # Create visualization
    plot_path = integrator.create_visualization()
    
    # Summary
    print("\n" + "="*50)
    print("VACUUM-ANEC INTEGRATION SUMMARY")
    print("="*50)
    
    if results['best_source']:
        best = results['sources'][results['best_source']]
        print(f"Best source: {results['best_source']}")
        print(f"ANEC flux: {best['anec_flux']:.2e} W")
        print(f"Target ratio: {best['target_ratio']:.2e}")
        print(f"Source type: {best['source_type']}")
        
        if best['target_ratio'] > 1.0:
            print("🎯 TARGET EXCEEDED - Experimentally viable!")
        elif best['target_ratio'] > 0.1:
            print("✅ Within order of magnitude - Promising approach")
        else:
            print("⚠️  Below target - Requires optimization")
    else:
        print("❌ No viable sources found for target flux")
    
    print(f"\nDetailed results: {report_path}")
    print(f"Visualization: {plot_path}")

if __name__ == "__main__":
    main()
