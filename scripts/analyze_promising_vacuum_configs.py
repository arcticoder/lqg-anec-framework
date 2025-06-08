#!/usr/bin/env python3
"""
Comprehensive Vacuum Configuration Analysis

This script performs parameter scans and optimization of vacuum engineering
configurations to identify the most promising laboratory sources for ANEC violation.

Features:
- Parameter sweeps for Casimir arrays and squeezed vacuum resonators
- Hybrid configuration optimization
- Multi-dimensional visualization of results
- Comprehensive feasibility assessment

Usage:
    python scripts/analyze_promising_vacuum_configs.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
from datetime import datetime
from itertools import product

from vacuum_engineering import (
    CasimirArray, DynamicCasimir, SqueezedVacuumResonator,
    vacuum_energy_to_anec_flux_compat as vacuum_energy_to_anec_flux, 
    MATERIAL_DATABASE
)

class VacuumConfigurationAnalyzer:
    """
    Comprehensive analyzer for vacuum engineering configurations.
    
    Performs parameter scans, optimization, and visualization to identify
    optimal laboratory configurations for ANEC violation.
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self.results = {}
        self.optimization_history = []
        
    def scan_casimir_arrays(self, spacing_range=(1e-9, 100e-9), 
                           layer_counts=(10, 50, 100, 200),
                           permittivities=(1.0, 2.0, 4.0, 8.0)):
        """
        Comprehensive parameter scan of Casimir array configurations.
        
        Args:
            spacing_range: (min, max) spacing in meters
            layer_counts: Number of layers to test
            permittivities: Relative permittivities to test
        """
        print("Scanning Casimir array configurations...")
        
        spacings = np.logspace(np.log10(spacing_range[0]), 
                              np.log10(spacing_range[1]), 20)
        
        results = {            'spacings': [],
            'layer_counts': [],
            'permittivities': [],
            'energy_densities': [],
            'anec_fluxes': [],
            'feasibility_scores': []
        }
        
        for spacing, n_layers, eps_r in product(spacings, layer_counts, permittivities):
            # Create uniform spacing array
            spacing_array = [spacing] * n_layers
            eps_array = [eps_r + 0.001j] * n_layers  # Add small imaginary part
            
            casimir = CasimirArray(temperature=300.0)
            pressure = casimir.stack_pressure(n_layers, spacing_array, eps_array)
            
            # Convert pressure to energy density (approximate)
            energy_density = pressure * spacing  # J/m³ = Pa * m
            anec_flux = vacuum_energy_to_anec_flux(energy_density)
            
            # Simple feasibility score based on magnitude and practicality
            size_penalty = np.exp(-(spacing * 1e9 - 10)**2 / 100)  # Peak at 10nm
            layer_penalty = np.exp(-(n_layers - 100)**2 / 10000)   # Peak at 100 layers
            feasibility = abs(anec_flux) * size_penalty * layer_penalty
            
            results['spacings'].append(spacing)
            results['layer_counts'].append(n_layers)
            results['permittivities'].append(eps_r)
            results['energy_densities'].append(energy_density)
            results['anec_fluxes'].append(anec_flux)
            results['feasibility_scores'].append(feasibility)
        
        self.results['casimir_scan'] = results
        print(f"Completed scan of {len(results['spacings'])} Casimir configurations")
        
    def scan_squeezed_vacuum(self, squeeze_range=(5, 30), n_points=50):        """
        Parameter scan of squeezed vacuum configurations.
        
        Args:
            squeeze_range: (min, max) squeezing in dB
            n_points: Number of points to sample
        """
        print("Scanning squeezed vacuum configurations...")
        
        squeeze_values = np.linspace(squeeze_range[0], squeeze_range[1], n_points)
        
        results = {
            'squeeze_db': [],
            'energy_densities': [],
            'anec_fluxes': [],
            'feasibility_scores': []
        }
        
        for squeeze_db in squeeze_values:
            squeezed = SqueezedVacuumResonator(resonator_frequency=1e12, 
                                             squeezing_parameter=squeeze_db/10.0)
            volume = 1e-6  # 1 cubic mm
            energy_density = squeezed.squeezed_energy_density(volume)
            anec_flux = vacuum_energy_to_anec_flux(energy_density)
            
            # Feasibility decreases with extreme squeezing (technical difficulty)
            tech_penalty = np.exp(-(squeeze_db - 15)**2 / 50)
            feasibility = abs(anec_flux) * tech_penalty
            
            results['squeeze_db'].append(squeeze_db)
            results['energy_densities'].append(energy_density)
            results['anec_fluxes'].append(anec_flux)
            results['feasibility_scores'].append(feasibility)
        
        self.results['squeezed_scan'] = results
        print(f"Completed scan of {n_points} squeezed vacuum configurations")
        
    def optimize_hybrid_configurations(self, n_trials=1000):
        """
        Optimize hybrid configurations combining multiple vacuum sources.
        
        Args:
            n_trials: Number of random configurations to test
        """
        print("Optimizing hybrid vacuum configurations...")
        
        best_configs = []
        
        for trial in range(n_trials):
            # Random parameter selection
            spacing = np.random.uniform(5e-9, 50e-9)
            n_layers = np.random.randint(50, 300)
            eps_r = np.random.uniform(1.0, 10.0)
            squeeze_db = np.random.uniform(10, 25)
            
            # Create components
            casimir = CasimirArray([spacing] * n_layers, [eps_r] * n_layers)
            squeezed = SqueezedVacuumResonator(squeeze_db)
            
            # Combined energy density (simple addition)
            total_density = casimir.total_density() + squeezed.total_density()
            total_flux = vacuum_energy_to_anec_flux(total_density)
            
            # Multi-objective feasibility score
            magnitude_score = abs(total_flux) / 1e20  # Normalize
            practicality_score = (
                np.exp(-(spacing * 1e9 - 10)**2 / 100) *
                np.exp(-(n_layers - 150)**2 / 5000) *
                np.exp(-(squeeze_db - 15)**2 / 25)
            )
            
            combined_score = magnitude_score * practicality_score
            
            config = {
                'trial': trial,
                'spacing': spacing,
                'n_layers': n_layers,
                'permittivity': eps_r,
                'squeeze_db': squeeze_db,
                'total_density': total_density,
                'anec_flux': total_flux,
                'feasibility_score': combined_score
            }
            
            best_configs.append(config)
            
            if trial % 100 == 0:
                print(f"  Trial {trial}/{n_trials}")
        
        # Sort by feasibility score
        best_configs.sort(key=lambda x: x['feasibility_score'], reverse=True)
        
        self.results['hybrid_optimization'] = best_configs
        print(f"Completed optimization of {n_trials} hybrid configurations")
        
    def visualize_results(self, output_dir='results'):
        """
        Generate comprehensive visualizations of all analysis results.
        
        Args:
            output_dir: Directory to save plots
        """
        print("Generating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Casimir Array Parameter Space
        if 'casimir_scan' in self.results:
            self._plot_casimir_parameter_space(output_dir)
        
        # 2. Squeezed Vacuum Performance
        if 'squeezed_scan' in self.results:
            self._plot_squeezed_vacuum_performance(output_dir)
        
        # 3. Hybrid Configuration Optimization
        if 'hybrid_optimization' in self.results:
            self._plot_hybrid_optimization(output_dir)
        
        # 4. Comparative Feasibility Analysis
        self._plot_comparative_analysis(output_dir)
        
        print(f"Visualizations saved to {output_dir}/")
        
    def _plot_casimir_parameter_space(self, output_dir):
        """Plot Casimir array parameter space analysis."""
        data = self.results['casimir_scan']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Spacing vs Energy Density
        ax = axes[0, 0]
        spacings_nm = np.array(data['spacings']) * 1e9
        scatter = ax.scatter(spacings_nm, np.abs(data['energy_densities']), 
                           c=data['layer_counts'], cmap='viridis', alpha=0.7)
        ax.set_xlabel('Spacing (nm)')
        ax.set_ylabel('|Energy Density| (J/m³)')
        ax.set_yscale('log')
        ax.set_title('Casimir Array: Spacing vs Energy Density')
        plt.colorbar(scatter, ax=ax, label='Layer Count')
        
        # Layer Count vs ANEC Flux
        ax = axes[0, 1]
        scatter = ax.scatter(data['layer_counts'], np.abs(data['anec_fluxes']),
                           c=data['permittivities'], cmap='plasma', alpha=0.7)
        ax.set_xlabel('Number of Layers')
        ax.set_ylabel('|ANEC Flux|')
        ax.set_yscale('log')
        ax.set_title('Layer Count vs ANEC Flux')
        plt.colorbar(scatter, ax=ax, label='Permittivity')
        
        # Feasibility Score Distribution
        ax = axes[1, 0]
        ax.hist(data['feasibility_scores'], bins=50, alpha=0.7, color='skyblue')
        ax.set_xlabel('Feasibility Score')
        ax.set_ylabel('Count')
        ax.set_title('Feasibility Score Distribution')
        
        # Best Configurations
        ax = axes[1, 1]
        # Find top 10% configurations
        n_top = len(data['feasibility_scores']) // 10
        sorted_indices = np.argsort(data['feasibility_scores'])[-n_top:]
        
        top_spacings = [data['spacings'][i] * 1e9 for i in sorted_indices]
        top_layers = [data['layer_counts'][i] for i in sorted_indices]
        top_scores = [data['feasibility_scores'][i] for i in sorted_indices]
        
        scatter = ax.scatter(top_spacings, top_layers, c=top_scores, 
                           cmap='Reds', s=60, alpha=0.8)
        ax.set_xlabel('Spacing (nm)')
        ax.set_ylabel('Layer Count')
        ax.set_title('Top 10% Configurations')
        plt.colorbar(scatter, ax=ax, label='Feasibility Score')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/casimir_parameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_squeezed_vacuum_performance(self, output_dir):
        """Plot squeezed vacuum performance analysis."""
        data = self.results['squeezed_scan']
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Energy density vs squeezing
        ax = axes[0]
        ax.plot(data['squeeze_db'], np.abs(data['energy_densities']), 
                'b-', linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Squeezing (dB)')
        ax.set_ylabel('|Energy Density| (J/m³)')
        ax.set_yscale('log')
        ax.set_title('Energy Density vs Squeezing')
        ax.grid(True, alpha=0.3)
        
        # ANEC flux vs squeezing
        ax = axes[1]
        ax.plot(data['squeeze_db'], np.abs(data['anec_fluxes']), 
                'r-', linewidth=2, marker='s', markersize=4)
        ax.set_xlabel('Squeezing (dB)')
        ax.set_ylabel('|ANEC Flux|')
        ax.set_yscale('log')
        ax.set_title('ANEC Flux vs Squeezing')
        ax.grid(True, alpha=0.3)
        
        # Feasibility optimization
        ax = axes[2]
        ax.plot(data['squeeze_db'], data['feasibility_scores'], 
                'g-', linewidth=2, marker='^', markersize=4)
        ax.set_xlabel('Squeezing (dB)')
        ax.set_ylabel('Feasibility Score')
        ax.set_title('Feasibility vs Squeezing')
        ax.grid(True, alpha=0.3)
        
        # Mark optimal point
        optimal_idx = np.argmax(data['feasibility_scores'])
        optimal_db = data['squeeze_db'][optimal_idx]
        optimal_score = data['feasibility_scores'][optimal_idx]
        ax.axvline(optimal_db, color='red', linestyle='--', alpha=0.7)
        ax.text(optimal_db + 1, optimal_score * 0.9, 
                f'Optimal: {optimal_db:.1f} dB', 
                rotation=90, va='top')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/squeezed_vacuum_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_hybrid_optimization(self, output_dir):
        """Plot hybrid configuration optimization results."""
        configs = self.results['hybrid_optimization'][:100]  # Top 100
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Extract data
        spacings = [c['spacing'] * 1e9 for c in configs]
        layers = [c['n_layers'] for c in configs]
        squeezing = [c['squeeze_db'] for c in configs]
        scores = [c['feasibility_score'] for c in configs]
        fluxes = [abs(c['anec_flux']) for c in configs]
        
        # Spacing vs Layers (colored by feasibility)
        ax = axes[0, 0]
        scatter = ax.scatter(spacings, layers, c=scores, cmap='viridis', s=60, alpha=0.7)
        ax.set_xlabel('Spacing (nm)')
        ax.set_ylabel('Layer Count')
        ax.set_title('Top Hybrid Configurations: Spacing vs Layers')
        plt.colorbar(scatter, ax=ax, label='Feasibility Score')
        
        # Squeezing vs ANEC Flux
        ax = axes[0, 1]
        scatter = ax.scatter(squeezing, fluxes, c=scores, cmap='plasma', s=60, alpha=0.7)
        ax.set_xlabel('Squeezing (dB)')
        ax.set_ylabel('|ANEC Flux|')
        ax.set_yscale('log')
        ax.set_title('Squeezing vs ANEC Flux')
        plt.colorbar(scatter, ax=ax, label='Feasibility Score')
        
        # Parameter correlation matrix
        ax = axes[1, 0]
        param_matrix = np.array([spacings, layers, squeezing, scores]).T
        correlation = np.corrcoef(param_matrix, rowvar=False)
        im = ax.imshow(correlation, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(['Spacing', 'Layers', 'Squeezing', 'Score'])
        ax.set_yticklabels(['Spacing', 'Layers', 'Squeezing', 'Score'])
        ax.set_title('Parameter Correlation Matrix')
        plt.colorbar(im, ax=ax)
        
        # Feasibility score evolution
        ax = axes[1, 1]
        all_scores = [c['feasibility_score'] for c in self.results['hybrid_optimization']]
        ax.plot(all_scores, alpha=0.7, linewidth=1)
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Feasibility Score')
        ax.set_title('Optimization Progress')
        ax.grid(True, alpha=0.3)
        
        # Mark best configuration
        best_trial = np.argmax(all_scores)
        ax.scatter([best_trial], [max(all_scores)], color='red', s=100, zorder=5)
        ax.text(best_trial, max(all_scores) * 1.05, 'Best', ha='center', color='red')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/hybrid_optimization_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_comparative_analysis(self, output_dir):
        """Plot comparative analysis across all vacuum sources."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Performance comparison
        ax = axes[0, 0]
        sources = []
        max_fluxes = []
        max_feasibilities = []
        
        if 'casimir_scan' in self.results:
            sources.append('Casimir Arrays')
            max_fluxes.append(max(np.abs(self.results['casimir_scan']['anec_fluxes'])))
            max_feasibilities.append(max(self.results['casimir_scan']['feasibility_scores']))
        
        if 'squeezed_scan' in self.results:
            sources.append('Squeezed Vacuum')
            max_fluxes.append(max(np.abs(self.results['squeezed_scan']['anec_fluxes'])))
            max_feasibilities.append(max(self.results['squeezed_scan']['feasibility_scores']))
        
        if 'hybrid_optimization' in self.results:
            sources.append('Hybrid Configs')
            max_fluxes.append(max([abs(c['anec_flux']) for c in self.results['hybrid_optimization']]))
            max_feasibilities.append(max([c['feasibility_score'] for c in self.results['hybrid_optimization']]))
        
        x_pos = np.arange(len(sources))
        ax.bar(x_pos, max_fluxes, alpha=0.7, color='skyblue')
        ax.set_xlabel('Vacuum Source Type')
        ax.set_ylabel('Maximum |ANEC Flux|')
        ax.set_yscale('log')
        ax.set_title('Peak Performance Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sources, rotation=45)
        
        # Feasibility comparison
        ax = axes[0, 1]
        ax.bar(x_pos, max_feasibilities, alpha=0.7, color='lightcoral')
        ax.set_xlabel('Vacuum Source Type')
        ax.set_ylabel('Maximum Feasibility Score')
        ax.set_title('Feasibility Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sources, rotation=45)
        
        # Technology readiness assessment
        ax = axes[1, 0]
        readiness_levels = [8, 6, 4]  # Casimir, Squeezed, Hybrid (TRL scale)
        colors = ['green', 'orange', 'red']
        ax.bar(x_pos, readiness_levels, alpha=0.7, color=colors)
        ax.set_xlabel('Vacuum Source Type')
        ax.set_ylabel('Technology Readiness Level')
        ax.set_title('Technology Readiness Assessment')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sources, rotation=45)
        ax.set_ylim(0, 10)
        
        # Implementation timeline
        ax = axes[1, 1]
        timelines = [2, 5, 10]  # Years to implementation
        ax.bar(x_pos, timelines, alpha=0.7, color='gold')
        ax.set_xlabel('Vacuum Source Type')
        ax.set_ylabel('Estimated Implementation Timeline (years)')
        ax.set_title('Implementation Timeline')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sources, rotation=45)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparative_vacuum_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_report(self, output_file='results/vacuum_configuration_analysis.json'):
        """
        Generate comprehensive JSON report of all analysis results.
        
        Args:
            output_file: Path to output JSON file
        """
        print("Generating comprehensive analysis report...")
        
        # Summary statistics
        summary = {
            'analysis_timestamp': datetime.now().isoformat(),
            'total_configurations_tested': 0,
            'best_performers': {},
            'recommendations': []
        }
        
        # Process Casimir results
        if 'casimir_scan' in self.results:
            casimir_data = self.results['casimir_scan']
            best_idx = np.argmax(casimir_data['feasibility_scores'])
            
            summary['best_performers']['casimir_array'] = {
                'spacing_nm': casimir_data['spacings'][best_idx] * 1e9,
                'layer_count': casimir_data['layer_counts'][best_idx],
                'permittivity': casimir_data['permittivities'][best_idx],
                'energy_density': casimir_data['energy_densities'][best_idx],
                'anec_flux': casimir_data['anec_fluxes'][best_idx],
                'feasibility_score': casimir_data['feasibility_scores'][best_idx]
            }
            summary['total_configurations_tested'] += len(casimir_data['spacings'])
        
        # Process squeezed vacuum results
        if 'squeezed_scan' in self.results:
            squeezed_data = self.results['squeezed_scan']
            best_idx = np.argmax(squeezed_data['feasibility_scores'])
            
            summary['best_performers']['squeezed_vacuum'] = {
                'squeezing_db': squeezed_data['squeeze_db'][best_idx],
                'energy_density': squeezed_data['energy_densities'][best_idx],
                'anec_flux': squeezed_data['anec_fluxes'][best_idx],
                'feasibility_score': squeezed_data['feasibility_scores'][best_idx]
            }
            summary['total_configurations_tested'] += len(squeezed_data['squeeze_db'])
        
        # Process hybrid optimization results
        if 'hybrid_optimization' in self.results:
            best_hybrid = self.results['hybrid_optimization'][0]  # Already sorted
            
            summary['best_performers']['hybrid_configuration'] = {
                'spacing_nm': best_hybrid['spacing'] * 1e9,
                'layer_count': best_hybrid['n_layers'],
                'permittivity': best_hybrid['permittivity'],
                'squeezing_db': best_hybrid['squeeze_db'],
                'total_energy_density': best_hybrid['total_density'],
                'anec_flux': best_hybrid['anec_flux'],
                'feasibility_score': best_hybrid['feasibility_score']
            }
            summary['total_configurations_tested'] += len(self.results['hybrid_optimization'])
        
        # Generate recommendations
        if summary['best_performers']:
            # Find overall best performer
            best_source = max(summary['best_performers'].items(), 
                            key=lambda x: x[1]['feasibility_score'])
            
            summary['recommendations'] = [
                f"Primary recommendation: {best_source[0]} configuration",
                f"Expected ANEC flux magnitude: {abs(best_source[1]['anec_flux']):.2e}",
                "Focus on optimizing fabrication techniques for selected configuration",
                "Develop comprehensive measurement protocols for validation",
                "Consider scaling studies for enhanced performance"
            ]
        
        # Create full report
        report = {
            'summary': summary,
            'detailed_results': self.results,
            'analysis_metadata': {
                'version': '1.0',
                'analysis_type': 'comprehensive_vacuum_configuration_scan',
                'computational_methods': [
                    'parameter_space_exploration',
                    'multi_objective_optimization',
                    'feasibility_assessment'
                ]
            }
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
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Comprehensive analysis report saved to {output_file}")
        return report

def main():
    """Main analysis routine."""
    print("=" * 60)
    print("COMPREHENSIVE VACUUM CONFIGURATION ANALYSIS")
    print("=" * 60)
    
    analyzer = VacuumConfigurationAnalyzer()
    
    # 1. Casimir Array Parameter Scan
    print("\n1. Casimir Array Parameter Space Exploration")
    analyzer.scan_casimir_arrays()
    
    # 2. Squeezed Vacuum Analysis
    print("\n2. Squeezed Vacuum Configuration Analysis")
    analyzer.scan_squeezed_vacuum()
    
    # 3. Hybrid Configuration Optimization
    print("\n3. Hybrid Configuration Optimization")
    analyzer.optimize_hybrid_configurations()
    
    # 4. Visualization
    print("\n4. Generating Visualizations")
    analyzer.visualize_results()
    
    # 5. Comprehensive Report
    print("\n5. Generating Comprehensive Report")
    report = analyzer.generate_report()
    
    # Print summary
    print("\n" + "=" * 60)
    print("ANALYSIS SUMMARY")
    print("=" * 60)
    
    if 'summary' in report:
        summary = report['summary']
        print(f"Total configurations tested: {summary['total_configurations_tested']}")
        print(f"Analysis completed: {summary['analysis_timestamp']}")
        
        if 'best_performers' in summary:
            print("\nBest performing configurations:")
            for source_type, config in summary['best_performers'].items():
                print(f"\n{source_type.upper()}:")
                print(f"  ANEC flux magnitude: {abs(config['anec_flux']):.2e}")
                print(f"  Feasibility score: {config['feasibility_score']:.4f}")
        
        if 'recommendations' in summary:
            print("\nRecommendations:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"  {i}. {rec}")
    
    print("\n" + "=" * 60)
    print("Analysis complete! Check results/ directory for detailed outputs.")
    print("=" * 60)

if __name__ == "__main__":
    main()
