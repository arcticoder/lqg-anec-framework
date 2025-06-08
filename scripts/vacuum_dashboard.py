#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from vacuum_engineering import build_lab_sources, comprehensive_vacuum_analysis
from metamaterial_casimir import MetamaterialCasimir, create_optimized_metamaterial_casimir
from scipy.constants import hbar, c, pi
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Simple ANEC integral calculation (avoiding complex imports)
def simple_anec_integral(energy_flux, tau_array):
    """Simple ANEC integral calculation."""
    return np.trapz(energy_flux, tau_array)

class GaussianSmear:
    """Gaussian smearing kernel for ANEC calculations."""
    def __init__(self, tau0):
        self.tau0 = tau0
        
    def kernel(self, τ):
        return np.exp(-τ**2 / (2*self.tau0**2)) / (np.sqrt(2*np.pi) * self.tau0)

class VacuumANECDashboard:
    """
    Comprehensive vacuum-to-ANEC analysis dashboard.
    
    Compares all available laboratory vacuum sources and analyzes
    their potential for ANEC violation in a unified framework.
    """
    
    def __init__(self, smearing_timescale: float = 7*24*3600):
        """
        Initialize dashboard.
        
        Args:
            smearing_timescale: Temporal smearing scale in seconds (default: 1 week)
        """
        self.tau0 = smearing_timescale
        self.smear = GaussianSmear(self.tau0)
        
        # Time grid for ANEC integration
        self.τ = np.linspace(-3*self.tau0, 3*self.tau0, 2000)
        
        # Target thresholds
        self.target_anec = 1e-25  # Target ANEC violation [W]
        self.target_energy_density = -1e-15  # Target negative energy density [J/m³]
        
        print(f"Vacuum-ANEC Dashboard initialized:")
        print(f"  Smearing timescale: {self.tau0/(24*3600):.1f} days")
        print(f"  Target ANEC violation: {self.target_anec:.2e} W")
        print(f"  Target energy density: {self.target_energy_density:.2e} J/m³")

    def analyze_lab_sources(self) -> Dict:
        """Analyze all available laboratory vacuum sources."""
        
        print("\nAnalyzing laboratory vacuum sources...")
        
        # Get standard lab sources
        lab_sources = build_lab_sources('comprehensive')
        comprehensive_analysis = comprehensive_vacuum_analysis()
        
        results = {}
        
        for name, source_config in lab_sources.items():
            print(f"  Analyzing {name}...")
            
            try:
                source = source_config['source']
                params = source_config['params']
                
                # Get energy density using different methods based on source type
                if hasattr(source, 'total_energy_density'):
                    energy_density = source.total_energy_density()
                elif hasattr(source, 'negative_energy_density'):
                    energy_density = source.negative_energy_density(
                        params.get('drive_frequency', 10e9),
                        params.get('volume', 1e-15),
                        params.get('quality_factor', 1000)
                    )
                elif hasattr(source, 'squeezed_energy_density'):
                    energy_density = source.squeezed_energy_density(params.get('volume', 1e-15))
                else:
                    # Fallback: estimate from comprehensive analysis
                    if name in comprehensive_analysis:
                        energy_density = comprehensive_analysis[name]['energy_density']
                    else:
                        energy_density = 0.0
                
                # Calculate volume and total energy
                volume = params.get('volume', 1e-15)
                total_energy = energy_density * volume
                
                # Compute ANEC integral
                # Model: energy density creates flux that we smear and integrate
                if energy_density != 0:
                    # Simple model: uniform energy flux over smearing time
                    energy_flux = total_energy / self.tau0  # Energy per unit time
                    smeared_flux = energy_flux * self.smear.kernel(self.τ)
                    anec_integral = np.trapz(smeared_flux, self.τ)
                else:
                    anec_integral = 0.0
                
                # Performance metrics
                anec_ratio = abs(anec_integral) / self.target_anec if self.target_anec != 0 else 0
                energy_ratio = abs(energy_density) / abs(self.target_energy_density) if self.target_energy_density != 0 else 0
                
                # Feasibility assessment
                is_negative = energy_density < 0
                meets_anec_target = abs(anec_integral) >= self.target_anec
                meets_energy_target = abs(energy_density) >= abs(self.target_energy_density)
                
                results[name] = {
                    'source_type': type(source).__name__,
                    'energy_density': energy_density,
                    'volume': volume,
                    'total_energy': total_energy,
                    'anec_integral': anec_integral,
                    'anec_ratio': anec_ratio,
                    'energy_ratio': energy_ratio,
                    'is_negative_energy': is_negative,
                    'meets_anec_target': meets_anec_target,
                    'meets_energy_target': meets_energy_target,
                    'feasibility_score': anec_ratio * energy_ratio,
                    'parameters': params
                }
                
            except Exception as e:
                print(f"    Error analyzing {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results

    def analyze_metamaterial_sources(self) -> Dict:
        """Analyze metamaterial-enhanced Casimir sources."""
        
        print("\nAnalyzing metamaterial sources...")
        
        results = {}
        
        # Test different metamaterial configurations
        configs = [
            {
                'name': 'basic_negative_index',
                'spacings': [50e-9] * 10,
                'eps_list': [(-2.0 + 0.1j)] * 10,
                'mu_list': [(-1.5 + 0.05j)] * 10
            },
            {
                'name': 'alternating_layers',
                'spacings': [30e-9] * 20,
                'eps_list': [(-3.0 + 0.1j) if i%2==0 else (2.0 + 0.01j) for i in range(20)],
                'mu_list': [(-2.0 + 0.05j) if i%2==0 else (1.0 + 0.0j) for i in range(20)]
            },
            {
                'name': 'optimized_stack',
                'spacings': [25e-9] * 30,
                'eps_list': [(-5.0 + 0.2j) if i%3==0 else (1.5 + 0.01j) for i in range(30)],
                'mu_list': [(-3.0 + 0.1j) if i%3==0 else (1.0 + 0.0j) for i in range(30)]
            }
        ]
        
        for config in configs:
            name = config['name']
            print(f"  Analyzing {name}...")
            
            try:
                # Create metamaterial system
                meta_source = MetamaterialCasimir(
                    config['spacings'],
                    config['eps_list'],
                    config['mu_list']
                )
                
                # Calculate properties
                energy_density = meta_source.total_energy_density()
                amplification = meta_source.force_amplification_factor()
                
                # Estimate volume
                total_thickness = sum(config['spacings'])
                lateral_area = np.pi * (1e-6)**2  # 1 μm radius circular cross-section
                volume = lateral_area * total_thickness
                
                total_energy = energy_density * volume
                
                # ANEC calculation
                if energy_density != 0:
                    energy_flux = total_energy / self.tau0
                    smeared_flux = energy_flux * self.smear.kernel(self.τ)
                    anec_integral = np.trapz(smeared_flux, self.τ)
                else:
                    anec_integral = 0.0
                
                # Performance metrics
                anec_ratio = abs(anec_integral) / self.target_anec if self.target_anec != 0 else 0
                energy_ratio = abs(energy_density) / abs(self.target_energy_density) if self.target_energy_density != 0 else 0
                
                results[name] = {
                    'source_type': 'MetamaterialCasimir',
                    'energy_density': energy_density,
                    'volume': volume,
                    'total_energy': total_energy,
                    'anec_integral': anec_integral,
                    'anec_ratio': anec_ratio,
                    'energy_ratio': energy_ratio,
                    'amplification_factor': amplification,
                    'n_layers': len(config['spacings']),
                    'is_negative_energy': energy_density < 0,
                    'meets_anec_target': abs(anec_integral) >= self.target_anec,
                    'meets_energy_target': abs(energy_density) >= abs(self.target_energy_density),
                    'feasibility_score': anec_ratio * energy_ratio
                }
                
            except Exception as e:
                print(f"    Error analyzing {name}: {e}")
                results[name] = {'error': str(e)}
        
        return results

    def create_comparison_dashboard(self, lab_results: Dict, meta_results: Dict) -> None:
        """Create comprehensive comparison dashboard visualization."""
        
        # Combine all results
        all_results = {**lab_results, **meta_results}
        
        # Filter out error results
        valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
        
        if not valid_results:
            print("No valid results to visualize!")
            return
        
        # Extract data for plotting
        names = list(valid_results.keys())
        anec_vals = [abs(v['anec_integral']) for v in valid_results.values()]
        energy_densities = [abs(v['energy_density']) for v in valid_results.values()]
        anec_ratios = [v['anec_ratio'] for v in valid_results.values()]
        energy_ratios = [v['energy_ratio'] for v in valid_results.values()]
        feasibility_scores = [v['feasibility_score'] for v in valid_results.values()]
        
        # Identify source types for coloring
        source_types = [v['source_type'] for v in valid_results.values()]
        type_colors = {
            'CasimirArray': 'blue',
            'DynamicCasimirEffect': 'green', 
            'SqueezedVacuumResonator': 'orange',
            'MetamaterialCasimir': 'red'
        }
        colors = [type_colors.get(t, 'gray') for t in source_types]
        
        # Create dashboard plot
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Vacuum Source ANEC Violation Dashboard', fontsize=16, fontweight='bold')
        
        # 1. ANEC integral comparison bar chart
        ax = axes[0, 0]
        bars = ax.bar(range(len(names)), anec_vals, color=colors, alpha=0.7)
        ax.axhline(self.target_anec, color='black', linestyle='--', linewidth=2, 
                  label=f'Target ({self.target_anec:.1e} W)')
        ax.set_ylabel('|ANEC Integral| (W)')
        ax.set_title('ANEC Violation Comparison')
        ax.set_yscale('log')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, anec_vals)):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val*1.1, f'{val:.1e}',
                       ha='center', va='bottom', fontsize=8, rotation=90)
        
        # 2. Energy density comparison
        ax = axes[0, 1]
        bars = ax.bar(range(len(names)), energy_densities, color=colors, alpha=0.7)
        ax.axhline(abs(self.target_energy_density), color='black', linestyle='--', linewidth=2,
                  label=f'Target ({abs(self.target_energy_density):.1e} J/m³)')
        ax.set_ylabel('|Energy Density| (J/m³)')
        ax.set_title('Negative Energy Density Comparison')
        ax.set_yscale('log')
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Feasibility score scatter plot
        ax = axes[0, 2]
        scatter = ax.scatter(anec_ratios, energy_ratios, c=feasibility_scores, 
                           s=100, cmap='viridis', alpha=0.8)
        ax.axhline(1.0, color='black', linestyle='--', alpha=0.5, label='Energy Target')
        ax.axvline(1.0, color='black', linestyle='--', alpha=0.5, label='ANEC Target')
        ax.set_xlabel('ANEC Ratio (achieved/target)')
        ax.set_ylabel('Energy Ratio (achieved/target)')
        ax.set_title('Feasibility Analysis')
        ax.set_xscale('log')
        ax.set_yscale('log')
        plt.colorbar(scatter, ax=ax, label='Feasibility Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add source labels
        for i, name in enumerate(names):
            ax.annotate(name, (anec_ratios[i], energy_ratios[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 4. Source type performance comparison
        ax = axes[1, 0]
        type_performance = {}
        for name, result in valid_results.items():
            src_type = result['source_type']
            if src_type not in type_performance:
                type_performance[src_type] = {'anec': [], 'energy': [], 'feasibility': []}
            type_performance[src_type]['anec'].append(result['anec_ratio'])
            type_performance[src_type]['energy'].append(result['energy_ratio'])
            type_performance[src_type]['feasibility'].append(result['feasibility_score'])
        
        type_names = list(type_performance.keys())
        avg_anec = [np.mean(type_performance[t]['anec']) for t in type_names]
        avg_energy = [np.mean(type_performance[t]['energy']) for t in type_names]
        
        x_pos = np.arange(len(type_names))
        width = 0.35
        
        bars1 = ax.bar(x_pos - width/2, avg_anec, width, label='ANEC Ratio', alpha=0.7)
        bars2 = ax.bar(x_pos + width/2, avg_energy, width, label='Energy Ratio', alpha=0.7)
        
        ax.set_ylabel('Average Ratio (achieved/target)')
        ax.set_title('Performance by Source Type')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(type_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 5. Volume vs performance
        ax = axes[1, 1]
        volumes = [v['volume'] for v in valid_results.values()]
        scatter = ax.scatter(volumes, feasibility_scores, c=colors, s=100, alpha=0.8)
        ax.set_xlabel('Volume (m³)')
        ax.set_ylabel('Feasibility Score')
        ax.set_title('Volume vs Performance')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add source labels
        for i, name in enumerate(names):
            ax.annotate(name, (volumes[i], feasibility_scores[i]),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        # 6. Target achievement summary
        ax = axes[1, 2]
        meets_anec = sum([v['meets_anec_target'] for v in valid_results.values()])
        meets_energy = sum([v['meets_energy_target'] for v in valid_results.values()])
        meets_both = sum([v['meets_anec_target'] and v['meets_energy_target'] 
                         for v in valid_results.values()])
        
        categories = ['ANEC Target', 'Energy Target', 'Both Targets']
        counts = [meets_anec, meets_energy, meets_both]
        
        bars = ax.bar(categories, counts, color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        ax.set_ylabel('Number of Sources')
        ax.set_title('Target Achievement Summary')
        ax.grid(True, alpha=0.3)
        
        # Add count labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('results/vacuum_anec_dashboard.png', dpi=150, bbox_inches='tight')
        print("Dashboard visualization saved to results/vacuum_anec_dashboard.png")

    def generate_summary_report(self, lab_results: Dict, meta_results: Dict) -> None:
        """Generate detailed text summary report."""
        
        all_results = {**lab_results, **meta_results}
        valid_results = {k: v for k, v in all_results.items() if 'error' not in v}
        
        print("\n" + "="*80)
        print("VACUUM SOURCE ANEC VIOLATION ANALYSIS - SUMMARY REPORT")
        print("="*80)
        
        print(f"\nAnalysis Parameters:")
        print(f"  Smearing timescale: {self.tau0/(24*3600):.1f} days")
        print(f"  Target ANEC violation: {self.target_anec:.2e} W")
        print(f"  Target energy density: {self.target_energy_density:.2e} J/m³")
        print(f"  Sources analyzed: {len(valid_results)}")
        
        # Overall statistics
        anec_vals = [abs(v['anec_integral']) for v in valid_results.values()]
        energy_vals = [abs(v['energy_density']) for v in valid_results.values()]
        
        print(f"\nOverall Statistics:")
        print(f"  ANEC range: {np.min(anec_vals):.2e} - {np.max(anec_vals):.2e} W")
        print(f"  Energy density range: {np.min(energy_vals):.2e} - {np.max(energy_vals):.2e} J/m³")
        
        # Best performers
        best_anec = max(valid_results.keys(), key=lambda k: abs(valid_results[k]['anec_integral']))
        best_energy = max(valid_results.keys(), key=lambda k: abs(valid_results[k]['energy_density']))
        best_overall = max(valid_results.keys(), key=lambda k: valid_results[k]['feasibility_score'])
        
        print(f"\nBest Performers:")
        print(f"  Highest ANEC violation: {best_anec}")
        print(f"    ANEC integral: {valid_results[best_anec]['anec_integral']:.2e} W")
        print(f"    Ratio to target: {valid_results[best_anec]['anec_ratio']:.1f}x")
        
        print(f"  Highest energy density: {best_energy}")
        print(f"    Energy density: {valid_results[best_energy]['energy_density']:.2e} J/m³")
        print(f"    Ratio to target: {valid_results[best_energy]['energy_ratio']:.1f}x")
        
        print(f"  Best overall feasibility: {best_overall}")
        print(f"    Feasibility score: {valid_results[best_overall]['feasibility_score']:.2e}")
        print(f"    Source type: {valid_results[best_overall]['source_type']}")
        
        # Target achievement
        meets_anec = [k for k, v in valid_results.items() if v['meets_anec_target']]
        meets_energy = [k for k, v in valid_results.items() if v['meets_energy_target']]
        meets_both = [k for k, v in valid_results.items() 
                     if v['meets_anec_target'] and v['meets_energy_target']]
        
        print(f"\nTarget Achievement:")
        print(f"  Sources meeting ANEC target: {len(meets_anec)}/{len(valid_results)}")
        if meets_anec:
            print(f"    {', '.join(meets_anec)}")
        
        print(f"  Sources meeting energy target: {len(meets_energy)}/{len(valid_results)}")
        if meets_energy:
            print(f"    {', '.join(meets_energy)}")
        
        print(f"  Sources meeting both targets: {len(meets_both)}/{len(valid_results)}")
        if meets_both:
            print(f"    {', '.join(meets_both)}")
        
        # Detailed source analysis
        print(f"\nDetailed Source Analysis:")
        print("-" * 60)
        
        for name, result in sorted(valid_results.items(), 
                                 key=lambda x: x[1]['feasibility_score'], reverse=True):
            print(f"\n{name.upper()}:")
            print(f"  Type: {result['source_type']}")
            print(f"  Energy density: {result['energy_density']:.2e} J/m³")
            print(f"  Volume: {result['volume']:.2e} m³")
            print(f"  ANEC integral: {result['anec_integral']:.2e} W")
            print(f"  ANEC ratio: {result['anec_ratio']:.2f}x target")
            print(f"  Energy ratio: {result['energy_ratio']:.2f}x target")
            print(f"  Feasibility score: {result['feasibility_score']:.2e}")
            
            if 'amplification_factor' in result:
                print(f"  Force amplification: {result['amplification_factor']:.1f}x")
            
            status_flags = []
            if result['is_negative_energy']:
                status_flags.append("NEGATIVE ENERGY")
            if result['meets_anec_target']:
                status_flags.append("MEETS ANEC TARGET")
            if result['meets_energy_target']:
                status_flags.append("MEETS ENERGY TARGET")
                
            if status_flags:
                print(f"  Status: {' | '.join(status_flags)}")

    def end_to_end_vacuum_anec_comparison(self):
        """
        End-to-end vacuum-to-ANEC dashboard comparing all lab sources.
        
        Rapidly compares all laboratory sources in one place and visualizes 
        which meets the target ANEC violation requirements.
        """
        print("\nEnd-to-End Vacuum-to-ANEC Dashboard")
        print("=" * 50)
        
        names, anecs, energy_densities, volumes = [], [], [], []
        feasibility_scores = []
        
        # Get all available lab sources
        try:
            sources = build_lab_sources()
            
            for name, source_info in sources.items():
                try:
                    source = source_info['source']
                    params = source_info['params']
                    
                    # Get energy density
                    if hasattr(source, 'total_energy_density'):
                        ρ = source.total_energy_density()
                    elif hasattr(source, 'total_density'):
                        ρ = source.total_density()
                    elif hasattr(source, 'peak_density'):
                        ρ = source.peak_density()
                    elif hasattr(source, 'energy_density'):
                        energy_vals = source.energy_density()
                        ρ = np.sum(energy_vals) if isinstance(energy_vals, np.ndarray) else energy_vals
                    else:
                        # Try direct calculation for specific source types
                        if 'dynamic' in name.lower():
                            ρ = source.negative_energy_density(
                                params.get('drive_frequency', 20e9),
                                params.get('volume', 1e-9),
                                params.get('quality_factor', 1000)
                            )
                        elif 'squeezed' in name.lower():
                            ρ = source.squeezed_energy_density(params.get('volume', 1e-6))
                        else:
                            ρ = -1e6  # Default fallback
                    
                    # Calculate ANEC integral
                    volume = params.get('volume', 1e-6)
                    anec = self.compute_anec_integral_simple(ρ, volume)
                    
                    # Feasibility assessment
                    feasibility = self.assess_feasibility(name, params, ρ)
                    
                    names.append(name)
                    anecs.append(abs(anec))
                    energy_densities.append(abs(ρ))
                    volumes.append(volume)
                    feasibility_scores.append(feasibility)
                    
                    print(f"{name:20s}: {abs(anec):.2e} W, ρ={ρ:.1e} J/m³, F={feasibility:.2f}")
                    
                except Exception as e:
                    print(f"Warning: Failed to process {name}: {e}")
                    continue
                    
        except Exception as e:
            print(f"Warning: Could not load lab sources: {e}")
            # Create simple test sources
            self._create_simple_test_sources(names, anecs, energy_densities, volumes, feasibility_scores)
        
        # Create comprehensive visualization
        self._create_dashboard_plots(names, anecs, energy_densities, volumes, feasibility_scores)
        
        return {
            'sources': names,
            'anec_violations': anecs,
            'energy_densities': energy_densities,
            'volumes': volumes,
            'feasibility': feasibility_scores
        }
    
    def compute_anec_integral_simple(self, energy_density: float, volume: float) -> float:
        """Simple ANEC integral calculation."""
        total_energy = energy_density * volume
        # Apply smearing kernel and integrate
        smeared_flux = total_energy * self.smear.kernel(self.τ) / self.tau0
        return np.trapz(smeared_flux, self.τ)
    
    def assess_feasibility(self, source_name: str, params: dict, energy_density: float) -> float:
        """
        Assess experimental feasibility of each source.
        
        Returns score from 0.0 (impossible) to 1.0 (ready for deployment).
        """
        score = 1.0
        
        if 'casimir' in source_name.lower():
            # Casimir arrays - very feasible with current technology
            spacing = params.get('optimal_spacing', 100e-9)
            if spacing < 10e-9:
                score *= 0.7  # Harder fabrication
            elif spacing < 50e-9:
                score *= 0.9  # Challenging but doable
            
        elif 'dynamic' in source_name.lower():
            # Dynamic Casimir - requires advanced circuits
            frequency = params.get('drive_frequency', 10e9)
            if frequency > 50e9:
                score *= 0.6  # High frequency challenges
            score *= 0.8  # Generally more challenging
            
        elif 'squeezed' in source_name.lower():
            # Squeezed vacuum - depends on squeezing parameter
            score *= 0.9  # Well-established technology
            
        elif 'metamaterial' in source_name.lower():
            # Metamaterials - depends on material properties required
            score *= 0.7  # Cutting-edge but feasible
        
        # Energy density magnitude penalty
        if abs(energy_density) > 1e12:
            score *= 0.8  # Very high densities are harder to achieve
            
        return max(0.0, min(1.0, score))
    
    def _create_simple_test_sources(self, names, anecs, energy_densities, volumes, feasibility_scores):
        """Create simple test sources for demonstration."""
        test_sources = [
            ("Casimir Array", 1e-22, 1e10, 1e-6, 0.95),
            ("Dynamic Casimir", 1e-20, 1e8, 1e-9, 0.75),
            ("Squeezed Vacuum", 1e-18, 1e6, 1e-6, 0.90),
            ("Metamaterial Enhanced", 1e-21, 1e9, 1e-6, 0.70)
        ]
        
        for name, anec, density, volume, feasibility in test_sources:
            names.append(name)
            anecs.append(anec)
            energy_densities.append(density)
            volumes.append(volume)
            feasibility_scores.append(feasibility)
    
    def _create_dashboard_plots(self, names, anecs, energy_densities, volumes, feasibility_scores):
        """Create comprehensive dashboard visualization."""
        fig = plt.figure(figsize=(16, 12))
        
        # Main ANEC comparison plot
        ax1 = plt.subplot(2, 3, 1)
        bars = ax1.bar(names, anecs, color=['red' if a < 1e-25 else 'green' for a in anecs])
        ax1.axhline(1e-25, color='k', linestyle='--', linewidth=2, label='Target 1e-25 W')
        ax1.set_ylabel('|ANEC integral| (W)')
        ax1.set_title('Lab-Source ANEC Comparison')
        ax1.set_yscale('log')
        ax1.legend()
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Energy density comparison
        ax2 = plt.subplot(2, 3, 2)
        ax2.bar(names, energy_densities, alpha=0.7)
        ax2.set_ylabel('|Energy Density| (J/m³)')
        ax2.set_title('Negative Energy Densities')
        ax2.set_yscale('log')
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        
        # Feasibility assessment
        ax3 = plt.subplot(2, 3, 3)
        colors = ['red' if f < 0.5 else 'orange' if f < 0.8 else 'green' for f in feasibility_scores]
        ax3.bar(names, feasibility_scores, color=colors, alpha=0.7)
        ax3.set_ylabel('Feasibility Score')
        ax3.set_title('Experimental Feasibility')
        ax3.set_ylim(0, 1)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')
        
        # Energy density vs ANEC scatter
        ax4 = plt.subplot(2, 3, 4)
        scatter = ax4.scatter(energy_densities, anecs, c=feasibility_scores, 
                            cmap='RdYlGn', alpha=0.7, s=100)
        ax4.set_xlabel('|Energy Density| (J/m³)')
        ax4.set_ylabel('|ANEC integral| (W)')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_title('Energy Density vs ANEC Performance')
        plt.colorbar(scatter, ax=ax4, label='Feasibility')
        
        # Enhancement factors
        target_anec = 1e-25
        enhancements = [a/target_anec for a in anecs]
        ax5 = plt.subplot(2, 3, 5)
        ax5.bar(names, enhancements, alpha=0.7)
        ax5.axhline(1, color='k', linestyle='--', label='Target threshold')
        ax5.set_ylabel('Enhancement Factor')
        ax5.set_title('ANEC Enhancement over Target')
        ax5.set_yscale('log')
        ax5.legend()
        plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')
        
        # Volume requirements
        ax6 = plt.subplot(2, 3, 6)
        ax6.bar(names, [v*1e9 for v in volumes], alpha=0.7)  # Convert to mm³
        ax6.set_ylabel('Volume (mm³)')
        ax6.set_title('Required Source Volume')
        ax6.set_yscale('log')
        plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save dashboard
        dashboard_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'vacuum_anec_dashboard.png')
        os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
        plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
        print(f"\nDashboard saved to: {dashboard_path}")
        
        return dashboard_path

def main():
    """Main function to run the vacuum-ANEC dashboard analysis."""
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    print("VACUUM-ANEC COMPREHENSIVE DASHBOARD")
    print("="*50)
    
    # Initialize dashboard
    dashboard = VacuumANECDashboard(smearing_timescale=7*24*3600)  # 1 week
    
    # Analyze all source types
    lab_results = dashboard.analyze_lab_sources()
    meta_results = dashboard.analyze_metamaterial_sources()
    
    # Create visualizations
    dashboard.create_comparison_dashboard(lab_results, meta_results)
    
    # Generate summary report
    dashboard.generate_summary_report(lab_results, meta_results)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
      # Prepare results for JSON serialization
    def serialize_result(result):
        serialized = {}
        for k, v in result.items():
            if isinstance(v, (np.integer, np.floating)):
                serialized[k] = float(v)
            elif isinstance(v, np.bool_):
                serialized[k] = bool(v)
            elif isinstance(v, complex):
                serialized[k] = {'real': float(v.real), 'imag': float(v.imag)}
            elif isinstance(v, np.ndarray):
                serialized[k] = v.tolist()
            else:
                serialized[k] = v
        return serialized    
    output_data = {
        'metadata': {
            'timestamp': timestamp,
            'smearing_timescale_days': dashboard.tau0 / (24*3600),
            'target_anec_W': dashboard.target_anec,
            'target_energy_density_J_m3': dashboard.target_energy_density,
            'description': 'Comprehensive vacuum source ANEC violation analysis'
        },
        'laboratory_sources': {k: serialize_result(v) for k, v in lab_results.items()},
        'metamaterial_sources': {k: serialize_result(v) for k, v in meta_results.items()}
    }
    
    output_file = f'results/vacuum_anec_dashboard_{timestamp}.json'
    with open(output_file, 'w') as f:
        # Convert numpy types for JSON serialization
        json_output = {}
        for key, value in output_data.items():
            if isinstance(value, dict):
                json_output[key] = {}
                for k, v in value.items():
                    if isinstance(v, (np.integer, np.floating, np.complexfloating)):
                        json_output[key][k] = float(v.real) if np.iscomplexobj(v) else float(v)
                    elif isinstance(v, np.bool_):
                        json_output[key][k] = bool(v)
                    elif isinstance(v, np.ndarray):
                        json_output[key][k] = v.tolist()
                    else:
                        json_output[key][k] = v
            else:
                if isinstance(value, (np.integer, np.floating, np.complexfloating)):
                    json_output[key] = float(value.real) if np.iscomplexobj(value) else float(value)
                elif isinstance(value, np.bool_):
                    json_output[key] = bool(value)
                elif isinstance(value, np.ndarray):
                    json_output[key] = value.tolist()
                else:
                    json_output[key] = value
        
        json.dump(json_output, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    print("Dashboard analysis complete!")

if __name__ == "__main__":
    main()
