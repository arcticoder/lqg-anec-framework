#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib.pyplot as plt
from metamaterial_casimir import MetamaterialCasimir
from vacuum_engineering import vacuum_energy_to_anec_flux_simple
from scipy.constants import hbar, c
import json
from datetime import datetime

def compute_anec_integral(smeared_density_times_kernel, τ):
    """Simple ANEC integral computation."""
    return np.trapz(smeared_density_times_kernel, τ)

class GaussianSmear:
    """Simple Gaussian smearing kernel for ANEC calculations."""
    def __init__(self, tau0):
        self.tau0 = tau0
        
    def kernel(self, τ):
        return np.exp(-τ**2 / (2*self.tau0**2)) / (np.sqrt(2*np.pi) * self.tau0)

def sweep_metamaterial_parameters():
    """
    Comprehensive parameter sweep for metamaterial Casimir systems.
    
    Sweeps over:
    - Layer spacings
    - Permittivity values (positive and negative)
    - Permeability values (positive and negative) 
    - Number of layers
    """
    print("Starting Metamaterial Casimir Parameter Sweep")
    print("=" * 50)
    
    # Smearing parameters
    week = 7*24*3600  # 1 week in seconds
    smear = GaussianSmear(tau0=week)
    τ = np.linspace(-3*week, 3*week, 1000)
    
    # Parameter ranges
    gaps = [10e-9, 20e-9, 50e-9, 100e-9, 200e-9]  # 10 nm to 200 nm
    eps_vals = [1.0, 2.0, -1.0, -2.0, -5.0]       # Permittivity range
    mu_vals = [1.0, 1.5, -1.0, -1.5, -3.0]        # Permeability range 
    layer_counts = [5, 10, 20, 50]                 # Number of layers
    
    results = []
    total_configs = len(gaps) * len(eps_vals) * len(mu_vals) * len(layer_counts)
    config_count = 0
    
    print(f"Total configurations to test: {total_configs}")
    
    for n_layers in layer_counts:
        for a in gaps:
            for eps in eps_vals:
                for mu in mu_vals:
                    config_count += 1
                    
                    if config_count % 10 == 0:
                        print(f"Progress: {config_count}/{total_configs} ({100*config_count/total_configs:.1f}%)")
                    
                    try:
                        # Create metamaterial system
                        spacings = [a] * n_layers
                        eps_list = [eps + 0.1j] * n_layers  # Add small imaginary part
                        mu_list = [mu + 0.05j] * n_layers
                        
                        src = MetamaterialCasimir(spacings, eps_list, mu_list)
                        
                        # Calculate energy density
                        energy_density = src.total_energy_density()
                        
                        # Calculate ANEC integral (simplified)
                        # Assume uniform energy density over smearing volume
                        volume = np.pi * (1e-6)**2 * a * n_layers  # Cylindrical volume estimate
                        energy_flux = energy_density * volume
                        
                        # Smear and integrate
                        smeared_flux = energy_flux * smear.kernel(τ)
                        anec = np.trapz(smeared_flux, τ)
                        
                        # Force amplification
                        amplification = src.force_amplification_factor()
                        
                        # Feasibility metrics
                        is_negative_energy = energy_density < 0
                        is_negative_index = (np.real(eps) < 0 and np.real(mu) < 0)
                        significant_anec = abs(anec) > 1e-30  # Threshold for significance
                        
                        result = {
                            'spacing_nm': a * 1e9,
                            'permittivity': eps,
                            'permeability': mu,
                            'n_layers': n_layers,
                            'energy_density': energy_density,
                            'anec_integral': anec,
                            'amplification_factor': amplification,
                            'is_negative_energy': is_negative_energy,
                            'is_negative_index': is_negative_index,
                            'significant_anec': significant_anec,
                            'volume': volume
                        }
                        
                        results.append(result)
                        
                    except Exception as e:
                        print(f"Error with config (a={a*1e9:.0f}nm, ε={eps}, μ={mu}, n={n_layers}): {e}")
                        continue
    
    return results

def analyze_sweep_results(results):
    """Analyze parameter sweep results and find optimal configurations."""
    
    # Convert to numpy arrays for analysis
    data = {}
    for key in results[0].keys():
        if isinstance(results[0][key], (int, float, complex)):
            data[key] = np.array([r[key] for r in results])
    
    print(f"\nAnalysis of {len(results)} configurations:")
    print("-" * 40)
    
    # Find configurations with negative energy
    neg_energy_mask = data['energy_density'] < 0
    n_neg_energy = np.sum(neg_energy_mask)
    print(f"Configurations with negative energy: {n_neg_energy}/{len(results)} ({100*n_neg_energy/len(results):.1f}%)")
    
    # Find negative index materials
    neg_index_mask = np.array([r['is_negative_index'] for r in results])
    n_neg_index = np.sum(neg_index_mask)
    print(f"Negative index configurations: {n_neg_index}/{len(results)} ({100*n_neg_index/len(results):.1f}%)")
    
    # Find configurations with significant ANEC violation
    sig_anec_mask = np.array([r['significant_anec'] for r in results])
    n_sig_anec = np.sum(sig_anec_mask)
    print(f"Significant ANEC violations: {n_sig_anec}/{len(results)} ({100*n_sig_anec/len(results):.1f}%)")
    
    # Best configurations by different metrics
    if n_neg_energy > 0:
        best_energy_idx = np.argmin(data['energy_density'][neg_energy_mask])
        best_energy = [r for r in results if r['energy_density'] < 0][best_energy_idx]
        print(f"\nMost negative energy density:")
        print(f"  Spacing: {best_energy['spacing_nm']:.0f} nm")
        print(f"  ε: {best_energy['permittivity']:.1f}, μ: {best_energy['permeability']:.1f}")
        print(f"  Layers: {best_energy['n_layers']}")
        print(f"  Energy density: {best_energy['energy_density']:.2e} J/m³")
        print(f"  Amplification: {best_energy['amplification_factor']:.1f}x")
    
    if n_sig_anec > 0:
        anec_vals = np.abs([r['anec_integral'] for r in results if r['significant_anec']])
        best_anec_idx = np.argmax(anec_vals)
        best_anec = [r for r in results if r['significant_anec']][best_anec_idx]
        print(f"\nLargest ANEC violation:")
        print(f"  Spacing: {best_anec['spacing_nm']:.0f} nm")
        print(f"  ε: {best_anec['permittivity']:.1f}, μ: {best_anec['permeability']:.1f}")
        print(f"  Layers: {best_anec['n_layers']}")
        print(f"  ANEC integral: {best_anec['anec_integral']:.2e}")
        print(f"  Amplification: {best_anec['amplification_factor']:.1f}x")
    
    # Highest amplification
    best_amp_idx = np.argmax(data['amplification_factor'])
    best_amp = results[best_amp_idx]
    print(f"\nHighest force amplification:")
    print(f"  Spacing: {best_amp['spacing_nm']:.0f} nm")
    print(f"  ε: {best_amp['permittivity']:.1f}, μ: {best_amp['permeability']:.1f}")
    print(f"  Layers: {best_amp['n_layers']}")
    print(f"  Amplification: {best_amp['amplification_factor']:.1f}x")
    
    return data

def create_sweep_visualizations(results, data):
    """Create visualization plots for parameter sweep results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Metamaterial Casimir Parameter Sweep Results', fontsize=16)
    
    # 1. Energy density vs spacing (colored by permittivity)
    ax = axes[0, 0]
    scatter = ax.scatter(data['spacing_nm'], data['energy_density'], 
                        c=data['permittivity'], cmap='RdBu', alpha=0.7)
    ax.set_xlabel('Spacing (nm)')
    ax.set_ylabel('Energy Density (J/m³)')
    ax.set_yscale('symlog')
    ax.set_title('Energy Density vs Spacing')
    plt.colorbar(scatter, ax=ax, label='Permittivity')
    ax.grid(True, alpha=0.3)
    
    # 2. ANEC integral vs amplification factor
    ax = axes[0, 1]
    anec_abs = np.abs(data['anec_integral'])
    mask = anec_abs > 0
    scatter = ax.scatter(data['amplification_factor'][mask], anec_abs[mask],
                        c=data['n_layers'][mask], cmap='viridis', alpha=0.7)
    ax.set_xlabel('Force Amplification Factor')
    ax.set_ylabel('|ANEC Integral|')
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_title('ANEC vs Amplification')
    plt.colorbar(scatter, ax=ax, label='Number of Layers')
    ax.grid(True, alpha=0.3)
    
    # 3. Permittivity vs Permeability (colored by energy density)
    ax = axes[0, 2]
    scatter = ax.scatter(data['permittivity'], data['permeability'],
                        c=data['energy_density'], cmap='coolwarm', alpha=0.7)
    ax.set_xlabel('Permittivity')
    ax.set_ylabel('Permeability')
    ax.set_title('ε-μ Parameter Space')
    plt.colorbar(scatter, ax=ax, label='Energy Density (J/m³)')
    ax.grid(True, alpha=0.3)
    
    # 4. Amplification vs layers (grouped by spacing)
    ax = axes[1, 0]
    unique_spacings = np.unique(data['spacing_nm'])
    for spacing in unique_spacings:
        mask = data['spacing_nm'] == spacing
        ax.scatter(data['n_layers'][mask], data['amplification_factor'][mask],
                  label=f'{spacing:.0f} nm', alpha=0.7)
    ax.set_xlabel('Number of Layers')
    ax.set_ylabel('Force Amplification Factor')
    ax.set_yscale('log')
    ax.set_title('Amplification vs Layer Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Energy density distribution
    ax = axes[1, 1]
    neg_energies = data['energy_density'][data['energy_density'] < 0]
    pos_energies = data['energy_density'][data['energy_density'] > 0]
    
    if len(neg_energies) > 0:
        ax.hist(neg_energies, bins=30, alpha=0.7, label='Negative', color='red')
    if len(pos_energies) > 0:
        ax.hist(pos_energies, bins=30, alpha=0.7, label='Positive', color='blue')
    
    ax.set_xlabel('Energy Density (J/m³)')
    ax.set_ylabel('Count')
    ax.set_xscale('symlog')
    ax.set_title('Energy Density Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 6. Feasibility matrix
    ax = axes[1, 2]
    
    # Create feasibility matrix
    eps_unique = np.unique(data['permittivity'])
    mu_unique = np.unique(data['permeability'])
    
    feasibility_matrix = np.zeros((len(eps_unique), len(mu_unique)))
    
    for i, eps in enumerate(eps_unique):
        for j, mu in enumerate(mu_unique):
            mask = (data['permittivity'] == eps) & (data['permeability'] == mu)
            if np.any(mask):
                # Count negative energy configurations
                neg_count = np.sum(data['energy_density'][mask] < 0)
                total_count = np.sum(mask)
                feasibility_matrix[i, j] = neg_count / total_count
    
    im = ax.imshow(feasibility_matrix, cmap='RdYlBu', aspect='auto', origin='lower')
    ax.set_xticks(range(len(mu_unique)))
    ax.set_xticklabels([f'{mu:.1f}' for mu in mu_unique])
    ax.set_yticks(range(len(eps_unique)))
    ax.set_yticklabels([f'{eps:.1f}' for eps in eps_unique])
    ax.set_xlabel('Permeability μ')
    ax.set_ylabel('Permittivity ε')
    ax.set_title('Negative Energy Feasibility')
    plt.colorbar(im, ax=ax, label='Fraction with Negative Energy')
    
    plt.tight_layout()
    plt.savefig('results/metamaterial_parameter_sweep.png', dpi=150, bbox_inches='tight')
    print("Parameter sweep visualizations saved to results/metamaterial_parameter_sweep.png")

def main():
    """Main function to run parameter sweep and analysis."""
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Run parameter sweep
    print("Running metamaterial Casimir parameter sweep...")
    results = sweep_metamaterial_parameters()
    
    # Analyze results
    data = analyze_sweep_results(results)
    
    # Create visualizations
    create_sweep_visualizations(results, data)
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'results/metamaterial_sweep_{timestamp}.json'
      # Convert numpy types to native Python for JSON serialization
    json_results = []
    for r in results:
        json_r = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                json_r[k] = v.tolist()
            elif isinstance(v, (np.integer, np.floating)):
                json_r[k] = float(v)
            elif isinstance(v, np.bool_):
                json_r[k] = bool(v)
            elif isinstance(v, complex):
                json_r[k] = {'real': float(v.real), 'imag': float(v.imag)}
            else:
                json_r[k] = v
        json_results.append(json_r)
    
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': timestamp,
                'total_configurations': len(results),
                'description': 'Metamaterial Casimir parameter sweep results'
            },
            'results': json_results
        }, f, indent=2)
    
    print(f"Results saved to {output_file}")
    print("\nMetamaterial parameter sweep complete!")

if __name__ == "__main__":
    main()
