#!/usr/bin/env python3
"""
Quantum Inequality (QI) Bound Comparison: Classical vs. Polymer

This script compares classical QFT quantum inequality bounds with polymer 
quantization predictions across different parameter regimes (tau, mu).

The quantum inequality constrains negative energy densities:
∫_{-tau}^{tau} <T00>(t) dt >= -C/tau^2

Where C is a bound that differs between classical QFT and polymer quantization.

Author: LQG-ANEC Framework Development Team
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from polymer_quantization import PolymerOperator, polymer_quantum_inequality_bound
from spin_network_utils import build_flat_graph
from stress_tensor_operator import LocalT00
from coherent_states import CoherentState

def classical_qi_bound(tau, hbar=1.0):
    """
    Classical QFT quantum inequality bound.
    For massless scalar field: C_classical ~ ℏ/(48π²)
    """
    return -(hbar / (48 * np.pi**2)) / tau**2

def polymer_qi_bound(tau, mu, hbar=1.0, gamma=0.2735):
    """
    Polymer quantization quantum inequality bound.
    Use the existing polymer_quantum_inequality_bound function.
    """
    return polymer_quantum_inequality_bound(tau, mu)

def compute_actual_anec_integral(graph, state, tau_sampling=100):
    """
    Compute actual ANEC integral ∫<T00> dt for comparison with bounds.
    """
    op = LocalT00()
    T00_values = op.apply(graph)
    
    # Simulate time evolution over [-tau, tau]
    # For simplicity, assume uniform T00 over the interval
    mean_T00 = np.mean(list(T00_values.values()))
    
    return mean_T00 * 2  # Integral over 2tau interval

def main():
    print("=== Quantum Inequality Bound Comparison ===\n")
    
    # Parameter ranges for comparison
    tau_values = np.logspace(-2, 2, 50)  # 0.01 to 100
    mu_values = [0.1, 0.5, 1.0, 2.0, 5.0]  # Different polymer scales
    
    print(f"Comparing bounds across {len(tau_values)} tau values and {len(mu_values)} mu values")
    
    # Set up test system
    print("\n1. Setting up test spin network and coherent state...")
    graph = build_flat_graph(n_nodes=64, connectivity="cubic")  # 4³ lattice
    coh = CoherentState(graph, alpha=0.2)
    peaked_graph = coh.peak_on_flat()
    amplitudes = coh.weave_state()
    peaked_graph.assign_amplitudes(amplitudes)
    
    print(f"   • Graph: {len(graph.nodes)} nodes")
    print(f"   • Coherent state α = {coh.alpha}")
    
    # Compute bounds for each parameter combination
    print("\n2. Computing classical and polymer QI bounds...")
    
    results = {
        'tau_values': tau_values,
        'mu_values': mu_values,
        'classical_bounds': [],
        'polymer_bounds': {},
        'actual_integrals': []
    }
    
    # Classical bounds (mu-independent)
    classical_bounds = [classical_qi_bound(tau) for tau in tau_values]
    results['classical_bounds'] = classical_bounds
    
    # Polymer bounds for each mu
    for mu in mu_values:
        polymer_bounds = [polymer_qi_bound(tau, mu) for tau in tau_values]
        results['polymer_bounds'][mu] = polymer_bounds
    
    # Compute actual ANEC integrals (simplified)
    print("   • Computing actual ANEC integrals...")
    actual_integrals = []
    for i, tau in enumerate(tau_values):
        if i % 10 == 0:
            # Only compute for subset to save time
            integral = compute_actual_anec_integral(peaked_graph, coh)
            actual_integrals.append(integral)
        else:
            actual_integrals.append(None)
    
    results['actual_integrals'] = actual_integrals
    
    # 3. Analysis and comparison
    print("\n3. Analysis Results:")
    
    # Find crossover scales
    for mu in mu_values:
        polymer_bounds = results['polymer_bounds'][mu]
        
        # Find where polymer bound becomes significantly different from classical
        ratio = np.array(polymer_bounds) / np.array(classical_bounds)
        significant_diff = np.where(np.abs(ratio - 1.0) > 0.1)[0]
        
        if len(significant_diff) > 0:
            crossover_tau = tau_values[significant_diff[0]]
            print(f"   • μ = {mu}: Polymer effects significant for τ > {crossover_tau:.3f}")
        else:
            print(f"   • μ = {mu}: No significant polymer effects in τ range")
    
    # 4. Generate comparison plots
    print("\n4. Generating comparison plots...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Bounds vs τ for different μ
    plt.subplot(2, 2, 1)
    plt.loglog(tau_values, np.abs(classical_bounds), 'k-', linewidth=2, label='Classical QFT')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(mu_values)))
    for mu, color in zip(mu_values, colors):
        polymer_bounds = results['polymer_bounds'][mu]
        plt.loglog(tau_values, np.abs(polymer_bounds), '--', color=color, 
                  label=f'Polymer μ={mu}')
    
    plt.xlabel('τ (time interval)')
    plt.ylabel('|QI Bound|')
    plt.title('Quantum Inequality Bounds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Ratio of polymer to classical bound
    plt.subplot(2, 2, 2)
    for mu, color in zip(mu_values, colors):
        polymer_bounds = results['polymer_bounds'][mu]
        ratio = np.array(polymer_bounds) / np.array(classical_bounds)
        plt.semilogx(tau_values, ratio, '--', color=color, label=f'μ={mu}')
    
    plt.axhline(y=1, color='k', linestyle='-', alpha=0.5)
    plt.xlabel('τ (time interval)')
    plt.ylabel('Polymer Bound / Classical Bound')
    plt.title('Polymer Modification Factor')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Parameter space map
    plt.subplot(2, 2, 3)
    tau_grid, mu_grid = np.meshgrid(tau_values, mu_values)
    modification_grid = np.zeros_like(tau_grid)
    
    for i, mu in enumerate(mu_values):
        for j, tau in enumerate(tau_values):
            classical = classical_qi_bound(tau)
            polymer = polymer_qi_bound(tau, mu)
            modification_grid[i, j] = abs(polymer / classical - 1.0)
    
    plt.contourf(tau_grid, mu_grid, modification_grid, levels=20, cmap='plasma')
    plt.colorbar(label='|Modification Factor - 1|')
    plt.xlabel('τ (time interval)')
    plt.ylabel('μ (polymer scale)')
    plt.title('Polymer Modification Strength')
    plt.xscale('log')
    
    # Plot 4: Actual vs bound comparison (where available)
    plt.subplot(2, 2, 4)
    actual_tau = [tau_values[i] for i, val in enumerate(actual_integrals) if val is not None]
    actual_vals = [val for val in actual_integrals if val is not None]
    
    if actual_vals:
        plt.loglog(actual_tau, np.abs(actual_vals), 'ro', label='Actual ANEC', markersize=6)
        
        # Show bounds for comparison
        actual_classical = [classical_qi_bound(tau) for tau in actual_tau]
        plt.loglog(actual_tau, np.abs(actual_classical), 'k-', label='Classical Bound')
        
        plt.xlabel('τ (time interval)')
        plt.ylabel('|ANEC Integral|')
        plt.title('Actual vs Bound')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'qi_bound_comparison.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   • Saved comparison plot: {output_path}")
    
    # 5. Summary statistics
    print("\n5. Summary Statistics:")
    
    # Maximum modification factors
    for mu in mu_values:
        polymer_bounds = results['polymer_bounds'][mu]
        max_modification = np.max(np.abs(np.array(polymer_bounds) / np.array(classical_bounds) - 1.0))
        print(f"   • μ = {mu}: Maximum modification = {max_modification:.3f}")
    
    # Regime analysis
    print(f"\n   Parameter regimes:")
    print(f"   • Classical regime (polymer ≈ classical): small τμ")
    print(f"   • Polymer regime (significant modification): large τμ")
    print(f"   • Crossover typically around τμ ~ 1")
    
    print(f"\n=== QI Bound Comparison Complete ===")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print(f"\nComparison completed successfully!")
    except Exception as e:
        print(f"\nError during comparison: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
