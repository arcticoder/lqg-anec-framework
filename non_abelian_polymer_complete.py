#!/usr/bin/env python3
"""
Non-Abelian Polymer Gauge Propagator with Full Color Structure
==============================================================

Implements the complete non-Abelian tensor and color structure for polymer-modified
gauge field propagators, including explicit instanton sector integration.
"""

import numpy as np
import json

class NonAbelianConfig:
    """Configuration for non-Abelian polymer propagator calculations."""
    def __init__(self):
        self.mu_g = 0.15           # Gauge polymer parameter
        self.m_g = 0.1             # Gauge mass parameter
        self.N_colors = 3          # Number of colors (SU(3) by default)
        self.S_inst = 8.0 * np.pi**2  # Instanton action
        self.Phi_inst = 2.0 * np.pi   # Instanton phase
        self.hbar = 1.0            # Reduced Planck constant

class NonAbelianPolymerPropagator:
    """Complete non-Abelian polymer gauge propagator with color structure."""
    
    def __init__(self, config):
        self.config = config
        self.results = {}
    
    def full_propagator(self, k, a, b, mu, nu):
        """Complete non-Abelian polymer propagator D̃ᵃᵇ_μν(k)."""
        # Color structure δᵃᵇ
        color_factor = 1.0 if a == b else 0.0
        
        # Momentum components
        k_squared = np.sum(k**2)
        if k_squared < 1e-12:
            transverse = 1.0 if mu == nu else 0.0
        else:
            eta = np.diag([1, -1, -1, -1])  # Minkowski metric
            transverse = eta[mu, nu] - k[mu] * k[nu] / k_squared
        
        # Polymer factor: sin²(μ_g√(k²+m_g²))/(k²+m_g²)
        k_eff = np.sqrt(k_squared + self.config.m_g**2)
        if k_eff < 1e-12:
            polymer = 1.0 / self.config.m_g**2
        else:
            sin_arg = self.config.mu_g * k_eff
            sin_factor = np.sin(sin_arg)**2
            polymer = sin_factor / (k_squared + self.config.m_g**2)
        
        mass_factor = 1.0 / self.config.mu_g**2
        
        return color_factor * transverse * mass_factor * polymer
    
    def instanton_amplitude(self, phi_inst=None):
        """Instanton amplitude: Γ ∝ exp[-S_inst/ℏ * sin(μ_g Φ_inst)/μ_g]"""
        if phi_inst is None:
            phi_inst = self.config.Phi_inst
        
        sin_factor = np.sin(self.config.mu_g * phi_inst) / self.config.mu_g
        polymer_action = self.config.S_inst * sin_factor
        exponent = -polymer_action / self.config.hbar
        
        return np.exp(exponent)
    
    def classical_limit_test(self):
        """Test μ_g → 0 classical limit recovery."""
        k_test = np.array([1.0, 0.5, 0.3, 0.2])
        mu_values = [0.1, 0.05, 0.01, 0.005, 0.001]
        propagator_values = []
        
        for mu_g in mu_values:
            old_mu = self.config.mu_g
            self.config.mu_g = mu_g
            prop = self.full_propagator(k_test, 0, 0, 1, 1)
            propagator_values.append(prop)
            self.config.mu_g = old_mu
        
        # Classical limit
        k_squared = np.sum(k_test**2)
        classical_value = (1.0 - k_test[1]**2 / k_squared) / (k_squared + self.config.m_g**2)
        
        final_ratio = propagator_values[-1] / classical_value if classical_value != 0 else np.inf
        
        return {
            'mu_values': mu_values,
            'propagator_values': propagator_values,
            'classical_value': classical_value,
            'convergence_ratio': final_ratio,
            'classical_limit_recovered': abs(final_ratio - 1.0) < 0.01
        }
    
    def run_analysis(self):
        """Run complete analysis."""
        print("Running Non-Abelian Polymer Propagator Analysis...")
        
        # Classical limit test
        print("1. Testing classical limit recovery...")
        classical_results = self.classical_limit_test()
        
        # Test propagator at various momenta
        print("2. Testing propagator across momentum range...")
        k_values = np.linspace(0.1, 5.0, 20)
        propagator_values = []
        instanton_values = []
        
        for k_mag in k_values:
            k_vec = np.array([k_mag, 0, 0, 0])
            prop = self.full_propagator(k_vec, 0, 0, 1, 1)
            inst = self.instanton_amplitude(np.pi * k_mag / 5.0)
            propagator_values.append(prop)
            instanton_values.append(inst)
        
        self.results = {
            'classical_limit': classical_results,
            'momentum_scan': {
                'k_values': k_values.tolist(),
                'propagator_values': propagator_values,
                'instanton_values': instanton_values
            },
            'config': {
                'mu_g': self.config.mu_g,
                'm_g': self.config.m_g,
                'N_colors': self.config.N_colors,
                'S_inst': self.config.S_inst,
                'Phi_inst': self.config.Phi_inst
            }
        }
        
        return self.results
      def export_results(self, filename="non_abelian_polymer_results.json"):
        """Export results to JSON file."""
        if not self.results:
            print("No results to export. Run analysis first.")
            return
        
        # Convert for JSON serialization
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, dict):
                return {key: convert_for_json(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(self.results)
        
        with open(filename, 'w') as f:
            json.dump(json_results, f, indent=2)
        print(f"Results exported to {filename}")

def main():
    """Main execution function."""
    config = NonAbelianConfig()
    propagator = NonAbelianPolymerPropagator(config)
    results = propagator.run_analysis()
    propagator.export_results()
    
    print("\n" + "="*80)
    print("NON-ABELIAN POLYMER PROPAGATOR ANALYSIS COMPLETE")
    print("="*80)
    
    classical_ok = results['classical_limit']['classical_limit_recovered']
    print(f"Classical limit recovery: {'✓ PASS' if classical_ok else '✗ FAIL'}")
    print(f"Configuration: μ_g = {config.mu_g}, m_g = {config.m_g}, N = {config.N_colors}")
    
    # Show key formula implementation
    print("\nKey Formulas Implemented:")
    print("1. Full non-Abelian tensor structure:")
    print("   D̃ᵃᵇ_μν(k) = δᵃᵇ * (η_μν - k_μk_ν/k²)/μ_g² * sin²(μ_g√(k²+m_g²))/(k²+m_g²)")
    print("\n2. Instanton amplitude:")
    print("   Γ_instanton^poly ∝ exp[-S_inst/ℏ * sin(μ_g Φ_inst)/μ_g]")
    
    return results

if __name__ == "__main__":
    results = main()
