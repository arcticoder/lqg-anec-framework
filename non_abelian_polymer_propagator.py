#!/usr/bin/env python3
"""
Non-Abelian Polymer Gauge Propagator with Full Color Structure
==============================================================

Implements the complete non-Abelian tensor and color structure for polymer-modified
gauge field propagators, including explicit instanton sector integration.

Key Features:
- Full SU(N) color structure with adjoint representation indices
- Transverse polymer-modified propagator with mass regularization
- Explicit exponential instanton formula with polymer corrections
- Integration with spin-foam/ANEC pipeline and uncertainty quantification
- Numerical validation across parameter ranges
"""

import numpy as np
import scipy.special
import scipy.integrate
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Optional, Union
import warnings
from dataclasses import dataclass
from pathlib import Path

@dataclass
class NonAbelianConfig:
    """Configuration for non-Abelian polymer propagator calculations."""
    mu_g: float = 0.15           # Gauge polymer parameter
    m_g: float = 0.1             # Gauge mass parameter
    N_colors: int = 3            # Number of colors (SU(3) by default)
    k_max: float = 10.0          # Maximum momentum for integration
    n_points: int = 1000         # Number of points for numerical integration
    S_inst: float = 8.0 * np.pi**2  # Instanton action (8π²/g²)
    Phi_inst: float = 2.0 * np.pi   # Instanton phase
    hbar: float = 1.0            # Reduced Planck constant (natural units)
    
class NonAbelianPolymerPropagator:
    """
    Complete non-Abelian polymer gauge propagator with color structure.
    
    Implements the full tensor structure:
    D̃ᵃᵇ_μν(k) = δᵃᵇ * (η_μν - k_μk_ν/k²)/μ_g² * sin²(μ_g√(k²+m_g²))/(k²+m_g²)
    """
    
    def __init__(self, config: NonAbelianConfig):
        self.config = config
        self.results = {}
        
    def color_structure(self, a: int, b: int) -> float:
        """
        Color structure factor δᵃᵇ for SU(N) gauge theory.
        
        Args:
            a, b: Color indices (0 to N_colors²-1 for adjoint representation)
            
        Returns:
            Kronecker delta δᵃᵇ
        """
        return 1.0 if a == b else 0.0
    
    def transverse_projector(self, k: np.ndarray, mu: int, nu: int) -> float:
        """
        Transverse projector (η_μν - k_μk_ν/k²).
        
        Args:
            k: 4-momentum vector [k₀, k₁, k₂, k₃]
            mu, nu: Lorentz indices (0-3)
            
        Returns:
            Transverse projector component
        """
        k_squared = np.sum(k**2)
        if k_squared < 1e-12:
            # Handle k=0 case
            return 1.0 if mu == nu else 0.0
            
        eta = np.diag([1, -1, -1, -1])  # Minkowski metric
        return eta[mu, nu] - k[mu] * k[nu] / k_squared
    
    def polymer_factor(self, k: np.ndarray) -> float:
        """
        Polymer modification factor sin²(μ_g√(k²+m_g²))/(k²+m_g²).
        
        Args:
            k: 4-momentum vector
            
        Returns:
            Polymer modification factor
        """
        k_squared = np.sum(k**2)
        k_eff = np.sqrt(k_squared + self.config.m_g**2)
        
        if k_eff < 1e-12:
            # μ_g → 0 limit: sin²(x)/x² → 1
            return 1.0 / self.config.m_g**2
            
        sin_arg = self.config.mu_g * k_eff
        sin_factor = np.sin(sin_arg)**2
        
        return sin_factor / (k_squared + self.config.m_g**2)
    
    def full_propagator(self, k: np.ndarray, a: int, b: int, mu: int, nu: int) -> float:
        """
        Complete non-Abelian polymer propagator D̃ᵃᵇ_μν(k).
        
        Args:
            k: 4-momentum vector
            a, b: Color indices
            mu, nu: Lorentz indices
            
        Returns:
            Full propagator component
        """
        color_factor = self.color_structure(a, b)
        transverse = self.transverse_projector(k, mu, nu)
        polymer = self.polymer_factor(k)
        mass_factor = 1.0 / self.config.mu_g**2
        
        return color_factor * transverse * mass_factor * polymer
    
    def instanton_action_polymer(self, phi_inst: float) -> float:
        """
        Polymer-modified instanton action with exponential formula.
        
        Args:
            phi_inst: Instanton phase
            
        Returns:
            Polymer-corrected instanton action
        """
        sin_factor = np.sin(self.config.mu_g * phi_inst) / self.config.mu_g
        return self.config.S_inst * sin_factor
    
    def instanton_amplitude(self, phi_inst: Optional[float] = None) -> float:
        """
        Complete instanton amplitude with polymer corrections.
        
        Γ_instanton^poly ∝ exp[-S_inst/ℏ * sin(μ_g Φ_inst)/μ_g]
        
        Args:
            phi_inst: Instanton phase (uses config default if None)
            
        Returns:
            Instanton amplitude
        """
        if phi_inst is None:
            phi_inst = self.config.Phi_inst
            
        polymer_action = self.instanton_action_polymer(phi_inst)
        exponent = -polymer_action / self.config.hbar
        
        return np.exp(exponent)
    
    def classical_limit_test(self) -> Dict[str, float]:
        """
        Test μ_g → 0 classical limit recovery.
        
        Returns:
            Dictionary with classical limit test results
        """
        # Test momentum
        k_test = np.array([1.0, 0.5, 0.3, 0.2])
        
        # Small μ_g values for limit test
        mu_values = [0.1, 0.05, 0.01, 0.005, 0.001]
        propagator_values = []
        
        for mu_g in mu_values:
            old_mu = self.config.mu_g
            self.config.mu_g = mu_g
            prop = self.full_propagator(k_test, 0, 0, 1, 1)  # Test component
            propagator_values.append(prop)
            self.config.mu_g = old_mu
        
        # Classical limit (μ_g → 0)
        k_squared = np.sum(k_test**2)
        classical_value = (1.0 - k_test[1]**2 / k_squared) / (k_squared + self.config.m_g**2)
        
        # Check convergence to classical limit
        final_ratio = propagator_values[-1] / classical_value if classical_value != 0 else np.inf
        
        return {
            'mu_values': mu_values,
            'propagator_values': propagator_values,
            'classical_value': classical_value,
            'convergence_ratio': final_ratio,
            'classical_limit_recovered': abs(final_ratio - 1.0) < 0.01
        }
    
    def momentum_integration(self, k_range: Tuple[float, float] = (0.1, 10.0)) -> Dict[str, np.ndarray]:
        """
        Numerical integration over momentum space.
        
        Args:
            k_range: Integration range for momentum magnitude
            
        Returns:
            Integration results
        """
        k_min, k_max = k_range
        k_values = np.linspace(k_min, k_max, self.config.n_points)
        
        # Compute propagator for different momentum values
        propagator_11 = []  # (1,1) Lorentz component
        propagator_22 = []  # (2,2) Lorentz component
        instanton_weights = []
        
        for k_mag in k_values:
            k_vec = np.array([k_mag, 0, 0, 0])  # Time-like momentum
            
            prop_11 = self.full_propagator(k_vec, 0, 0, 1, 1)
            prop_22 = self.full_propagator(k_vec, 0, 0, 2, 2)
            
            # Instanton contribution at this momentum scale
            phi_eff = self.config.Phi_inst * k_mag / k_max  # Scale-dependent phase
            inst_weight = self.instanton_amplitude(phi_eff)
            
            propagator_11.append(prop_11)
            propagator_22.append(prop_22)
            instanton_weights.append(inst_weight)
        
        return {
            'k_values': k_values,
            'propagator_11': np.array(propagator_11),
            'propagator_22': np.array(propagator_22),
            'instanton_weights': np.array(instanton_weights),
            'total_propagator': np.array(propagator_11) + np.array(instanton_weights)
        }
    
    def uncertainty_quantification(self, n_samples: int = 1000) -> Dict[str, np.ndarray]:
        """
        Uncertainty quantification for polymer parameters.
        
        Args:
            n_samples: Number of Monte Carlo samples
            
        Returns:
            UQ statistics
        """
        # Parameter uncertainties (assumed Gaussian)
        mu_g_std = 0.05 * self.config.mu_g
        m_g_std = 0.1 * self.config.m_g
        
        # Monte Carlo sampling
        mu_samples = np.random.normal(self.config.mu_g, mu_g_std, n_samples)
        m_samples = np.random.normal(self.config.m_g, m_g_std, n_samples)
        
        # Test momentum
        k_test = np.array([2.0, 1.0, 0.5, 0.3])
        
        propagator_samples = []
        instanton_samples = []
        
        for i in range(n_samples):
            # Temporarily modify parameters
            old_mu, old_m = self.config.mu_g, self.config.m_g
            self.config.mu_g = abs(mu_samples[i])  # Ensure positive
            self.config.m_g = abs(m_samples[i])    # Ensure positive
            
            # Compute propagator and instanton amplitude
            prop = self.full_propagator(k_test, 0, 0, 1, 1)
            inst = self.instanton_amplitude()
            
            propagator_samples.append(prop)
            instanton_samples.append(inst)
            
            # Restore parameters
            self.config.mu_g, self.config.m_g = old_mu, old_m
        
        propagator_samples = np.array(propagator_samples)
        instanton_samples = np.array(instanton_samples)
        
        return {
            'propagator_mean': np.mean(propagator_samples),
            'propagator_std': np.std(propagator_samples),
            'propagator_95_ci': np.percentile(propagator_samples, [2.5, 97.5]),
            'instanton_mean': np.mean(instanton_samples),
            'instanton_std': np.std(instanton_samples),
            'instanton_95_ci': np.percentile(instanton_samples, [2.5, 97.5]),
            'samples': {
                'propagator': propagator_samples,
                'instanton': instanton_samples
            }
        }
    
    def spin_foam_integration(self, n_time_steps: int = 100) -> Dict[str, np.ndarray]:
        """
        Integration with spin-foam evolution including ANEC monitoring.
        
        Args:
            n_time_steps: Number of time evolution steps
            
        Returns:
            Time evolution results with ANEC violations
        """
        dt = 0.1
        times = np.arange(0, n_time_steps * dt, dt)
        
        # Initialize field configurations
        field_values = []
        anec_violations = []
        propagator_evolution = []
        
        # Base momentum for evolution
        k_base = np.array([1.0, 0.0, 0.0, 0.0])
        
        for i, t in enumerate(times):
            # Time-dependent momentum (simulated evolution)
            k_t = k_base * (1.0 + 0.1 * np.sin(t))
            
            # Compute propagator at this time
            prop = self.full_propagator(k_t, 0, 0, 1, 1)
            propagator_evolution.append(prop)
            
            # Simulate field value evolution with polymer corrections
            field = prop * np.exp(-0.05 * t) * (1.0 + 0.2 * np.cos(2.0 * t))
            field_values.append(field)
            
            # ANEC violation estimate (simplified)
            # Real implementation would compute ∫T₀₀ρ dt over null geodesics
            if i > 0:
                field_derivative = (field_values[i] - field_values[i-1]) / dt
                anec_violation = field_derivative**2 - 0.5 * field**2  # Simplified stress tensor
                anec_violations.append(anec_violation)
            else:
                anec_violations.append(0.0)
        
        return {
            'times': times,
            'field_values': np.array(field_values),
            'propagator_evolution': np.array(propagator_evolution),
            'anec_violations': np.array(anec_violations),
            'anec_violation_integral': np.trapz(anec_violations, times[:len(anec_violations)]) if len(anec_violations) > 1 else 0.0
        }
    
    def run_comprehensive_analysis(self) -> Dict:
        """
        Run complete analysis including all components.
        
        Returns:
            Comprehensive results dictionary
        """
        print("Running Non-Abelian Polymer Propagator Analysis...")
        
        # 1. Classical limit test
        print("1. Testing classical limit recovery...")
        classical_results = self.classical_limit_test()
        
        # 2. Momentum integration
        print("2. Performing momentum space integration...")
        momentum_results = self.momentum_integration()
        
        # 3. Uncertainty quantification
        print("3. Running uncertainty quantification...")
        uq_results = self.uncertainty_quantification()
        
        # 4. Spin-foam integration
        print("4. Integrating with spin-foam evolution...")
        spinfoam_results = self.spin_foam_integration()
        
        # 5. Instanton sector analysis
        print("5. Analyzing instanton sector...")
        phi_values = np.linspace(0, 4*np.pi, 100)
        instanton_amplitudes = [self.instanton_amplitude(phi) for phi in phi_values]
        
        self.results = {
            'classical_limit': classical_results,
            'momentum_integration': momentum_results,
            'uncertainty_quantification': uq_results,
            'spin_foam_evolution': spinfoam_results,
            'instanton_analysis': {
                'phi_values': phi_values,
                'amplitudes': instanton_amplitudes
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
    
    def export_results(self, filename: str = "non_abelian_polymer_results.json"):
        """Export results to JSON file."""
        if not self.results:
            print("No results to export. Run analysis first.")
            return
            
        # Convert numpy arrays to lists for JSON serialization
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
    
    def plot_results(self):
        """Generate comprehensive plots of results."""
        if not self.results:
            print("No results to plot. Run analysis first.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Non-Abelian Polymer Propagator Analysis', fontsize=16)
        
        # 1. Classical limit convergence
        ax = axes[0, 0]
        classical_data = self.results['classical_limit']
        ax.semilogx(classical_data['mu_values'], classical_data['propagator_values'], 'bo-')
        ax.axhline(classical_data['classical_value'], color='r', linestyle='--', label='Classical limit')
        ax.set_xlabel('μ_g')
        ax.set_ylabel('Propagator value')
        ax.set_title('Classical Limit Recovery')
        ax.legend()
        ax.grid(True)
        
        # 2. Momentum space propagator
        ax = axes[0, 1]
        mom_data = self.results['momentum_integration']
        ax.loglog(mom_data['k_values'], abs(mom_data['propagator_11']), 'b-', label='D₁₁')
        ax.loglog(mom_data['k_values'], abs(mom_data['propagator_22']), 'r-', label='D₂₂')
        ax.set_xlabel('k')
        ax.set_ylabel('|Propagator|')
        ax.set_title('Momentum Space Propagator')
        ax.legend()
        ax.grid(True)
        
        # 3. Instanton amplitudes
        ax = axes[0, 2]
        inst_data = self.results['instanton_analysis']
        ax.plot(inst_data['phi_values'], inst_data['amplitudes'], 'g-', linewidth=2)
        ax.set_xlabel('Φ_inst')
        ax.set_ylabel('Instanton amplitude')
        ax.set_title('Instanton Sector')
        ax.grid(True)
        
        # 4. Uncertainty quantification
        ax = axes[1, 0]
        uq_data = self.results['uncertainty_quantification']
        ax.hist(uq_data['samples']['propagator'], bins=50, alpha=0.7, label='Propagator')
        ax.axvline(uq_data['propagator_mean'], color='r', linestyle='-', label='Mean')
        ax.axvline(uq_data['propagator_95_ci'][0], color='r', linestyle='--', alpha=0.7)
        ax.axvline(uq_data['propagator_95_ci'][1], color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Propagator value')
        ax.set_ylabel('Frequency')
        ax.set_title('Uncertainty Quantification')
        ax.legend()
        ax.grid(True)
        
        # 5. Spin-foam evolution
        ax = axes[1, 1]
        sf_data = self.results['spin_foam_evolution']
        ax.plot(sf_data['times'], sf_data['field_values'], 'b-', label='Field evolution')
        ax.plot(sf_data['times'], sf_data['propagator_evolution'], 'r--', label='Propagator')
        ax.set_xlabel('Time')
        ax.set_ylabel('Field/Propagator')
        ax.set_title('Spin-Foam Evolution')
        ax.legend()
        ax.grid(True)
        
        # 6. ANEC violations
        ax = axes[1, 2]
        if len(sf_data['anec_violations']) > 1:
            ax.plot(sf_data['times'][1:], sf_data['anec_violations'], 'purple', linewidth=2)
        ax.axhline(0, color='k', linestyle='-', alpha=0.3)
        ax.set_xlabel('Time')
        ax.set_ylabel('ANEC violation')
        ax.set_title(f'ANEC Violations (∫ = {sf_data["anec_violation_integral"]:.3f})')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig('non_abelian_polymer_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function."""
    # Configuration
    config = NonAbelianConfig(
        mu_g=0.15,
        m_g=0.1,
        N_colors=3,
        S_inst=8.0 * np.pi**2,
        Phi_inst=2.0 * np.pi
    )
    
    # Initialize and run analysis
    propagator = NonAbelianPolymerPropagator(config)
    results = propagator.run_comprehensive_analysis()
    
    # Export results
    propagator.export_results("non_abelian_polymer_results.json")
    
    # Generate plots
    propagator.plot_results()
    
    # Print summary
    print("\n" + "="*80)
    print("NON-ABELIAN POLYMER PROPAGATOR ANALYSIS COMPLETE")
    print("="*80)
    
    classical_ok = results['classical_limit']['classical_limit_recovered']
    anec_integral = results['spin_foam_evolution']['anec_violation_integral']
    prop_mean = results['uncertainty_quantification']['propagator_mean']
    prop_std = results['uncertainty_quantification']['propagator_std']
    
    print(f"Classical limit recovery: {'✓ PASS' if classical_ok else '✗ FAIL'}")
    print(f"ANEC violation integral: {anec_integral:.4f}")
    print(f"Propagator (mean ± std): {prop_mean:.4f} ± {prop_std:.4f}")
    print(f"Configuration: μ_g = {config.mu_g}, m_g = {config.m_g}, N = {config.N_colors}")
    
    return results

if __name__ == "__main__":
    results = main()
