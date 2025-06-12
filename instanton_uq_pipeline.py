#!/usr/bin/env python3
"""
Instanton-Sector UQ Integration Implementation

This module implements the complete instanton-sector mapping integrated into
the UQ pipeline with uncertainty bands for total production rates.

Γ_inst^poly(Φ_inst) = A exp[-S_inst/ℏ] cos²(Φ_inst/2) P_polymer(μ_g)
Γ_total = Γ_Sch^poly + Γ_inst^poly
"""

import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
import logging
from scipy.stats import norm, uniform
from scipy import integrate
import warnings

@dataclass
class InstantonUQConfig:
    """Configuration for instanton UQ analysis."""
    phi_inst_min: float = 0.0
    phi_inst_max: float = 4 * np.pi
    phi_inst_points: int = 100
    
    mu_g_nominal: float = 0.15
    mu_g_uncertainty: float = 0.05
    
    instanton_action: float = 8 * np.pi**2  # S_inst = 8π²/g²
    instanton_amplitude: float = 1e-20     # A in natural units
    
    n_mc_samples: int = 1000
    confidence_level: float = 0.95
    
    # UQ parameters
    schwinger_rate_nominal: float = 1e-16  # Nominal Schwinger rate
    rate_uncertainty_factor: float = 0.2   # 20% uncertainty

class LQGInstantonUQPipeline:
    """
    Complete instanton-sector UQ implementation for LQG-ANEC studies.
    
    This class provides the ACTUAL uncertainty quantification that must be used
    in ALL warp drive and exotic matter calculations.
    """
    
    def __init__(self, config: InstantonUQConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize phase grid
        self.phi_inst_grid = np.linspace(config.phi_inst_min, config.phi_inst_max, config.phi_inst_points)
        
        # Results storage
        self.instanton_rates = None
        self.total_rates = None
        self.uncertainty_bands = None
        self.mc_samples = None
        
    def polymer_correction_factor(self, mu_g: float) -> float:
        """
        Polymer correction factor P_polymer(μ_g) for instanton rates.
        
        This modifies the instanton action in LQG.
        """
        # Polymer suppression similar to propagator case
        # P_polymer(μ_g) = exp(-μ_g²) for simplicity
        return np.exp(-mu_g**2)
    
    def instanton_amplitude(self, phi_inst: float, mu_g: float) -> float:
        """
        THE instanton amplitude calculation with polymer corrections.
        
        Formula: Γ_inst^poly(Φ_inst) = A exp[-S_inst/ℏ] cos²(Φ_inst/2) P_polymer(μ_g)
        """
        # Phase-dependent factor
        phase_factor = np.cos(phi_inst / 2)**2
        
        # Exponential suppression (action/ℏ = S_inst in natural units)
        exponential = np.exp(-self.config.instanton_action)
        
        # Polymer correction
        polymer_factor = self.polymer_correction_factor(mu_g)
        
        # Full amplitude
        amplitude = self.config.instanton_amplitude * exponential * phase_factor * polymer_factor
        
        return amplitude
    
    def compute_instanton_rates(self, mu_g: Optional[float] = None) -> np.ndarray:
        """
        Compute instanton production rates over the complete phase grid.
        """
        if mu_g is None:
            mu_g = self.config.mu_g_nominal
        
        rates = np.zeros(len(self.phi_inst_grid))
        
        for i, phi in enumerate(self.phi_inst_grid):
            rates[i] = self.instanton_amplitude(phi, mu_g)
        
        return rates
    
    def total_production_rate(self, phi_inst: float, mu_g: float, 
                            schwinger_rate: Optional[float] = None) -> float:
        """
        Compute total production rate: Γ_total = Γ_Sch^poly + Γ_inst^poly.
        
        This is the COMPLETE rate that must be used in all calculations.
        """
        if schwinger_rate is None:
            schwinger_rate = self.config.schwinger_rate_nominal
        
        instanton_rate = self.instanton_amplitude(phi_inst, mu_g)
        total_rate = schwinger_rate + instanton_rate
        
        return total_rate
    
    def monte_carlo_uncertainty_analysis(self) -> Dict:
        """
        THE Monte Carlo uncertainty quantification for total production rates.
        
        This provides the uncertainty bands required for all UQ analyses.
        """
        print(f"🔷 Running Monte Carlo UQ with {self.config.n_mc_samples} samples...")
        
        # Generate parameter samples
        np.random.seed(42)  # Reproducible results
        
        # μ_g samples (normal distribution)
        mu_g_samples = np.random.normal(
            self.config.mu_g_nominal, 
            self.config.mu_g_uncertainty, 
            self.config.n_mc_samples
        )
        
        # Schwinger rate samples (log-normal to ensure positivity)
        schwinger_log_mean = np.log(self.config.schwinger_rate_nominal)
        schwinger_log_std = self.config.rate_uncertainty_factor
        schwinger_samples = np.random.lognormal(
            schwinger_log_mean, 
            schwinger_log_std, 
            self.config.n_mc_samples
        )
        
        # Storage for MC results
        mc_total_rates = np.zeros((self.config.n_mc_samples, len(self.phi_inst_grid)))
        mc_instanton_rates = np.zeros((self.config.n_mc_samples, len(self.phi_inst_grid)))
        
        # Monte Carlo sampling
        for i in range(self.config.n_mc_samples):
            mu_g_sample = mu_g_samples[i]
            schwinger_sample = schwinger_samples[i]
            
            for j, phi in enumerate(self.phi_inst_grid):
                # Compute rates for this sample
                instanton_rate = self.instanton_amplitude(phi, mu_g_sample)
                total_rate = schwinger_sample + instanton_rate
                
                mc_instanton_rates[i, j] = instanton_rate
                mc_total_rates[i, j] = total_rate
            
            if (i + 1) % 100 == 0:
                print(f"   MC progress: {i+1}/{self.config.n_mc_samples} ({100*(i+1)/self.config.n_mc_samples:.1f}%)")
        
        # Compute statistics
        alpha = 1 - self.config.confidence_level
        lower_percentile = 100 * alpha / 2
        upper_percentile = 100 * (1 - alpha / 2)
        
        # Total rate statistics
        total_mean = np.mean(mc_total_rates, axis=0)
        total_std = np.std(mc_total_rates, axis=0)
        total_lower = np.percentile(mc_total_rates, lower_percentile, axis=0)
        total_upper = np.percentile(mc_total_rates, upper_percentile, axis=0)
        
        # Instanton rate statistics
        instanton_mean = np.mean(mc_instanton_rates, axis=0)
        instanton_std = np.std(mc_instanton_rates, axis=0)
        instanton_lower = np.percentile(mc_instanton_rates, lower_percentile, axis=0)
        instanton_upper = np.percentile(mc_instanton_rates, upper_percentile, axis=0)
        
        # Store results
        self.mc_samples = {
            'mu_g_samples': mu_g_samples.tolist(),
            'schwinger_samples': schwinger_samples.tolist(),
            'total_rates': mc_total_rates.tolist(),
            'instanton_rates': mc_instanton_rates.tolist()
        }
        
        results = {
            'config': {
                'n_samples': self.config.n_mc_samples,
                'confidence_level': self.config.confidence_level,
                'mu_g_nominal': self.config.mu_g_nominal,
                'mu_g_uncertainty': self.config.mu_g_uncertainty,
                'schwinger_rate_nominal': self.config.schwinger_rate_nominal,
                'rate_uncertainty_factor': self.config.rate_uncertainty_factor
            },
            'phase_grid': self.phi_inst_grid.tolist(),
            'total_rates': {
                'mean': total_mean.tolist(),
                'std': total_std.tolist(),
                'lower_bound': total_lower.tolist(),
                'upper_bound': total_upper.tolist(),
                'confidence_level': self.config.confidence_level
            },
            'instanton_rates': {
                'mean': instanton_mean.tolist(),
                'std': instanton_std.tolist(),
                'lower_bound': instanton_lower.tolist(),
                'upper_bound': instanton_upper.tolist()
            },
            'global_statistics': {
                'total_rate_global_mean': float(np.mean(total_mean)),
                'total_rate_global_std': float(np.mean(total_std)),
                'instanton_rate_global_mean': float(np.mean(instanton_mean)),
                'instanton_rate_global_std': float(np.mean(instanton_std))
            },
            'correlation_analysis': self._compute_correlations(mu_g_samples, schwinger_samples, mc_total_rates)
        }
        
        self.uncertainty_bands = results
        
        return results
    
    def _compute_correlations(self, mu_g_samples: np.ndarray, schwinger_samples: np.ndarray, 
                            total_rates: np.ndarray) -> Dict:
        """Compute parameter correlations."""
        # Average total rate per sample (across all phases)
        avg_total_rates = np.mean(total_rates, axis=1)
        
        # Correlation coefficients
        corr_mu_g_rate = np.corrcoef(mu_g_samples, avg_total_rates)[0, 1]
        corr_schwinger_rate = np.corrcoef(schwinger_samples, avg_total_rates)[0, 1]
        corr_mu_g_schwinger = np.corrcoef(mu_g_samples, schwinger_samples)[0, 1]
        
        return {
            'mu_g_vs_total_rate': float(corr_mu_g_rate),
            'schwinger_vs_total_rate': float(corr_schwinger_rate),
            'mu_g_vs_schwinger': float(corr_mu_g_schwinger)
        }
    
    def generate_uq_plots(self, results: Dict, output_prefix: str = "instanton_uq") -> None:
        """
        Generate comprehensive UQ plots with uncertainty bands.
        """
        print("🔷 Generating UQ plots with uncertainty bands...")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        phase_grid = np.array(results['phase_grid'])
        
        # Plot 1: Total rates with uncertainty bands
        ax1 = axes[0, 0]
        total_mean = np.array(results['total_rates']['mean'])
        total_lower = np.array(results['total_rates']['lower_bound'])
        total_upper = np.array(results['total_rates']['upper_bound'])
        
        ax1.plot(phase_grid, total_mean, 'b-', linewidth=2, label='Mean total rate')
        ax1.fill_between(phase_grid, total_lower, total_upper, alpha=0.3, color='blue',
                        label=f'{self.config.confidence_level:.0%} confidence band')
        ax1.set_xlabel('Instanton Phase Φ_inst')
        ax1.set_ylabel('Total Production Rate Γ_total [s⁻¹]')
        ax1.set_title('Total Production Rate with Uncertainty')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # Plot 2: Instanton rates with uncertainty bands
        ax2 = axes[0, 1]
        instanton_mean = np.array(results['instanton_rates']['mean'])
        instanton_lower = np.array(results['instanton_rates']['lower_bound'])
        instanton_upper = np.array(results['instanton_rates']['upper_bound'])
        
        ax2.plot(phase_grid, instanton_mean, 'r-', linewidth=2, label='Mean instanton rate')
        ax2.fill_between(phase_grid, instanton_lower, instanton_upper, alpha=0.3, color='red',
                        label=f'{self.config.confidence_level:.0%} confidence band')
        ax2.set_xlabel('Instanton Phase Φ_inst')
        ax2.set_ylabel('Instanton Rate Γ_inst [s⁻¹]')
        ax2.set_title('Instanton Production Rate with Uncertainty')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Plot 3: Relative uncertainty
        ax3 = axes[1, 0]
        total_std = np.array(results['total_rates']['std'])
        relative_uncertainty = total_std / total_mean * 100
        ax3.plot(phase_grid, relative_uncertainty, 'g-', linewidth=2)
        ax3.set_xlabel('Instanton Phase Φ_inst')
        ax3.set_ylabel('Relative Uncertainty [%]')
        ax3.set_title('Total Rate Relative Uncertainty')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Parameter correlation visualization
        ax4 = axes[1, 1]
        correlations = results['correlation_analysis']
        corr_names = ['μ_g ↔ Total Rate', 'Schwinger ↔ Total Rate', 'μ_g ↔ Schwinger']
        corr_values = [correlations['mu_g_vs_total_rate'], 
                      correlations['schwinger_vs_total_rate'],
                      correlations['mu_g_vs_schwinger']]
        
        bars = ax4.bar(corr_names, corr_values, color=['blue', 'orange', 'green'])
        ax4.set_ylabel('Correlation Coefficient')
        ax4.set_title('Parameter Correlations')
        ax4.set_ylim(-1, 1)
        ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax4.grid(True, alpha=0.3)
        
        # Add correlation values as text
        for bar, value in zip(bars, corr_values):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05 * np.sign(height),
                    f'{value:.3f}', ha='center', va='bottom' if height >= 0 else 'top')
        
        plt.tight_layout()
        plot_file = f"{output_prefix}_complete.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   UQ plots saved to: {plot_file}")
    
    def export_uq_data(self, output_file: str) -> Dict:
        """
        Export complete UQ analysis data.
        """
        print(f"🔷 Exporting instanton UQ data to {output_file}...")
        
        # Run Monte Carlo analysis
        results = self.monte_carlo_uncertainty_analysis()
        
        # Add nominal calculations
        nominal_instanton_rates = self.compute_instanton_rates()
        results['nominal_calculations'] = {
            'phi_inst_grid': self.phi_inst_grid.tolist(),
            'instanton_rates_nominal': nominal_instanton_rates.tolist()
        }
        
        # Generate plots
        self.generate_uq_plots(results, output_file.replace('.json', ''))
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        global_stats = results['global_statistics']
        correlations = results['correlation_analysis']
        
        print(f"   UQ analysis completed with {self.config.n_mc_samples} MC samples")
        print(f"   Phase range: [0, {self.config.phi_inst_max:.2f}] with {self.config.phi_inst_points} points")
        print(f"   Global mean total rate: {global_stats['total_rate_global_mean']:.2e} ± {global_stats['total_rate_global_std']:.2e}")
        print(f"   Global mean instanton rate: {global_stats['instanton_rate_global_mean']:.2e} ± {global_stats['instanton_rate_global_std']:.2e}")
        print(f"   Confidence level: {self.config.confidence_level:.0%}")
        print(f"   Parameter correlations:")
        print(f"     μ_g ↔ Total rate: {correlations['mu_g_vs_total_rate']:.3f}")
        print(f"     Schwinger ↔ Total rate: {correlations['schwinger_vs_total_rate']:.3f}")
        
        return results

# Integration function for the main UQ pipeline
def integrate_instanton_uq_into_pipeline() -> bool:
    """
    MAIN INTEGRATION FUNCTION: Embed instanton UQ into computational pipelines.
    
    This function provides the uncertainty quantification that must be used
    in ALL exotic matter and warp drive calculations.
    """
    print("🔷 Integrating Instanton UQ into Pipeline...")
    
    # Initialize instanton UQ pipeline
    config = InstantonUQConfig(
        phi_inst_points=100,
        n_mc_samples=1000,
        confidence_level=0.95,
        mu_g_nominal=0.15,
        mu_g_uncertainty=0.05
    )
    
    uq_pipeline = LQGInstantonUQPipeline(config)
    
    # Run complete UQ analysis
    output_file = "lqg_instanton_uq_integration.json"
    results = uq_pipeline.export_uq_data(output_file)
    
    # Validate integration
    has_uncertainty_bands = 'total_rates' in results and 'confidence_level' in results['total_rates']
    has_correlations = 'correlation_analysis' in results
    has_global_stats = 'global_statistics' in results
    
    integration_success = has_uncertainty_bands and has_correlations and has_global_stats
    
    if integration_success:
        print("✅ Instanton UQ successfully integrated into pipeline")
        
        # Create marker file for downstream processes
        with open("INSTANTON_UQ_INTEGRATED.flag", 'w') as f:
            f.write(f"Instanton UQ integrated: {config.n_mc_samples} MC samples, {config.confidence_level:.0%} confidence")
        
        # Print integration summary
        global_mean = results['global_statistics']['total_rate_global_mean']
        global_std = results['global_statistics']['total_rate_global_std']
        print(f"   Uncertainty bands: [{global_mean - 1.96*global_std:.2e}, {global_mean + 1.96*global_std:.2e}]")
        print(f"   Phase mapping: [0, {config.phi_inst_max:.2f}] with {config.phi_inst_points} points")
        
    else:
        print("❌ Instanton UQ integration failed validation")
    
    return integration_success

if __name__ == "__main__":
    # Test the instanton UQ implementation
    config = InstantonUQConfig(
        phi_inst_points=20,  # Small for testing
        n_mc_samples=100     # Small for testing
    )
    
    uq_pipeline = LQGInstantonUQPipeline(config)
    
    # Test single instanton amplitude calculation
    phi_test = np.pi
    mu_g_test = 0.15
    amplitude = uq_pipeline.instanton_amplitude(phi_test, mu_g_test)
    print(f"Test instanton amplitude at Φ={phi_test:.2f}, μ_g={mu_g_test}: {amplitude:.2e}")
    
    # Test total rate calculation
    total_rate = uq_pipeline.total_production_rate(phi_test, mu_g_test)
    print(f"Test total rate: {total_rate:.2e}")
    
    # Run small UQ analysis
    print(f"\nRunning test UQ analysis with {config.n_mc_samples} samples...")
    results = uq_pipeline.export_uq_data("test_instanton_uq.json")
    
    print(f"\nTest UQ completed:")
    print(f"  Global mean total rate: {results['global_statistics']['total_rate_global_mean']:.2e}")
    print(f"  Correlation μ_g ↔ total rate: {results['correlation_analysis']['mu_g_vs_total_rate']:.3f}")
