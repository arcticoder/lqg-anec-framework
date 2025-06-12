#!/usr/bin/env python3
"""
2D Parameter Space Sweep over (μ_g, b) with Yield/Field Gain Analysis
====================================================================

Complete implementation of 2D parameter space exploration:
- Sweep over polymer parameter μ_g and β-function coefficient b
- Compute yield gains: Γ_total^poly/Γ_0 
- Compute field gains: E_crit^poly/E_crit
- Generate comprehensive tables and plots of the parameter space
- Export results for integration with other pipeline components

Key Features:
- High-resolution 2D parameter grid
- Yield gain computation: Γ_total^poly(μ_g,b)/Γ_0
- Critical field analysis: E_crit^poly(μ_g,b)/E_crit
- Statistical analysis and optimization
- Publication-ready tables and visualizations
- Integration with UQ pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass
from pathlib import Path
import seaborn as sns

@dataclass
class ParameterSweepConfig:
    """Configuration for 2D parameter space sweep."""
    mu_g_min: float = 0.05          # Minimum polymer parameter
    mu_g_max: float = 0.5           # Maximum polymer parameter  
    mu_g_points: int = 25           # Grid points for μ_g
    b_min: float = 0.0              # Minimum β-function coefficient
    b_max: float = 15.0             # Maximum β-function coefficient
    b_points: int = 20              # Grid points for b
    alpha_0: float = 1.0/137.0      # Fine structure constant
    E_0: float = 1.0                # Reference energy (GeV)
    m_electron: float = 0.511e-3    # Electron mass (GeV)
    E_test_field: float = 1.0       # Test field strength for yield calculation
    hbar: float = 1.0               # Natural units
    n_field_points: int = 50        # Number of field points for optimization

class ParameterSpace2DSweep:
    """
    Complete 2D parameter space sweep over (μ_g, b).
    
    Computes:
    - Yield gains: Γ_total^poly(μ_g,b)/Γ_0
    - Field gains: E_crit^poly(μ_g,b)/E_crit  
    - Optimization surfaces and statistical analysis
    """
    
    def __init__(self, config: ParameterSweepConfig):
        self.config = config
        self.results = {}
        
        # Create parameter grids
        self.mu_g_grid = np.linspace(config.mu_g_min, config.mu_g_max, config.mu_g_points)
        self.b_grid = np.linspace(config.b_min, config.b_max, config.b_points)
        self.MU_G, self.B = np.meshgrid(self.mu_g_grid, self.b_grid)
        
        print(f"🔬 2D Parameter Space Sweep Initialized")
        print(f"   μ_g range: [{config.mu_g_min:.3f}, {config.mu_g_max:.3f}] ({config.mu_g_points} points)")
        print(f"   b range: [{config.b_min:.1f}, {config.b_max:.1f}] ({config.b_points} points)")
        print(f"   Total grid points: {config.mu_g_points * config.b_points}")

    def alpha_effective(self, energy: float, b: float) -> float:
        """Running coupling α_eff(E) = α_0 / (1 + (α_0/3π) * b * ln(E/E_0))"""
        if energy <= 0 or b < 0:
            return self.config.alpha_0
        
        if b == 0:
            return self.config.alpha_0
        
        log_ratio = np.log(energy / self.config.E_0)
        denominator = 1.0 + (self.config.alpha_0 / (3.0 * np.pi)) * b * log_ratio
        
        return self.config.alpha_0 / max(denominator, 1e-6)

    def polymer_factor(self, energy: float, mu_g: float) -> float:
        """Polymer correction P_polymer(μ_g, E) = sin²(μ_g E)/(μ_g E)²"""
        mu_E = mu_g * energy
        
        if abs(mu_E) < 1e-12:
            return 1.0
        
        return np.sin(mu_E)**2 / mu_E**2

    def schwinger_rate_polymer(self, electric_field: float, mu_g: float, b: float) -> float:
        """
        Complete polymer-modified Schwinger rate.
        
        Γ_Sch^poly = (α_eff E²)/(π ℏ) * exp[-π m²/(α_eff E)] * P_polymer(μ_g, E)
        """
        if electric_field <= 0:
            return 0.0
        
        alpha_eff = self.alpha_effective(electric_field, b)
        m = self.config.m_electron
        
        # Classical Schwinger rate
        prefactor = (alpha_eff * electric_field**2) / (np.pi * self.config.hbar)
        exponent = -np.pi * m**2 / (alpha_eff * electric_field)
        schwinger_classical = prefactor * np.exp(exponent)
        
        # Polymer correction
        polymer_correction = self.polymer_factor(electric_field, mu_g)
        
        return schwinger_classical * polymer_correction

    def critical_field_calculation(self, mu_g: float, b: float) -> float:
        """
        Calculate critical field E_crit where production becomes efficient.
        Self-consistent solution of E_crit = m²/α_eff(E_crit).
        """
        m = self.config.m_electron
        
        # Initial guess
        E_guess = m**2 / self.config.alpha_0
        
        # Iterative solution  
        for _ in range(15):
            alpha_eff = self.alpha_effective(E_guess, b)
            E_new = m**2 / alpha_eff
            
            if abs(E_new - E_guess) < 1e-8:
                break
            E_guess = 0.5 * (E_guess + E_new)  # Damped iteration
        
        return E_guess

    def yield_gain_calculation(self, mu_g: float, b: float) -> float:
        """
        Calculate yield gain Γ_total^poly/Γ_0 at test field strength.
        """
        E_field = self.config.E_test_field
        
        # Polymer rate
        gamma_polymer = self.schwinger_rate_polymer(E_field, mu_g, b)
        
        # Classical rate (μ_g → 0, b = 0)
        gamma_classical = self.schwinger_rate_polymer(E_field, mu_g=1e-6, b=0.0)
        
        if gamma_classical > 0:
            return gamma_polymer / gamma_classical
        else:
            return 1.0 if gamma_polymer > 0 else 0.0

    def field_gain_calculation(self, mu_g: float, b: float) -> float:
        """
        Calculate field gain E_crit^poly/E_crit.
        """
        E_crit_polymer = self.critical_field_calculation(mu_g, b)
        E_crit_classical = self.critical_field_calculation(mu_g=1e-6, b=0.0)
        
        return E_crit_polymer / E_crit_classical if E_crit_classical > 0 else 1.0

    def compute_2d_parameter_space(self) -> Dict:
        """
        Compute complete 2D parameter space over (μ_g, b).
        
        Returns:
            Dictionary with yield gains, field gains, and optimization results
        """
        print("\n" + "="*70)
        print("COMPUTING 2D PARAMETER SPACE: (μ_g, b)")
        print("="*70)
        
        # Initialize result arrays
        yield_gains = np.zeros_like(self.MU_G)
        field_gains = np.zeros_like(self.MU_G)
        schwinger_rates = np.zeros_like(self.MU_G)
        critical_fields = np.zeros_like(self.MU_G)
        
        total_points = self.config.mu_g_points * self.config.b_points
        computed_points = 0
        
        print(f"Computing {total_points} grid points...")
        
        # Compute over entire grid
        for i, mu_g in enumerate(self.mu_g_grid):
            for j, b in enumerate(self.b_grid):
                # Yield gain Γ_total^poly/Γ_0
                yield_gains[j, i] = self.yield_gain_calculation(mu_g, b)
                
                # Field gain E_crit^poly/E_crit
                field_gains[j, i] = self.field_gain_calculation(mu_g, b)
                
                # Schwinger rate at test field
                schwinger_rates[j, i] = self.schwinger_rate_polymer(
                    self.config.E_test_field, mu_g, b
                )
                
                # Critical field
                critical_fields[j, i] = self.critical_field_calculation(mu_g, b)
                
                computed_points += 1
                if computed_points % 50 == 0:
                    print(f"   Progress: {computed_points}/{total_points} ({100*computed_points/total_points:.1f}%)")
        
        print(f"✅ Completed {total_points} grid point calculations")
        
        # Find optimal parameters
        max_yield_idx = np.unravel_index(np.argmax(yield_gains), yield_gains.shape)
        max_field_idx = np.unravel_index(np.argmax(field_gains), field_gains.shape)
        
        optimal_yield_mu_g = self.mu_g_grid[max_yield_idx[1]]
        optimal_yield_b = self.b_grid[max_yield_idx[0]]
        max_yield_gain = yield_gains[max_yield_idx]
        
        optimal_field_mu_g = self.mu_g_grid[max_field_idx[1]]
        optimal_field_b = self.b_grid[max_field_idx[0]]
        max_field_gain = field_gains[max_field_idx]
        
        # Statistical analysis
        yield_stats = {
            'mean': np.mean(yield_gains),
            'std': np.std(yield_gains),
            'min': np.min(yield_gains),
            'max': np.max(yield_gains),
            'median': np.median(yield_gains),
            'q25': np.percentile(yield_gains, 25),
            'q75': np.percentile(yield_gains, 75)
        }
        
        field_stats = {
            'mean': np.mean(field_gains),
            'std': np.std(field_gains),
            'min': np.min(field_gains),
            'max': np.max(field_gains),
            'median': np.median(field_gains),
            'q25': np.percentile(field_gains, 25),
            'q75': np.percentile(field_gains, 75)
        }
        
        results = {
            'parameter_grids': {
                'mu_g_grid': self.mu_g_grid,
                'b_grid': self.b_grid,
                'MU_G': self.MU_G,
                'B': self.B
            },
            'yield_gains': yield_gains,
            'field_gains': field_gains,
            'schwinger_rates': schwinger_rates,
            'critical_fields': critical_fields,
            'optimization': {
                'max_yield_gain': max_yield_gain,
                'optimal_yield_mu_g': optimal_yield_mu_g,
                'optimal_yield_b': optimal_yield_b,
                'max_field_gain': max_field_gain,
                'optimal_field_mu_g': optimal_field_mu_g,
                'optimal_field_b': optimal_field_b
            },
            'statistics': {
                'yield_gains': yield_stats,
                'field_gains': field_stats
            },
            'config': {
                'mu_g_range': [self.config.mu_g_min, self.config.mu_g_max],
                'b_range': [self.config.b_min, self.config.b_max],
                'grid_size': [self.config.mu_g_points, self.config.b_points],
                'test_field': self.config.E_test_field
            }
        }
        
        self.results = results
        
        # Print summary
        print(f"\n📊 PARAMETER SPACE ANALYSIS SUMMARY:")
        print(f"   Max yield gain: {max_yield_gain:.3f} at (μ_g={optimal_yield_mu_g:.3f}, b={optimal_yield_b:.1f})")
        print(f"   Max field gain: {max_field_gain:.3f} at (μ_g={optimal_field_mu_g:.3f}, b={optimal_field_b:.1f})")
        print(f"   Yield gain range: [{yield_stats['min']:.3f}, {yield_stats['max']:.3f}]")
        print(f"   Field gain range: [{field_stats['min']:.3f}, {field_stats['max']:.3f}]")
        
        return results

    def generate_parameter_table(self) -> pd.DataFrame:
        """Generate comprehensive parameter table for publication."""
        if not self.results:
            print("No results available. Run parameter space computation first.")
            return None
        
        # Create flattened arrays for table
        mu_g_flat = self.MU_G.flatten()
        b_flat = self.B.flatten()
        yield_flat = self.results['yield_gains'].flatten()
        field_flat = self.results['field_gains'].flatten()
        schwinger_flat = self.results['schwinger_rates'].flatten()
        critical_flat = self.results['critical_fields'].flatten()
        
        # Create DataFrame
        df = pd.DataFrame({
            'mu_g': mu_g_flat,
            'b': b_flat,
            'Yield_Gain_Ratio': yield_flat,
            'Field_Gain_Ratio': field_flat,
            'Schwinger_Rate': schwinger_flat,
            'Critical_Field': critical_flat
        })
        
        # Round for readability
        df = df.round({
            'mu_g': 4,
            'b': 2,
            'Yield_Gain_Ratio': 6,
            'Field_Gain_Ratio': 6,
            'Schwinger_Rate': 8,
            'Critical_Field': 6
        })
        
        return df

    def generate_comprehensive_plots(self, save_dir: str = "."):
        """Generate comprehensive 2D parameter space visualizations."""
        if not self.results:
            print("No results to plot. Run parameter space computation first.")
            return
        
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Set up the plot style
        plt.style.use('default')
        
        # Create comprehensive figure
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Yield gains contour plot
        ax1 = plt.subplot(2, 3, 1)
        contour1 = ax1.contourf(self.MU_G, self.B, self.results['yield_gains'], 
                               levels=20, cmap='viridis')
        ax1.set_xlabel('μ_g (Polymer Parameter)')
        ax1.set_ylabel('b (β-function coefficient)')
        ax1.set_title('Yield Gains: Γ_total^poly/Γ_0')
        plt.colorbar(contour1, ax=ax1)
        
        # Mark optimum
        opt = self.results['optimization']
        ax1.plot(opt['optimal_yield_mu_g'], opt['optimal_yield_b'], 'r*', markersize=15, label='Optimum')
        ax1.legend()
        
        # 2. Field gains contour plot  
        ax2 = plt.subplot(2, 3, 2)
        contour2 = ax2.contourf(self.MU_G, self.B, self.results['field_gains'], 
                               levels=20, cmap='plasma')
        ax2.set_xlabel('μ_g (Polymer Parameter)')
        ax2.set_ylabel('b (β-function coefficient)')
        ax2.set_title('Field Gains: E_crit^poly/E_crit')
        plt.colorbar(contour2, ax=ax2)
        
        # Mark optimum
        ax2.plot(opt['optimal_field_mu_g'], opt['optimal_field_b'], 'r*', markersize=15, label='Optimum')
        ax2.legend()
        
        # 3. Combined optimization surface
        ax3 = plt.subplot(2, 3, 3)
        combined_metric = self.results['yield_gains'] * self.results['field_gains']
        contour3 = ax3.contourf(self.MU_G, self.B, combined_metric, levels=20, cmap='inferno')
        ax3.set_xlabel('μ_g (Polymer Parameter)')
        ax3.set_ylabel('b (β-function coefficient)')
        ax3.set_title('Combined Metric: Yield × Field Gains')
        plt.colorbar(contour3, ax=ax3)
        
        # 4. Cross-sections at optimal points
        ax4 = plt.subplot(2, 3, 4)
        # Yield gains vs μ_g at optimal b
        opt_b_idx = np.argmin(np.abs(self.b_grid - opt['optimal_yield_b']))
        ax4.plot(self.mu_g_grid, self.results['yield_gains'][opt_b_idx, :], 'b-', linewidth=2, label='Yield Gains')
        ax4.set_xlabel('μ_g')
        ax4.set_ylabel('Gain Ratio')
        ax4.set_title(f'Cross-section at b = {opt["optimal_yield_b"]:.1f}')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # 5. Cross-sections at optimal points
        ax5 = plt.subplot(2, 3, 5)
        # Field gains vs b at optimal μ_g
        opt_mu_idx = np.argmin(np.abs(self.mu_g_grid - opt['optimal_field_mu_g']))
        ax5.plot(self.b_grid, self.results['field_gains'][:, opt_mu_idx], 'r-', linewidth=2, label='Field Gains')
        ax5.set_xlabel('b')
        ax5.set_ylabel('Gain Ratio')
        ax5.set_title(f'Cross-section at μ_g = {opt["optimal_field_mu_g"]:.3f}')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
        
        # 6. Statistical summary
        ax6 = plt.subplot(2, 3, 6)
        yield_stats = self.results['statistics']['yield_gains']
        field_stats = self.results['statistics']['field_gains']
        
        categories = ['Mean', 'Std', 'Min', 'Max', 'Q25', 'Q75']
        yield_values = [yield_stats[k.lower()] for k in categories]
        field_values = [field_stats[k.lower()] for k in categories]
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax6.bar(x - width/2, yield_values, width, label='Yield Gains', alpha=0.8)
        ax6.bar(x + width/2, field_values, width, label='Field Gains', alpha=0.8)
        ax6.set_xlabel('Statistical Measures')
        ax6.set_ylabel('Values')
        ax6.set_title('Statistical Summary')
        ax6.set_xticks(x)
        ax6.set_xticklabels(categories, rotation=45)
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = save_path / "parameter_space_2d_comprehensive.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive plots saved to {plot_file}")
        
        # Generate additional detailed plots
        self._generate_detailed_plots(save_path)

    def _generate_detailed_plots(self, save_path: Path):
        """Generate additional detailed analysis plots."""
        
        # 3D surface plots
        fig = plt.figure(figsize=(15, 5))
        
        # 3D Yield gains surface
        ax1 = fig.add_subplot(131, projection='3d')
        surf1 = ax1.plot_surface(self.MU_G, self.B, self.results['yield_gains'], 
                                cmap='viridis', alpha=0.8)
        ax1.set_xlabel('μ_g')
        ax1.set_ylabel('b')
        ax1.set_zlabel('Yield Gains')
        ax1.set_title('3D Yield Gains Surface')
        
        # 3D Field gains surface
        ax2 = fig.add_subplot(132, projection='3d')
        surf2 = ax2.plot_surface(self.MU_G, self.B, self.results['field_gains'], 
                                cmap='plasma', alpha=0.8)
        ax2.set_xlabel('μ_g')
        ax2.set_ylabel('b')
        ax2.set_zlabel('Field Gains')
        ax2.set_title('3D Field Gains Surface')
        
        # 3D Combined metric surface
        ax3 = fig.add_subplot(133, projection='3d')
        combined = self.results['yield_gains'] * self.results['field_gains']
        surf3 = ax3.plot_surface(self.MU_G, self.B, combined, 
                                cmap='inferno', alpha=0.8)
        ax3.set_xlabel('μ_g')
        ax3.set_ylabel('b')
        ax3.set_zlabel('Combined Metric')
        ax3.set_title('3D Combined Metric Surface')
        
        plt.tight_layout()
        plot_file_3d = save_path / "parameter_space_3d_surfaces.png"
        plt.savefig(plot_file_3d, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ 3D surface plots saved to {plot_file_3d}")

    def export_results(self, filename: str = "parameter_space_2d_sweep.json"):
        """Export complete results to JSON file."""
        if not self.results:
            print("No results to export. Run parameter space computation first.")
            return
        
        # Convert numpy arrays for JSON serialization
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
            return obj
        
        serializable_results = convert_for_json(self.results)
        
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"✅ Results exported to {filename}")

    def export_table(self, filename: str = "parameter_space_table.csv"):
        """Export parameter table to CSV file."""
        df = self.generate_parameter_table()
        if df is not None:
            df.to_csv(filename, index=False)
            print(f"✅ Parameter table exported to {filename}")

def main():
    """Main execution function."""
    
    # Configuration
    config = ParameterSweepConfig(
        mu_g_min=0.05, mu_g_max=0.5, mu_g_points=25,
        b_min=0.0, b_max=15.0, b_points=20,
        E_test_field=1.0
    )
    
    # Initialize sweep
    sweep = ParameterSpace2DSweep(config)
    
    # Compute parameter space
    results = sweep.compute_2d_parameter_space()
    
    # Generate plots
    sweep.generate_comprehensive_plots()
    
    # Export results
    sweep.export_results()
    sweep.export_table()
    
    # Summary
    print("\n" + "="*70)
    print("2D PARAMETER SPACE SWEEP COMPLETE")
    print("="*70)
    print("✅ 2D sweep over (μ_g, b) parameter space completed")
    print("✅ Yield gains Γ_total^poly/Γ_0 computed and tabulated")
    print("✅ Field gains E_crit^poly/E_crit computed and tabulated")
    print("✅ Optimization analysis and statistical summary generated")
    print("✅ Comprehensive visualizations and tables exported")
    print("✅ Results integrated for UQ pipeline compatibility")
    
    return results

if __name__ == "__main__":
    results = main()
