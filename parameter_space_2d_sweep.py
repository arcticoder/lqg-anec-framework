#!/usr/bin/env python3
"""
2D Parameter-Space Sweep: μ_g and b Grid Analysis
=================================================

TASK 3 COMPLETION: Implement a 2D parameter-space sweep over μ_g ∈ [0.1,0.6] 
and b ∈ [0,10], computing and tabulating yield-vs-field gains:
- Γ_total^poly/Γ_0 (yield gain ratio)
- E_crit^poly/E_crit (critical field ratio)

Complete 2D grid analysis with visualization and data export.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
# import pandas as pd  # Optional: for advanced table handling

@dataclass
class ParameterSweepConfig:
    """Configuration for 2D parameter space sweep."""
    mu_g_min: float = 0.1        # Minimum polymer parameter
    mu_g_max: float = 0.6        # Maximum polymer parameter
    mu_g_points: int = 20        # Grid points in μ_g
    b_min: float = 0.0           # Minimum β-function coefficient
    b_max: float = 10.0          # Maximum β-function coefficient
    b_points: int = 25           # Grid points in b
    E_field_ref: float = 1e16    # Reference electric field (V/m)
    E_crit_classical: float = 1.32e18  # Classical critical field (V/m)

class TwoDimensionalSweep:
    """
    Complete 2D parameter space sweep implementation.
    """
    
    def __init__(self, config: ParameterSweepConfig = None):
        self.config = config or ParameterSweepConfig()
        
        # Import running coupling framework from Task 2
        from running_coupling_schwinger_integration import RunningCouplingFramework
        self.coupling_framework = RunningCouplingFramework()
        
        print("📊 2D PARAMETER SWEEP INITIALIZED")
        print(f"   μ_g range: [{self.config.mu_g_min}, {self.config.mu_g_max}] ({self.config.mu_g_points} points)")
        print(f"   b range: [{self.config.b_min}, {self.config.b_max}] ({self.config.b_points} points)")
        print(f"   Total grid: {self.config.mu_g_points} × {self.config.b_points} = {self.config.mu_g_points * self.config.b_points:,} points")
    
    def create_parameter_grid(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create 2D parameter grid."""
        mu_g_range = np.linspace(self.config.mu_g_min, self.config.mu_g_max, self.config.mu_g_points)
        b_range = np.linspace(self.config.b_min, self.config.b_max, self.config.b_points)
        
        return mu_g_range, b_range
    
    def gamma_total_polymer(self, E_field: float, mu_g: float, b: float) -> float:
        """
        Calculate total polymerized rate: Γ_total^poly
        Includes both running coupling and polymer corrections.
        """
        # Update polymer parameter in the framework
        old_mu_g = self.coupling_framework.config.mu_g
        self.coupling_framework.config.mu_g = mu_g
        
        # Calculate rate with running coupling and polymer corrections
        rate = self.coupling_framework.schwinger_rate_with_running_coupling(E_field, b)
        
        # Restore original parameter
        self.coupling_framework.config.mu_g = old_mu_g
        
        return rate
    
    def gamma_classical(self, E_field: float) -> float:
        """
        Calculate classical reference rate: Γ_0 (no polymer, no running coupling)
        """
        return self.coupling_framework.schwinger_rate_classical(E_field)
    
    def critical_field_polymer(self, mu_g: float, b: float) -> float:
        """
        Calculate polymer-corrected critical field: E_crit^poly
        """
        # Polymer modification to critical field
        # E_crit^poly = E_crit × F_polymer(μ_g) × F_running(b)
        
        # Polymer factor (simplified model)
        mu_pi = np.pi * mu_g
        sinc_factor = np.sin(mu_pi) / mu_pi if mu_pi > 0 else 1.0
        polymer_factor = sinc_factor**2
        
        # Running coupling factor
        alpha_enhancement = self.coupling_framework.alpha_eff(1.0, b) / self.coupling_framework.config.alpha_0
        running_factor = 1.0 / alpha_enhancement  # Stronger coupling reduces critical field
        
        return self.config.E_crit_classical * polymer_factor * running_factor
    
    def compute_2d_sweep(self) -> Dict:
        """
        Compute complete 2D parameter sweep.
        """
        print(f"\n🔄 COMPUTING 2D PARAMETER SWEEP...")
        
        mu_g_range, b_range = self.create_parameter_grid()
        
        # Initialize result arrays
        gamma_ratio = np.zeros((self.config.mu_g_points, self.config.b_points))
        E_crit_ratio = np.zeros((self.config.mu_g_points, self.config.b_points))
        
        # Classical reference values
        gamma_0 = self.gamma_classical(self.config.E_field_ref)
        E_crit_0 = self.config.E_crit_classical
        
        print(f"   Classical reference rate: {gamma_0:.2e} s⁻¹m⁻³")
        print(f"   Classical critical field: {E_crit_0:.2e} V/m")
        
        # Sweep over parameter space
        total_points = self.config.mu_g_points * self.config.b_points
        computed_points = 0
        
        for i, mu_g in enumerate(mu_g_range):
            for j, b in enumerate(b_range):
                # Yield gain ratio: Γ_total^poly/Γ_0
                gamma_poly = self.gamma_total_polymer(self.config.E_field_ref, mu_g, b)
                gamma_ratio[i, j] = gamma_poly / gamma_0 if gamma_0 > 0 else 1.0
                
                # Critical field ratio: E_crit^poly/E_crit
                E_crit_poly = self.critical_field_polymer(mu_g, b)
                E_crit_ratio[i, j] = E_crit_poly / E_crit_0
                
                computed_points += 1
                
                # Progress update
                if computed_points % 100 == 0 or computed_points == total_points:
                    progress = 100 * computed_points / total_points
                    print(f"   Progress: {progress:.1f}% ({computed_points:,}/{total_points:,} points)")
        
        # Find optimal parameters
        max_gamma_idx = np.unravel_index(np.argmax(gamma_ratio), gamma_ratio.shape)
        min_E_crit_idx = np.unravel_index(np.argmin(E_crit_ratio), E_crit_ratio.shape)
        
        optimal_results = {
            'max_yield_gain': {
                'mu_g': mu_g_range[max_gamma_idx[0]],
                'b': b_range[max_gamma_idx[1]],
                'gamma_ratio': gamma_ratio[max_gamma_idx]
            },
            'min_critical_field': {
                'mu_g': mu_g_range[min_E_crit_idx[0]],
                'b': b_range[min_E_crit_idx[1]],
                'E_crit_ratio': E_crit_ratio[min_E_crit_idx]
            }
        }
        
        print(f"\n🎯 OPTIMAL PARAMETERS:")
        print(f"   Max yield gain: {optimal_results['max_yield_gain']['gamma_ratio']:.3f}× at μ_g={optimal_results['max_yield_gain']['mu_g']:.2f}, b={optimal_results['max_yield_gain']['b']:.1f}")
        print(f"   Min critical field: {optimal_results['min_critical_field']['E_crit_ratio']:.3f}× at μ_g={optimal_results['min_critical_field']['mu_g']:.2f}, b={optimal_results['min_critical_field']['b']:.1f}")
        
        return {
            'parameter_ranges': {
                'mu_g_range': mu_g_range.tolist(),
                'b_range': b_range.tolist()
            },
            'results': {
                'gamma_ratio': gamma_ratio.tolist(),
                'E_crit_ratio': E_crit_ratio.tolist()
            },
            'optimal_parameters': optimal_results,
            'reference_values': {
                'gamma_0': gamma_0,
                'E_crit_0': E_crit_0,
                'E_field_ref': self.config.E_field_ref
            }
        }
    
    def generate_2d_plots(self, sweep_results: Dict, output_dir: str = ".") -> None:
        """
        Generate comprehensive 2D visualization plots.
        """
        print(f"\n📈 GENERATING 2D VISUALIZATION PLOTS...")
        
        mu_g_range = np.array(sweep_results['parameter_ranges']['mu_g_range'])
        b_range = np.array(sweep_results['parameter_ranges']['b_range'])
        gamma_ratio = np.array(sweep_results['results']['gamma_ratio'])
        E_crit_ratio = np.array(sweep_results['results']['E_crit_ratio'])
        
        # Create meshgrid for contour plots
        B, MU_G = np.meshgrid(b_range, mu_g_range)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Yield gain ratio (contour)
        im1 = axes[0,0].contourf(B, MU_G, gamma_ratio, levels=20, cmap='viridis')
        axes[0,0].set_xlabel('β-function coefficient b')
        axes[0,0].set_ylabel('μ_g')
        axes[0,0].set_title('Yield Gain Ratio: Γ_total^poly/Γ_0')
        plt.colorbar(im1, ax=axes[0,0])
        
        # Mark optimal point
        opt_gamma = sweep_results['optimal_parameters']['max_yield_gain']
        axes[0,0].plot(opt_gamma['b'], opt_gamma['mu_g'], 'r*', markersize=15, label='Optimal')
        axes[0,0].legend()
        
        # Plot 2: Critical field ratio (contour)
        im2 = axes[0,1].contourf(B, MU_G, E_crit_ratio, levels=20, cmap='plasma')
        axes[0,1].set_xlabel('β-function coefficient b')
        axes[0,1].set_ylabel('μ_g')
        axes[0,1].set_title('Critical Field Ratio: E_crit^poly/E_crit')
        plt.colorbar(im2, ax=axes[0,1])
        
        # Mark optimal point
        opt_E_crit = sweep_results['optimal_parameters']['min_critical_field']
        axes[0,1].plot(opt_E_crit['b'], opt_E_crit['mu_g'], 'r*', markersize=15, label='Optimal')
        axes[0,1].legend()
        
        # Plot 3: Combined optimization surface
        combined_metric = gamma_ratio / E_crit_ratio  # Higher yield, lower critical field
        im3 = axes[0,2].contourf(B, MU_G, combined_metric, levels=20, cmap='coolwarm')
        axes[0,2].set_xlabel('β-function coefficient b')
        axes[0,2].set_ylabel('μ_g')
        axes[0,2].set_title('Combined Metric: (Γ_ratio)/(E_crit_ratio)')
        plt.colorbar(im3, ax=axes[0,2])
        
        # Plot 4: Cross-sections at fixed μ_g
        axes[1,0].plot(b_range, gamma_ratio[len(mu_g_range)//4, :], label=f'μ_g = {mu_g_range[len(mu_g_range)//4]:.2f}')
        axes[1,0].plot(b_range, gamma_ratio[len(mu_g_range)//2, :], label=f'μ_g = {mu_g_range[len(mu_g_range)//2]:.2f}')
        axes[1,0].plot(b_range, gamma_ratio[3*len(mu_g_range)//4, :], label=f'μ_g = {mu_g_range[3*len(mu_g_range)//4]:.2f}')
        axes[1,0].set_xlabel('b')
        axes[1,0].set_ylabel('Γ_total^poly/Γ_0')
        axes[1,0].set_title('Yield Gain vs b (fixed μ_g)')
        axes[1,0].legend()
        axes[1,0].grid(True)
        
        # Plot 5: Cross-sections at fixed b
        axes[1,1].plot(mu_g_range, gamma_ratio[:, len(b_range)//4], label=f'b = {b_range[len(b_range)//4]:.1f}')
        axes[1,1].plot(mu_g_range, gamma_ratio[:, len(b_range)//2], label=f'b = {b_range[len(b_range)//2]:.1f}')
        axes[1,1].plot(mu_g_range, gamma_ratio[:, 3*len(b_range)//4], label=f'b = {b_range[3*len(b_range)//4]:.1f}')
        axes[1,1].set_xlabel('μ_g')
        axes[1,1].set_ylabel('Γ_total^poly/Γ_0')
        axes[1,1].set_title('Yield Gain vs μ_g (fixed b)')
        axes[1,1].legend()
        axes[1,1].grid(True)
        
        # Plot 6: Statistical distribution
        axes[1,2].hist(gamma_ratio.flatten(), bins=30, alpha=0.7, label='Yield Ratio', density=True)
        axes[1,2].hist(E_crit_ratio.flatten(), bins=30, alpha=0.7, label='E_crit Ratio', density=True)
        axes[1,2].set_xlabel('Ratio Value')
        axes[1,2].set_ylabel('Density')
        axes[1,2].set_title('Distribution of Enhancement Ratios')
        axes[1,2].legend()
        axes[1,2].grid(True)
        
        plt.tight_layout()
        plot_file = f"{output_dir}/2d_parameter_sweep_analysis.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"   ✅ 2D plots saved: {plot_file}")
    
    def generate_data_tables(self, sweep_results: Dict, output_dir: str = ".") -> None:
        """
        Generate tabulated data for experimental comparison.
        """
        print(f"\n📋 GENERATING DATA TABLES...")
        
        mu_g_range = np.array(sweep_results['parameter_ranges']['mu_g_range'])
        b_range = np.array(sweep_results['parameter_ranges']['b_range'])
        gamma_ratio = np.array(sweep_results['results']['gamma_ratio'])
        E_crit_ratio = np.array(sweep_results['results']['E_crit_ratio'])
        
        # Create comprehensive data table
        table_data = []
        
        for i, mu_g in enumerate(mu_g_range):
            for j, b in enumerate(b_range):
                table_data.append({
                    'mu_g': mu_g,
                    'b': b,
                    'gamma_ratio': gamma_ratio[i, j],
                    'E_crit_ratio': E_crit_ratio[i, j],
                    'combined_metric': gamma_ratio[i, j] / E_crit_ratio[i, j]
                })
          # Convert to basic table format
        # df = pd.DataFrame(table_data)  # Would use pandas if available
        
        # Manual CSV creation
        import csv
        
        # Save full table
        table_file = f"{output_dir}/2d_parameter_sweep_table.csv"
        with open(table_file, 'w', newline='') as csvfile:
            fieldnames = ['mu_g', 'b', 'gamma_ratio', 'E_crit_ratio', 'combined_metric']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in table_data:
                writer.writerow(row)
        
        # Create summary table with key parameter ranges
        summary_data = []
        
        # Sample key points
        key_mu_g_indices = [0, len(mu_g_range)//4, len(mu_g_range)//2, 3*len(mu_g_range)//4, -1]
        key_b_indices = [0, len(b_range)//4, len(b_range)//2, 3*len(b_range)//4, -1]
        
        for i in key_mu_g_indices:
            for j in key_b_indices:
                summary_data.append({
                    'mu_g': mu_g_range[i],
                    'b': b_range[j],
                    'gamma_ratio': gamma_ratio[i, j],
                    'E_crit_ratio': E_crit_ratio[i, j]
                })
          # Create summary table with key parameter ranges
        summary_data = []
        
        # Sample key points
        key_mu_g_indices = [0, len(mu_g_range)//4, len(mu_g_range)//2, 3*len(mu_g_range)//4, -1]
        key_b_indices = [0, len(b_range)//4, len(b_range)//2, 3*len(b_range)//4, -1]
        
        for i in key_mu_g_indices:
            for j in key_b_indices:
                summary_data.append({
                    'mu_g': mu_g_range[i],
                    'b': b_range[j],
                    'gamma_ratio': gamma_ratio[i, j],
                    'E_crit_ratio': E_crit_ratio[i, j]
                })
        
        # Save summary table
        summary_file = f"{output_dir}/2d_parameter_sweep_summary.csv"
        with open(summary_file, 'w', newline='') as csvfile:
            fieldnames = ['mu_g', 'b', 'gamma_ratio', 'E_crit_ratio']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in summary_data:
                writer.writerow(row)        
        print(f"   ✅ Full table: {table_file} ({len(table_data):,} rows)")
        print(f"   ✅ Summary table: {summary_file} ({len(summary_data)} rows)")
        
        # Print formatted summary table
        print(f"\n📊 SUMMARY TABLE (selected points):")
        print(f"{'mu_g':>6} {'b':>6} {'gamma_ratio':>12} {'E_crit_ratio':>12}")
        print("-" * 42)
        for row in summary_data:
            print(f"{row['mu_g']:6.2f} {row['b']:6.1f} {row['gamma_ratio']:12.3f} {row['E_crit_ratio']:12.3f}")
    
    def export_complete_results(self, sweep_results: Dict, 
                               output_file: str = "2d_parameter_sweep_complete.json") -> None:
        """Export complete 2D sweep results."""
        print(f"\n💾 EXPORTING COMPLETE RESULTS...")
        
        export_data = {
            'task_info': {
                'task_number': 3,
                'description': '2D parameter-space sweep over μ_g and b',
                'parameter_ranges': 'μ_g ∈ [0.1,0.6], b ∈ [0,10]',
                'computed_quantities': ['Γ_total^poly/Γ_0', 'E_crit^poly/E_crit']
            },
            'configuration': {
                'mu_g_range': [self.config.mu_g_min, self.config.mu_g_max],
                'b_range': [self.config.b_min, self.config.b_max],
                'grid_size': [self.config.mu_g_points, self.config.b_points],
                'total_points': self.config.mu_g_points * self.config.b_points,
                'reference_field': self.config.E_field_ref,
                'classical_critical_field': self.config.E_crit_classical
            },
            'sweep_results': sweep_results,
            'task_completion': {
                '2d_grid_computed': True,
                'yield_ratios_calculated': True,
                'critical_field_ratios_calculated': True,
                'optimal_parameters_identified': True,
                'data_tables_generated': True,
                'plots_generated': True
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"   ✅ Complete results exported to {output_file}")
    
    def validate_2d_sweep(self, sweep_results: Dict) -> Dict[str, bool]:
        """Validate the 2D sweep results."""
        print(f"\n✅ VALIDATING 2D SWEEP RESULTS...")
        
        gamma_ratio = np.array(sweep_results['results']['gamma_ratio'])
        E_crit_ratio = np.array(sweep_results['results']['E_crit_ratio'])
        
        tests = {}
        
        # Test 1: All ratios are positive
        tests['positive_ratios'] = np.all(gamma_ratio > 0) and np.all(E_crit_ratio > 0)
        
        # Test 2: Classical limit (μ_g→0, b→0) gives ratio ≈ 1
        classical_gamma = gamma_ratio[0, 0]  # μ_g=0.1, b=0
        tests['classical_limit_reasonable'] = 0.5 < classical_gamma < 2.0
        
        # Test 3: Enhancement exists for some parameters
        max_gamma = np.max(gamma_ratio)
        tests['enhancement_exists'] = max_gamma > 1.1
        
        # Test 4: Grid coverage is complete
        tests['complete_coverage'] = gamma_ratio.shape == (self.config.mu_g_points, self.config.b_points)
        
        # Test 5: Optimal parameters are within bounds
        opt_gamma = sweep_results['optimal_parameters']['max_yield_gain']
        mu_g_in_bounds = self.config.mu_g_min <= opt_gamma['mu_g'] <= self.config.mu_g_max
        b_in_bounds = self.config.b_min <= opt_gamma['b'] <= self.config.b_max
        tests['optimal_parameters_valid'] = mu_g_in_bounds and b_in_bounds
        
        for test_name, passed in tests.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        return tests


def demonstrate_task_3():
    """Demonstrate complete Task 3 implementation."""
    print("="*70)
    print("TASK 3: 2D PARAMETER-SPACE SWEEP (μ_g, b)")
    print("="*70)
    
    config = ParameterSweepConfig(
        mu_g_min=0.1, mu_g_max=0.6, mu_g_points=20,
        b_min=0.0, b_max=10.0, b_points=25
    )
    
    sweep_framework = TwoDimensionalSweep(config)
    
    # Compute complete 2D sweep
    sweep_results = sweep_framework.compute_2d_sweep()
    
    # Validate results
    validation_results = sweep_framework.validate_2d_sweep(sweep_results)
    
    # Generate visualizations
    sweep_framework.generate_2d_plots(sweep_results)
    
    # Generate data tables
    sweep_framework.generate_data_tables(sweep_results)
    
    # Export complete results
    sweep_framework.export_complete_results(sweep_results)
    
    print(f"\n🎯 TASK 3 COMPLETION SUMMARY:")
    print(f"   ✅ 2D grid computed: {config.mu_g_points} × {config.b_points} = {config.mu_g_points * config.b_points:,} points")
    print(f"   ✅ Parameter ranges: μ_g ∈ [{config.mu_g_min}, {config.mu_g_max}], b ∈ [{config.b_min}, {config.b_max}]")
    print(f"   ✅ Yield ratios: Γ_total^poly/Γ_0 computed for all points")
    print(f"   ✅ Critical field ratios: E_crit^poly/E_crit computed for all points")
    print(f"   ✅ Optimal parameters identified and marked")
    print(f"   ✅ Comprehensive plots: 6-panel analysis generated")
    print(f"   ✅ Data tables: Full and summary CSV files exported")
    print(f"   ✅ All validation tests: {all(validation_results.values())}")
    
    # Show key results
    opt_yield = sweep_results['optimal_parameters']['max_yield_gain']
    opt_field = sweep_results['optimal_parameters']['min_critical_field']
    
    print(f"\n📊 KEY RESULTS:")
    print(f"   Maximum yield gain: {opt_yield['gamma_ratio']:.3f}× at (μ_g={opt_yield['mu_g']:.2f}, b={opt_yield['b']:.1f})")
    print(f"   Minimum critical field: {opt_field['E_crit_ratio']:.3f}× at (μ_g={opt_field['mu_g']:.2f}, b={opt_field['b']:.1f})")
    
    gamma_ratio = np.array(sweep_results['results']['gamma_ratio'])
    print(f"   Yield enhancement range: [{np.min(gamma_ratio):.3f}, {np.max(gamma_ratio):.3f}]")
    
    return {
        'sweep_results': sweep_results,
        'validation': validation_results,
        'task_completed': all(validation_results.values())
    }


if __name__ == "__main__":
    results = demonstrate_task_3()
    print(f"\n🏆 TASK 3 STATUS: {'COMPLETED' if results['task_completed'] else 'INCOMPLETE'}")
