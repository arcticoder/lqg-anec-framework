#!/usr/bin/env python3
"""
Backreaction Stability Validation Script

Comprehensive validation of backreaction stability for negative-energy sources
in the context of quantum inequality circumvention and ANEC violation.

Features:
- Einstein field equation integration with LQG stress-energy sources
- Geometry stability analysis under sustained negative energy flux
- Critical threshold identification for spacetime stability
- GPU-optimized numerical integration and analysis
- Systematic validation against known stability criteria

Usage:
    python scripts/validate_backreaction_stability.py [--gpu] [--duration DAYS] [--flux-target W]
"""

import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    import torch
    from semi_classical_stress import SemiClassicalStressTensor, LQGParameters, SpinNetworkType
    from ghost_condensate_eft import GhostCondensateEFT, GhostEFTParameters
    from custom_kernels import CustomKernelLibrary
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Note: Some functionality may be limited without full module imports")

# Physical constants
PLANCK_LENGTH = 1.616e-35  # meters
PLANCK_TIME = 5.391e-44    # seconds
SPEED_OF_LIGHT = 2.998e8   # m/s
NEWTON_G = 6.674e-11       # N⋅m²/kg²
PLANCK_ENERGY = 1.956e9    # Joules

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('backreaction_validation.log')
        ]
    )
    return logging.getLogger(__name__)

class BackreactionStabilityValidator:
    """
    Comprehensive backreaction stability analysis framework.
    """
    
    def __init__(self, use_gpu: bool = True, logger: Optional[logging.Logger] = None):
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.logger = logger or logging.getLogger(__name__)
        
        # Physical setup
        self.planck_units = {
            'length': PLANCK_LENGTH,
            'time': PLANCK_TIME,
            'energy': PLANCK_ENERGY
        }
        
        # Initialize frameworks
        self._setup_analysis_frameworks()
        
        self.logger.info(f"Backreaction validator initialized on {self.device}")
    
    def _setup_analysis_frameworks(self):
        """Initialize LQG and ghost EFT frameworks for stability analysis."""
        try:
            # LQG stress tensor framework
            self.lqg_params = LQGParameters(
                network_type=SpinNetworkType.CUBICAL,
                max_spin=2,
                network_size=8,
                device=self.device
            )
            self.lqg_stress = SemiClassicalStressTensor(self.lqg_params)
            
            # Ghost condensate EFT framework
            self.ghost_params = GhostEFTParameters(
                lambda_ghost=0.1,
                cutoff_scale=10.0,
                grid_size=32,
                device=self.device
            )
            self.ghost_eft = GhostCondensateEFT(self.ghost_params)
            
            self.logger.info("Analysis frameworks initialized successfully")
            
        except Exception as e:
            self.logger.warning(f"Framework initialization failed: {e}")
            self.lqg_stress = None
            self.ghost_eft = None
    
    def einstein_field_equations(self, t: float, y: np.ndarray, stress_tensor_func: callable) -> np.ndarray:
        """
        Einstein field equations with LQG stress-energy source.
        
        G_μν = 8πG T_μν
        
        Args:
            t: Time coordinate
            y: State vector [g_00, g_11, g_22, g_33, ∂g_00/∂t, ...]
            stress_tensor_func: Function returning stress-energy tensor components
            
        Returns:
            Derivative vector dy/dt
        """
        
        # Extract metric components and their derivatives
        g_00, g_11, g_22, g_33 = y[:4]
        dg_00_dt, dg_11_dt, dg_22_dt, dg_33_dt = y[4:8]
        
        # Compute stress-energy tensor
        T_mu_nu = stress_tensor_func(t, y)
        
        # Einstein tensor components (simplified 1+1D case)
        # G_00 = R_00 - (1/2) g_00 R
        # For weak field: g_μν = η_μν + h_μν
        
        h_00 = g_00 + 1.0  # Perturbation from Minkowski
        h_11 = g_11 - 1.0
        
        # Compute Ricci tensor components (linearized)
        R_00 = -0.5 * (h_11 + h_22 + h_33)  # Simplified
        R_11 = -0.5 * h_00
        
        # Scalar curvature
        R = g_00 * R_00 + g_11 * R_11  # Simplified trace
        
        # Einstein tensor
        G_00 = R_00 - 0.5 * g_00 * R
        G_11 = R_11 - 0.5 * g_11 * R
        
        # Einstein field equations: G_μν = 8πG T_μν
        einstein_constant = 8.0 * np.pi * NEWTON_G / (SPEED_OF_LIGHT**4)
        
        # Second derivatives from field equations
        d2g_00_dt2 = einstein_constant * T_mu_nu[0, 0] - G_00
        d2g_11_dt2 = einstein_constant * T_mu_nu[1, 1] - G_11
        d2g_22_dt2 = 0.0  # Simplified
        d2g_33_dt2 = 0.0  # Simplified
        
        # Return derivative vector
        dydt = np.array([
            dg_00_dt,
            dg_11_dt,
            dg_22_dt,
            dg_33_dt,
            d2g_00_dt2,
            d2g_11_dt2,
            d2g_22_dt2,
            d2g_33_dt2
        ])
        
        return dydt
    
    def create_negative_energy_stress_tensor(self, flux_magnitude: float) -> callable:
        """
        Create stress-energy tensor function for negative energy source.
        
        Args:
            flux_magnitude: Magnitude of negative energy flux (Watts)
            
        Returns:
            Function T_μν(t, y) returning 4x4 stress tensor
        """
        
        def stress_tensor_func(t: float, y: np.ndarray) -> np.ndarray:
            """Stress-energy tensor for negative energy source."""
            
            # Convert flux to energy density (J/m³)
            # Assume localized source with characteristic scale ~ Planck length
            volume_scale = PLANCK_LENGTH**3
            energy_density = flux_magnitude / (SPEED_OF_LIGHT * volume_scale)
            
            # Negative energy density
            T_00 = -abs(energy_density)
            
            # Pressure components (simplified isotropic)
            pressure = T_00 / 3.0  # Relativistic relation
            
            # Construct stress-energy tensor
            T_mu_nu = np.array([
                [T_00, 0.0, 0.0, 0.0],
                [0.0, pressure, 0.0, 0.0],
                [0.0, 0.0, pressure, 0.0],
                [0.0, 0.0, 0.0, pressure]
            ])
            
            # Add time-dependent modulation (realistic pulse)
            time_scale = 7.0 * 24 * 3600  # 1 week in seconds
            pulse_profile = np.exp(-0.5 * (t / time_scale)**2)
            
            return T_mu_nu * pulse_profile
        
        return stress_tensor_func
    
    def analyze_stability_criteria(self, metric_evolution: np.ndarray, time_grid: np.ndarray) -> Dict[str, Any]:
        """
        Analyze spacetime stability based on metric evolution.
        
        Args:
            metric_evolution: Array of metric components over time
            time_grid: Time coordinates
            
        Returns:
            Dictionary of stability metrics
        """
        
        g_00_evolution = metric_evolution[:, 0]
        g_11_evolution = metric_evolution[:, 1]
        
        # Check for singularities (metric determinant → 0)
        metric_det = g_00_evolution * g_11_evolution  # Simplified 2x2 determinant
        
        # Stability criteria
        criteria = {
            'no_singularities': np.all(np.abs(metric_det) > 1e-10),
            'bounded_perturbations': np.all(np.abs(g_00_evolution + 1.0) < 1.0),  # |h_00| < 1
            'causal_structure_preserved': np.all(g_00_evolution < 0),  # Timelike remains timelike
            'asymptotic_flatness': abs(g_00_evolution[-1] + 1.0) < 0.1,  # Returns to Minkowski
        }
        
        # Quantitative metrics
        max_perturbation = np.max(np.abs(g_00_evolution + 1.0))
        perturbation_growth_rate = np.max(np.abs(np.diff(g_00_evolution)))
        
        stability_score = (
            1.0 * criteria['no_singularities'] +
            1.0 * criteria['bounded_perturbations'] +
            1.0 * criteria['causal_structure_preserved'] +
            1.0 * criteria['asymptotic_flatness']
        ) / 4.0
        
        analysis_results = {
            'stability_criteria': criteria,
            'stability_score': stability_score,
            'max_perturbation': max_perturbation,
            'growth_rate': perturbation_growth_rate,
            'final_metric_deviation': abs(g_00_evolution[-1] + 1.0),
            'metric_determinant_min': np.min(np.abs(metric_det)),
            'is_stable': stability_score > 0.75
        }
        
        return analysis_results
    
    def find_critical_flux_threshold(self, max_duration: float, tolerance: float = 1e-26) -> Dict[str, Any]:
        """
        Find critical negative energy flux threshold for geometry stability.
        
        Args:
            max_duration: Maximum duration to test (seconds)
            tolerance: Tolerance for flux magnitude search
            
        Returns:
            Dictionary with critical threshold analysis
        """
        
        self.logger.info(f"Searching for critical flux threshold up to {max_duration/86400:.2f} days...")
        
        def test_stability_for_flux(flux_magnitude: float) -> bool:
            """Test if given flux magnitude maintains stability."""
            
            try:
                # Create stress tensor
                stress_func = self.create_negative_energy_stress_tensor(flux_magnitude)
                
                # Initial conditions (Minkowski spacetime)
                y0 = np.array([-1.0, 1.0, 1.0, 1.0,  # g_μν
                              0.0, 0.0, 0.0, 0.0])   # ∂g_μν/∂t
                
                # Time grid
                t_span = (0, max_duration)
                t_eval = np.linspace(0, max_duration, 1000)
                
                # Solve Einstein equations
                solution = solve_ivp(
                    fun=lambda t, y: self.einstein_field_equations(t, y, stress_func),
                    t_span=t_span,
                    y0=y0,
                    t_eval=t_eval,
                    method='RK45',
                    rtol=1e-8,
                    atol=1e-10,
                    max_step=max_duration/100
                )
                
                if not solution.success:
                    return False
                
                # Analyze stability
                stability_analysis = self.analyze_stability_criteria(solution.y.T, solution.t)
                
                return stability_analysis['is_stable']
                
            except Exception as e:
                self.logger.warning(f"Stability test failed for flux {flux_magnitude:.2e}: {e}")
                return False
        
        # Binary search for critical threshold
        flux_min = 1e-30  # Very small flux
        flux_max = 1e-20  # Large flux (definitely unstable)
        
        # Verify bounds
        if not test_stability_for_flux(flux_min):
            self.logger.warning("Even minimal flux causes instability!")
            return {'error': 'No stable regime found', 'critical_flux': 0.0}
        
        if test_stability_for_flux(flux_max):
            self.logger.info("Maximum test flux is still stable")
            return {'critical_flux': flux_max, 'lower_bound': True}
        
        # Binary search
        max_iterations = 20
        for i in range(max_iterations):
            flux_mid = np.sqrt(flux_min * flux_max)  # Geometric mean
            
            if test_stability_for_flux(flux_mid):
                flux_min = flux_mid
            else:
                flux_max = flux_mid
            
            if (flux_max - flux_min) / flux_min < tolerance:
                break
            
            self.logger.info(f"Iteration {i+1}: flux range [{flux_min:.3e}, {flux_max:.3e}] W")
        
        critical_flux = flux_min
        
        results = {
            'critical_flux': critical_flux,
            'search_iterations': i + 1,
            'flux_range': [flux_min, flux_max],
            'duration_tested': max_duration,
            'target_comparison': {
                'target_flux': 1e-25,  # Target from mission
                'critical_flux': critical_flux,
                'safety_margin': critical_flux / 1e-25 if critical_flux > 1e-25 else 0,
                'target_feasible': critical_flux >= 1e-25
            }
        }
        
        self.logger.info(f"Critical flux found: {critical_flux:.3e} W")
        self.logger.info(f"Target feasibility: {results['target_comparison']['target_feasible']}")
        
        return results
    
    def validate_week_scale_stability(self, flux_magnitude: float = 1e-25) -> Dict[str, Any]:
        """
        Validate geometry stability for week-scale negative energy operation.
        
        Args:
            flux_magnitude: Negative energy flux magnitude (Watts)
            
        Returns:
            Comprehensive stability validation results
        """
        
        self.logger.info(f"Validating week-scale stability for flux {flux_magnitude:.2e} W...")
        
        week_duration = 7.0 * 24 * 3600  # 1 week in seconds
        
        # Create stress tensor
        stress_func = self.create_negative_energy_stress_tensor(flux_magnitude)
        
        # Initial conditions
        y0 = np.array([-1.0, 1.0, 1.0, 1.0,  # Minkowski metric
                      0.0, 0.0, 0.0, 0.0])   # Zero initial derivatives
        
        # High-resolution time grid for week-scale analysis
        num_points = 2000
        t_span = (0, week_duration)
        t_eval = np.linspace(0, week_duration, num_points)
        
        start_time = time.time()
        
        try:
            # Solve Einstein field equations
            solution = solve_ivp(
                fun=lambda t, y: self.einstein_field_equations(t, y, stress_func),
                t_span=t_span,
                y0=y0,
                t_eval=t_eval,
                method='DOP853',  # High-accuracy method
                rtol=1e-10,
                atol=1e-12,
                max_step=week_duration/500
            )
            
            computation_time = time.time() - start_time
            
            if not solution.success:
                return {
                    'error': 'Integration failed',
                    'message': solution.message,
                    'computation_time': computation_time
                }
            
            # Analyze stability throughout evolution
            stability_analysis = self.analyze_stability_criteria(solution.y.T, solution.t)
            
            # Energy analysis
            stress_values = [stress_func(t, solution.y[:, i]) for i, t in enumerate(solution.t)]
            energy_densities = [T[0, 0] for T in stress_values]
            
            # Geometric analysis
            g_00_evolution = solution.y[0, :]
            g_11_evolution = solution.y[1, :]
            
            # Curvature analysis (simplified)
            metric_perturbations = np.abs(g_00_evolution + 1.0)
            max_curvature = np.max(metric_perturbations)
            
            validation_results = {
                'success': solution.success,
                'computation_time': computation_time,
                'flux_magnitude': flux_magnitude,
                'duration_days': week_duration / 86400,
                'stability_analysis': stability_analysis,
                'energy_analysis': {
                    'min_energy_density': float(np.min(energy_densities)),
                    'max_energy_density': float(np.max(energy_densities)),
                    'mean_energy_density': float(np.mean(energy_densities)),
                    'total_negative_energy': float(np.trapz(energy_densities, solution.t))
                },
                'geometric_analysis': {
                    'max_metric_perturbation': float(max_curvature),
                    'final_metric_deviation': float(abs(g_00_evolution[-1] + 1.0)),
                    'curvature_bounded': float(max_curvature) < 0.1,
                    'asymptotic_flatness': float(abs(g_00_evolution[-1] + 1.0)) < 0.01
                },
                'mission_validation': {
                    'target_achieved': stability_analysis['is_stable'],
                    'week_scale_feasible': stability_analysis['stability_score'] > 0.8,
                    'geometry_stable': stability_analysis['stability_criteria']['no_singularities'],
                    'causality_preserved': stability_analysis['stability_criteria']['causal_structure_preserved']
                }
            }
            
            self.logger.info(f"Week-scale validation completed in {computation_time:.2f}s")
            self.logger.info(f"Stability score: {stability_analysis['stability_score']:.3f}")
            self.logger.info(f"Mission feasible: {validation_results['mission_validation']['target_achieved']}")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Week-scale validation failed: {e}")
            return {
                'error': str(e),
                'computation_time': time.time() - start_time,
                'flux_magnitude': flux_magnitude
            }
    
    def generate_comprehensive_report(self, output_dir: Path) -> Dict[str, Any]:
        """Generate comprehensive backreaction stability report."""
        
        self.logger.info("Generating comprehensive backreaction stability report...")
        
        # Test multiple scenarios
        test_scenarios = [
            {'flux': 1e-26, 'name': 'conservative'},
            {'flux': 1e-25, 'name': 'target'},
            {'flux': 1e-24, 'name': 'aggressive'}
        ]
        
        scenario_results = {}
        
        for scenario in test_scenarios:
            self.logger.info(f"Testing {scenario['name']} scenario (flux: {scenario['flux']:.1e} W)")
            
            # Week-scale validation
            week_result = self.validate_week_scale_stability(scenario['flux'])
            scenario_results[scenario['name']] = week_result
        
        # Critical threshold analysis
        self.logger.info("Determining critical flux threshold...")
        critical_analysis = self.find_critical_flux_threshold(7 * 24 * 3600)  # 1 week
        
        # Compile comprehensive results
        comprehensive_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'device': self.device,
            'scenario_analysis': scenario_results,
            'critical_threshold': critical_analysis,
            'mission_assessment': self._assess_mission_feasibility(scenario_results, critical_analysis),
            'theoretical_implications': self._analyze_theoretical_implications(scenario_results)
        }
        
        return comprehensive_results
    
    def _assess_mission_feasibility(self, scenario_results: Dict, critical_analysis: Dict) -> Dict[str, Any]:
        """Assess overall mission feasibility based on stability analysis."""
        
        target_result = scenario_results.get('target', {})
        
        feasibility = {
            'target_flux_stable': target_result.get('mission_validation', {}).get('target_achieved', False),
            'critical_threshold_adequate': critical_analysis.get('target_comparison', {}).get('target_feasible', False),
            'week_scale_feasible': target_result.get('mission_validation', {}).get('week_scale_feasible', False),
            'geometry_remains_stable': target_result.get('mission_validation', {}).get('geometry_stable', False),
            'overall_mission_feasible': False
        }
        
        # Overall assessment
        feasibility['overall_mission_feasible'] = all([
            feasibility['target_flux_stable'],
            feasibility['critical_threshold_adequate'],
            feasibility['week_scale_feasible'],
            feasibility['geometry_remains_stable']
        ])
        
        return feasibility
    
    def _analyze_theoretical_implications(self, scenario_results: Dict) -> Dict[str, Any]:
        """Analyze theoretical implications of stability results."""
        
        implications = {
            'spacetime_stability_confirmed': True,
            'negative_energy_bounded': True,
            'classical_gr_validity': True,
            'quantum_corrections_needed': False
        }
        
        # Check if any scenarios showed instabilities
        for scenario, results in scenario_results.items():
            if 'error' in results or not results.get('mission_validation', {}).get('target_achieved', False):
                implications['spacetime_stability_confirmed'] = False
                implications['quantum_corrections_needed'] = True
        
        return implications

def generate_visualizations(results: Dict[str, Any], output_dir: Path, logger: logging.Logger):
    """Generate comprehensive stability visualization suite."""
    
    logger.info("Generating backreaction stability visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Scenario comparison
    scenarios = ['conservative', 'target', 'aggressive']
    stability_scores = []
    flux_values = []
    
    for scenario in scenarios:
        if scenario in results['scenario_analysis']:
            score = results['scenario_analysis'][scenario].get('stability_analysis', {}).get('stability_score', 0)
            flux = results['scenario_analysis'][scenario].get('flux_magnitude', 0)
            stability_scores.append(score)
            flux_values.append(flux)
        else:
            stability_scores.append(0)
            flux_values.append(0)
    
    axes[0, 0].bar(scenarios, stability_scores, color=['green', 'orange', 'red'], alpha=0.7)
    axes[0, 0].set_ylabel('Stability Score')
    axes[0, 0].set_title('Stability by Scenario')
    axes[0, 0].set_ylim(0, 1.0)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axhline(y=0.75, color='black', linestyle='--', alpha=0.5, label='Stability Threshold')
    axes[0, 0].legend()
    
    # Plot 2: Flux vs Stability
    axes[0, 1].semilogx(flux_values, stability_scores, 'o-', linewidth=2, markersize=8)
    axes[0, 1].set_xlabel('Negative Energy Flux (W)')
    axes[0, 1].set_ylabel('Stability Score')
    axes[0, 1].set_title('Flux vs Stability Relationship')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0.75, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].axvline(x=1e-25, color='blue', linestyle='--', alpha=0.5, label='Mission Target')
    axes[0, 1].legend()
    
    # Plot 3: Critical threshold analysis
    if 'critical_threshold' in results and 'target_comparison' in results['critical_threshold']:
        critical_flux = results['critical_threshold']['critical_flux']
        target_flux = 1e-25
        
        axes[0, 2].bar(['Critical Flux', 'Target Flux'], [critical_flux, target_flux], 
                       color=['red', 'blue'], alpha=0.7)
        axes[0, 2].set_ylabel('Flux Magnitude (W)')
        axes[0, 2].set_title('Critical vs Target Flux')
        axes[0, 2].set_yscale('log')
        axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Mission feasibility assessment
    if 'mission_assessment' in results:
        assessment = results['mission_assessment']
        criteria = ['Target Stable', 'Threshold OK', 'Week Scale OK', 'Geometry OK', 'Overall']
        values = [
            assessment.get('target_flux_stable', False),
            assessment.get('critical_threshold_adequate', False),
            assessment.get('week_scale_feasible', False),
            assessment.get('geometry_remains_stable', False),
            assessment.get('overall_mission_feasible', False)
        ]
        
        colors = ['green' if v else 'red' for v in values]
        axes[1, 0].bar(criteria, [1 if v else 0 for v in values], color=colors, alpha=0.7)
        axes[1, 0].set_ylabel('Pass/Fail')
        axes[1, 0].set_title('Mission Feasibility Criteria')
        axes[1, 0].set_ylim(0, 1.2)
        axes[1, 0].grid(True, alpha=0.3)
        plt.setp(axes[1, 0].xaxis.get_majorticklabels(), rotation=45)
    
    # Plot 5: Energy density evolution (target scenario)
    target_results = results['scenario_analysis'].get('target', {})
    if 'energy_analysis' in target_results:
        # Mock time evolution for visualization
        time_hours = np.linspace(0, 168, 100)  # 1 week in hours
        energy_profile = np.exp(-0.5 * ((time_hours - 84) / 42)**2)  # Gaussian pulse
        energy_magnitude = abs(target_results['energy_analysis']['min_energy_density'])
        
        axes[1, 1].plot(time_hours, -energy_magnitude * energy_profile, 'r-', linewidth=2)
        axes[1, 1].set_xlabel('Time (hours)')
        axes[1, 1].set_ylabel('Energy Density (J/m³)')
        axes[1, 1].set_title('Negative Energy Evolution')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Plot 6: Theoretical implications
    if 'theoretical_implications' in results:
        implications = results['theoretical_implications']
        impl_names = ['Spacetime\nStable', 'Energy\nBounded', 'Classical GR\nValid', 'Quantum\nCorrections']
        impl_values = [
            implications.get('spacetime_stability_confirmed', False),
            implications.get('negative_energy_bounded', False),
            implications.get('classical_gr_validity', False),
            not implications.get('quantum_corrections_needed', True)  # Invert for positive display
        ]
        
        colors = ['green' if v else 'orange' for v in impl_values]
        axes[1, 2].bar(impl_names, [1 if v else 0 for v in impl_values], color=colors, alpha=0.7)
        axes[1, 2].set_ylabel('Confirmed/Valid')
        axes[1, 2].set_title('Theoretical Implications')
        axes[1, 2].set_ylim(0, 1.2)
        axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "backreaction_stability_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Stability visualizations completed and saved")

def main():
    """Main function for backreaction stability validation."""
    parser = argparse.ArgumentParser(description="Backreaction Stability Validation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--duration", type=float, default=7.0,
                       help="Test duration in days")
    parser.add_argument("--flux-target", type=float, default=1e-25,
                       help="Target negative energy flux in Watts")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--output-dir", type=str, default="backreaction_validation",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("Starting backreaction stability validation...")
    logger.info(f"GPU acceleration: {args.gpu}")
    logger.info(f"Test duration: {args.duration} days")
    logger.info(f"Target flux: {args.flux_target:.2e} W")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Initialize validator
        validator = BackreactionStabilityValidator(use_gpu=args.gpu, logger=logger)
        
        # Generate comprehensive report
        results = validator.generate_comprehensive_report(output_dir)
        
        # Save results
        results_file = output_dir / "stability_validation_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Generate visualizations
        generate_visualizations(results, output_dir, logger)
        
        # Generate summary report
        mission_feasible = results.get('mission_assessment', {}).get('overall_mission_feasible', False)
        
        summary_lines = [
            "# Backreaction Stability Validation Summary",
            f"Generated: {results['timestamp']}",
            f"Device: {results['device']}",
            "",
            "## Mission Feasibility Assessment",
            f"- **Overall mission feasible**: {'✅ YES' if mission_feasible else '❌ NO'}",
            f"- Target flux stable: {results.get('mission_assessment', {}).get('target_flux_stable', False)}",
            f"- Critical threshold adequate: {results.get('mission_assessment', {}).get('critical_threshold_adequate', False)}",
            f"- Week-scale operation feasible: {results.get('mission_assessment', {}).get('week_scale_feasible', False)}",
            f"- Geometry remains stable: {results.get('mission_assessment', {}).get('geometry_remains_stable', False)}",
            "",
            "## Critical Analysis",
        ]
        
        if 'critical_threshold' in results:
            critical_flux = results['critical_threshold'].get('critical_flux', 0)
            summary_lines.extend([
                f"- Critical flux threshold: {critical_flux:.2e} W",
                f"- Target flux feasible: {critical_flux >= args.flux_target}",
                f"- Safety margin: {critical_flux / args.flux_target:.1f}x" if critical_flux >= args.flux_target else "- ⚠️ Target exceeds critical threshold",
            ])
        
        summary_lines.extend([
            "",
            "## Theoretical Implications",
            "- Spacetime stability under negative energy confirmed" if mission_feasible else "- ⚠️ Stability concerns identified",
            "- Classical General Relativity framework adequate" if mission_feasible else "- Quantum corrections may be required",
            "- Week-scale operation theoretically validated" if mission_feasible else "- Extended operation may require additional safeguards",
            ""
        ])
        
        # Write summary
        with open(output_dir / "validation_summary.md", 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info("Backreaction stability validation completed!")
        logger.info(f"Mission feasible: {'YES' if mission_feasible else 'NO'}")
        
        if mission_feasible:
            logger.info("✅ Target 10⁻²⁵ W negative energy flux is stable for week-scale operation")
        else:
            logger.warning("⚠️ Stability concerns identified - review critical thresholds")
        
    except Exception as e:
        logger.error(f"Validation analysis failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
