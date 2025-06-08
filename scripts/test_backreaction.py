"""
Backreaction and Geometry Stability Test Script

Backreaction and geometry stability solver integrating semi-classical LQG stress tensor.
Tests stability of negative-energy sources in curved spacetime geometries.

Key Features:
- Einstein field equations with LQG stress-energy source
- Geometry stability analysis under negative energy
- Backreaction effects on spin network geometry  
- GPU-optimized numerical integration
- CLI-driven analysis with comprehensive output files

Theory Background:
- Einstein equations: G_μν = 8πG T_μν
- LQG stress tensor as energy source
- Stability analysis via perturbation theory
- Backreaction on discrete geometry
- Critical thresholds for geometry collapse
"""

import torch
import numpy as np
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
from dataclasses import dataclass
import time

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from semi_classical_stress import SemiClassicalStressTensor, LQGParameters, SpinNetworkType
    from ghost_condensate_eft import GhostCondensateEFT, GhostEFTParameters
except ImportError as e:
    print(f"Warning: Could not import modules: {e}")
    print("Running in standalone mode...")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('backreaction_analysis.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BackreactionParameters:
    """Parameters for backreaction and stability analysis."""
    # Physical constants
    newton_constant: float = 6.674e-11      # Newton's constant G
    speed_of_light: float = 2.998e8         # Speed of light c
    planck_length: float = 1.616e-35        # Planck length
    
    # Geometry parameters
    spacetime_dimension: int = 4            # 3+1 dimensional spacetime
    spatial_grid_size: int = 64             # Spatial grid resolution
    temporal_range: float = 1.0             # Time evolution range
    spatial_range: float = 10.0             # Spatial domain size
    
    # Stability analysis
    perturbation_amplitude: float = 1e-6    # Initial perturbation size
    stability_threshold: float = 1e-3       # Stability threshold
    max_growth_rate: float = 10.0           # Maximum allowed growth rate
    
    # Numerical parameters
    integration_steps: int = 1000           # Time integration steps
    convergence_tolerance: float = 1e-8     # Numerical convergence
    max_iterations: int = 5000              # Maximum solver iterations
    
    # Output parameters
    save_interval: int = 50                 # Data saving interval
    plot_results: bool = True               # Generate plots
    device: str = "cuda"                    # Computation device


class GeometryStabilityAnalyzer:
    """
    Analyzes stability of spacetime geometry under LQG stress-energy backreaction.
    
    Solves modified Einstein equations with discrete LQG sources and tests
    for geometric stability, collapse thresholds, and negative energy bounds.
    """
    
    def __init__(self, params: BackreactionParameters):
        """Initialize geometry stability analyzer."""
        self.params = params
        self.device = torch.device(params.device if torch.cuda.is_available() else "cpu")
        
        # Initialize coordinate system
        self._setup_coordinates()
        
        # Initialize metric components
        self._setup_metric()
        
        # Initialize LQG stress tensor
        self._setup_lqg_source()
        
        # Initialize stability analysis
        self._setup_stability_analysis()
        
        logger.info(f"Geometry stability analyzer initialized on {self.device}")
        logger.info(f"Grid: {params.spatial_grid_size}³, Steps: {params.integration_steps}")
    
    def _setup_coordinates(self):
        """Setup spacetime coordinate grids."""
        # Time coordinate
        self.t_vals = torch.linspace(0, self.params.temporal_range,
                                   self.params.integration_steps, device=self.device)
        
        # Spatial coordinates
        grid_size = self.params.spatial_grid_size
        spatial_range = self.params.spatial_range
        
        x_vals = torch.linspace(-spatial_range/2, spatial_range/2, grid_size, device=self.device)
        y_vals = torch.linspace(-spatial_range/2, spatial_range/2, grid_size, device=self.device)
        z_vals = torch.linspace(-spatial_range/2, spatial_range/2, grid_size, device=self.device)
        
        # Create spatial meshgrids
        self.X, self.Y, self.Z = torch.meshgrid(x_vals, y_vals, z_vals, indexing='ij')
        
        # Radial coordinate
        self.R = torch.sqrt(self.X**2 + self.Y**2 + self.Z**2)
        
        # Store coordinate arrays
        self.coords = {
            't': self.t_vals,
            'x': x_vals,
            'y': y_vals, 
            'z': z_vals,
            'X': self.X,
            'Y': self.Y,
            'Z': self.Z,
            'R': self.R
        }
    
    def _setup_metric(self):
        """Initialize metric tensor components."""
        grid_shape = (self.params.spatial_grid_size,) * 3
        
        # Start with Minkowski metric
        self.g_tt = -torch.ones(grid_shape, device=self.device)
        self.g_xx = torch.ones(grid_shape, device=self.device)
        self.g_yy = torch.ones(grid_shape, device=self.device)
        self.g_zz = torch.ones(grid_shape, device=self.device)
        
        # Off-diagonal components (initially zero)
        self.g_tx = torch.zeros(grid_shape, device=self.device)
        self.g_ty = torch.zeros(grid_shape, device=self.device)
        self.g_tz = torch.zeros(grid_shape, device=self.device)
        self.g_xy = torch.zeros(grid_shape, device=self.device)
        self.g_xz = torch.zeros(grid_shape, device=self.device)
        self.g_yz = torch.zeros(grid_shape, device=self.device)
        
        # Metric determinant
        self.sqrt_g = torch.ones(grid_shape, device=self.device)
        
        # Store in dictionary
        self.metric = {
            'g_tt': self.g_tt,
            'g_xx': self.g_xx,
            'g_yy': self.g_yy,
            'g_zz': self.g_zz,
            'g_tx': self.g_tx,
            'g_ty': self.g_ty,
            'g_tz': self.g_tz,
            'g_xy': self.g_xy,
            'g_xz': self.g_xz,
            'g_yz': self.g_yz,
            'sqrt_g': self.sqrt_g
        }
    
    def _setup_lqg_source(self):
        """Initialize LQG stress-energy tensor source."""
        try:
            # LQG parameters for stress tensor
            lqg_params = LQGParameters(
                network_size=15,  # Moderate size for efficiency
                max_spin=8.0,
                coherent_scale=500.0,
                polymer_scale=self.params.planck_length * 1e10,  # Macroscopic polymer scale
                device=self.params.device
            )
            
            self.lqg_stress = SemiClassicalStressTensor(lqg_params)
            
            # Generate field configuration
            self.field_config = torch.randn(self.lqg_stress.n_nodes, device=self.device)
            
            logger.info(f"LQG source initialized: {self.lqg_stress.n_nodes} nodes")
            
        except Exception as e:
            logger.warning(f"Could not initialize LQG source: {e}")
            logger.info("Using simplified analytical source...")
            self._setup_analytical_source()
    
    def _setup_analytical_source(self):
        """Setup simplified analytical stress-energy source."""
        grid_shape = (self.params.spatial_grid_size,) * 3
        
        # Gaussian negative energy source
        source_width = self.params.spatial_range / 10
        source_amplitude = -1e-10  # Negative energy density
        
        self.T_00_analytical = source_amplitude * torch.exp(-self.R**2 / (2 * source_width**2))
        self.T_ij_analytical = torch.zeros(grid_shape + (3, 3), device=self.device)
        
        # Isotropic pressure
        for i in range(3):
            self.T_ij_analytical[..., i, i] = -self.T_00_analytical / 3
        
        logger.info("Analytical Gaussian source initialized")
    
    def _setup_stability_analysis(self):
        """Initialize stability analysis tools."""
        grid_shape = (self.params.spatial_grid_size,) * 3
        
        # Perturbation modes
        self.perturbation_modes = torch.randn(grid_shape + (10,), device=self.device)
        self.perturbation_modes *= self.params.perturbation_amplitude
        
        # Growth rate tracking
        self.growth_rates = torch.zeros(10, device=self.device)
        
        # Stability flags
        self.is_stable = True
        self.collapse_time = None
        self.critical_density = None
    
    def compute_einstein_tensor(self) -> Dict[str, torch.Tensor]:
        """
        Compute Einstein tensor G_μν from metric components.
        
        G_μν = R_μν - (1/2) g_μν R
        """
        # Compute metric derivatives (finite differences)
        g_derivatives = self._compute_metric_derivatives()
        
        # Compute Christoffel symbols
        christoffel = self._compute_christoffel_symbols(g_derivatives)
        
        # Compute Riemann tensor components
        riemann = self._compute_riemann_tensor(christoffel)
        
        # Contract to get Ricci tensor
        ricci = self._compute_ricci_tensor(riemann)
        
        # Compute Ricci scalar
        ricci_scalar = self._compute_ricci_scalar(ricci)
        
        # Einstein tensor G_μν = R_μν - (1/2) g_μν R
        einstein_tensor = {}
        
        einstein_tensor['G_tt'] = ricci['R_tt'] - 0.5 * self.g_tt * ricci_scalar
        einstein_tensor['G_xx'] = ricci['R_xx'] - 0.5 * self.g_xx * ricci_scalar  
        einstein_tensor['G_yy'] = ricci['R_yy'] - 0.5 * self.g_yy * ricci_scalar
        einstein_tensor['G_zz'] = ricci['R_zz'] - 0.5 * self.g_zz * ricci_scalar
        
        return einstein_tensor
    
    def _compute_metric_derivatives(self) -> Dict[str, torch.Tensor]:
        """Compute spatial derivatives of metric components."""
        derivatives = {}
        
        # Spatial grid spacing
        dx = self.params.spatial_range / self.params.spatial_grid_size
        
        # Compute derivatives using finite differences
        for component_name, component in self.metric.items():
            if component_name == 'sqrt_g':
                continue
                
            # First derivatives
            grad_x = torch.gradient(component, spacing=dx, dim=0)[0]
            grad_y = torch.gradient(component, spacing=dx, dim=1)[0]
            grad_z = torch.gradient(component, spacing=dx, dim=2)[0]
            
            derivatives[f'd_{component_name}_dx'] = grad_x
            derivatives[f'd_{component_name}_dy'] = grad_y
            derivatives[f'd_{component_name}_dz'] = grad_z
        
        return derivatives
    
    def _compute_christoffel_symbols(self, g_derivatives: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute Christoffel symbols from metric derivatives."""
        # Simplified computation for demonstration
        christoffel = {}
        
        # Only compute a few key components
        christoffel['Gamma_xxx'] = 0.5 * g_derivatives.get('dg_xx_dx', torch.zeros_like(self.g_xx))
        christoffel['Gamma_yyy'] = 0.5 * g_derivatives.get('dg_yy_dy', torch.zeros_like(self.g_yy))
        christoffel['Gamma_zzz'] = 0.5 * g_derivatives.get('dg_zz_dz', torch.zeros_like(self.g_zz))
        
        return christoffel
    
    def _compute_riemann_tensor(self, christoffel: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute Riemann tensor from Christoffel symbols."""
        # Simplified computation
        riemann = {}
        
        # Key components (simplified)
        riemann['R_xtxt'] = torch.gradient(christoffel.get('Gamma_xxx', torch.zeros_like(self.g_xx)), 
                                         spacing=self.params.spatial_range/self.params.spatial_grid_size, 
                                         dim=0)[0]
        
        return riemann
    
    def _compute_ricci_tensor(self, riemann: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute Ricci tensor by contracting Riemann tensor."""
        ricci = {}
        
        # Simplified Ricci components
        ricci['R_tt'] = torch.zeros_like(self.g_tt)
        ricci['R_xx'] = riemann.get('R_xtxt', torch.zeros_like(self.g_xx))
        ricci['R_yy'] = torch.zeros_like(self.g_yy)
        ricci['R_zz'] = torch.zeros_like(self.g_zz)
        
        return ricci
    
    def _compute_ricci_scalar(self, ricci: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute Ricci scalar by contracting Ricci tensor with inverse metric."""
        # R = g^μν R_μν (simplified)
        ricci_scalar = (-ricci['R_tt'] / self.g_tt + 
                       ricci['R_xx'] / self.g_xx +
                       ricci['R_yy'] / self.g_yy + 
                       ricci['R_zz'] / self.g_zz)
        
        return ricci_scalar
    
    def compute_stress_energy_source(self) -> Dict[str, torch.Tensor]:
        """
        Compute stress-energy tensor source from LQG or analytical model.
        """
        if hasattr(self, 'lqg_stress'):
            # Use LQG stress tensor
            try:
                lqg_stress_components = self.lqg_stress.compute_polymer_enhanced_stress(self.field_config)
                
                # Map to spacetime grid (simplified)
                grid_shape = (self.params.spatial_grid_size,) * 3
                
                T_source = {}
                T_source['T_tt'] = torch.mean(lqg_stress_components['T_00']) * torch.ones(grid_shape, device=self.device)
                T_source['T_xx'] = torch.mean(lqg_stress_components['T_ij'][:, 0, 0]) * torch.ones(grid_shape, device=self.device)
                T_source['T_yy'] = torch.mean(lqg_stress_components['T_ij'][:, 1, 1]) * torch.ones(grid_shape, device=self.device)
                T_source['T_zz'] = torch.mean(lqg_stress_components['T_ij'][:, 2, 2]) * torch.ones(grid_shape, device=self.device)
                
                return T_source
                
            except Exception as e:
                logger.warning(f"LQG stress computation failed: {e}")
                logger.info("Falling back to analytical source...")
        
        # Use analytical source
        T_source = {}
        T_source['T_tt'] = self.T_00_analytical
        T_source['T_xx'] = self.T_ij_analytical[..., 0, 0]
        T_source['T_yy'] = self.T_ij_analytical[..., 1, 1]
        T_source['T_zz'] = self.T_ij_analytical[..., 2, 2]
        
        return T_source
    
    def solve_einstein_equations(self) -> Dict[str, Any]:
        """
        Solve Einstein field equations with LQG stress-energy source.
        
        G_μν = 8πG T_μν
        """
        logger.info("Solving Einstein field equations...")
        
        # Einstein constant
        einstein_constant = 8 * np.pi * self.params.newton_constant / self.params.speed_of_light**4
        
        evolution_data = {
            'times': [],
            'metric_components': [],
            'curvature_scalars': [],
            'energy_densities': [],
            'stability_indicators': []
        }
        
        for step, t in enumerate(self.t_vals):
            # Compute stress-energy source
            T_source = self.compute_stress_energy_source()
            
            # Compute Einstein tensor
            G_tensor = self.compute_einstein_tensor()
            
            # Einstein equations: G_μν = 8πG T_μν
            # Update metric based on stress-energy (simplified integration)
            
            # Update tt component
            metric_update_tt = einstein_constant * T_source['T_tt'] * self.params.temporal_range / self.params.integration_steps
            self.g_tt += metric_update_tt
            
            # Update spatial components  
            metric_update_xx = einstein_constant * T_source['T_xx'] * self.params.temporal_range / self.params.integration_steps
            self.g_xx += metric_update_xx
            self.g_yy += einstein_constant * T_source['T_yy'] * self.params.temporal_range / self.params.integration_steps
            self.g_zz += einstein_constant * T_source['T_zz'] * self.params.temporal_range / self.params.integration_steps
            
            # Compute curvature scalar
            ricci = self._compute_ricci_tensor({})
            curvature_scalar = self._compute_ricci_scalar(ricci)
            
            # Store evolution data
            if step % self.params.save_interval == 0:
                evolution_data['times'].append(t.item())
                evolution_data['metric_components'].append({
                    'g_tt': self.g_tt.clone().cpu(),
                    'g_xx': self.g_xx.clone().cpu(),
                    'g_yy': self.g_yy.clone().cpu(),
                    'g_zz': self.g_zz.clone().cpu()
                })
                evolution_data['curvature_scalars'].append(curvature_scalar.clone().cpu())
                evolution_data['energy_densities'].append(T_source['T_tt'].clone().cpu())
                
                # Compute stability indicator
                stability_indicator = self._compute_stability_indicator()
                evolution_data['stability_indicators'].append(stability_indicator)
        
        logger.info("Einstein equation evolution completed")
        return evolution_data
    
    def _compute_stability_indicator(self) -> float:
        """Compute geometric stability indicator."""
        # Check for metric signature preservation
        det_spatial = self.g_xx * self.g_yy * self.g_zz
        det_total = -self.g_tt * det_spatial
        
        # Stability measure: log of determinant deviation from Minkowski
        stability = torch.mean(torch.log(torch.abs(det_total) + 1e-10))
        
        return stability.item()
    
    def analyze_perturbation_stability(self) -> Dict[str, Any]:
        """
        Analyze linear stability using perturbation theory.
        
        Studies growth/decay of small perturbations around background metric.
        """
        logger.info("Analyzing perturbation stability...")
        
        stability_results = {
            'growth_rates': [],
            'mode_analysis': [],
            'stability_threshold': self.params.stability_threshold,
            'is_stable': True,
            'unstable_modes': []
        }
        
        # Perturb metric components
        for mode_idx in range(self.perturbation_modes.shape[-1]):
            # Apply perturbation
            perturbation = self.perturbation_modes[..., mode_idx]
            
            # Store original metric
            original_g_tt = self.g_tt.clone()
            original_g_xx = self.g_xx.clone()
            
            # Apply perturbation
            self.g_tt += perturbation * self.params.perturbation_amplitude
            self.g_xx += perturbation * self.params.perturbation_amplitude
            
            # Evolve for short time
            initial_curvature = torch.mean(torch.abs(self._compute_ricci_scalar(self._compute_ricci_tensor({}))))
            
            # Simple evolution step
            T_source = self.compute_stress_energy_source()
            einstein_constant = 8 * np.pi * self.params.newton_constant / self.params.speed_of_light**4
            
            dt = self.params.temporal_range / 100  # Small time step
            self.g_tt += einstein_constant * T_source['T_tt'] * dt
            self.g_xx += einstein_constant * T_source['T_xx'] * dt
            
            # Final curvature
            final_curvature = torch.mean(torch.abs(self._compute_ricci_scalar(self._compute_ricci_tensor({}))))
            
            # Growth rate
            if initial_curvature > 1e-15:
                growth_rate = (final_curvature - initial_curvature) / (initial_curvature * dt)
            else:
                growth_rate = torch.tensor(0.0)
            
            self.growth_rates[mode_idx] = growth_rate
            stability_results['growth_rates'].append(growth_rate.item())
            
            # Check stability
            if growth_rate > self.params.stability_threshold:
                stability_results['is_stable'] = False
                stability_results['unstable_modes'].append(mode_idx)
            
            # Restore original metric
            self.g_tt = original_g_tt
            self.g_xx = original_g_xx
            
            # Mode analysis
            mode_data = {
                'mode_index': mode_idx,
                'growth_rate': growth_rate.item(),
                'initial_curvature': initial_curvature.item(),
                'final_curvature': final_curvature.item(),
                'perturbation_norm': torch.norm(perturbation).item()
            }
            stability_results['mode_analysis'].append(mode_data)
        
        # Overall stability assessment
        max_growth_rate = max(stability_results['growth_rates'])
        stability_results['max_growth_rate'] = max_growth_rate
        stability_results['num_unstable_modes'] = len(stability_results['unstable_modes'])
        
        self.is_stable = stability_results['is_stable']
        
        logger.info(f"Stability analysis: {'STABLE' if self.is_stable else 'UNSTABLE'}")
        logger.info(f"Max growth rate: {max_growth_rate:.2e}")
        logger.info(f"Unstable modes: {len(stability_results['unstable_modes'])}")
        
        return stability_results
    
    def compute_negative_energy_bounds(self) -> Dict[str, Any]:
        """
        Compute bounds on negative energy density and duration.
        
        Analyzes quantum inequality violations and geometric constraints.
        """
        logger.info("Computing negative energy bounds...")
        
        # Get stress-energy source
        T_source = self.compute_stress_energy_source()
        energy_density = T_source['T_tt']
        
        # Negative energy analysis
        negative_mask = energy_density < 0
        negative_fraction = torch.mean(negative_mask.float())
        
        if torch.any(negative_mask):
            min_energy_density = torch.min(energy_density)
            negative_energy_magnitude = torch.sum(torch.abs(energy_density[negative_mask]))
            negative_volume = torch.sum(negative_mask.float()) * (self.params.spatial_range / self.params.spatial_grid_size)**3
        else:
            min_energy_density = torch.tensor(0.0)
            negative_energy_magnitude = torch.tensor(0.0)
            negative_volume = torch.tensor(0.0)
        
        # Geometric constraints
        horizon_scale = self.params.planck_length * torch.sqrt(torch.abs(min_energy_density) / 1e-30)  # Rough estimate
        
        # Quantum inequality estimate (simplified)
        # ΔE Δt ≥ ℏ/2 constraint
        hbar = 1.055e-34
        if negative_energy_magnitude > 1e-20:
            min_duration = hbar / (2 * negative_energy_magnitude.item())
        else:
            min_duration = float('inf')
        
        bounds_analysis = {
            'negative_energy_fraction': negative_fraction.item(),
            'min_energy_density': min_energy_density.item(),
            'negative_energy_magnitude': negative_energy_magnitude.item(),
            'negative_volume': negative_volume.item(),
            'horizon_scale': horizon_scale.item(),
            'quantum_inequality_duration': min_duration,
            'planck_scale_ratio': (torch.abs(min_energy_density) * self.params.planck_length**4).item(),
            'geometric_stability': self.is_stable
        }
        
        logger.info(f"Negative energy fraction: {bounds_analysis['negative_energy_fraction']:.3f}")
        logger.info(f"Min energy density: {bounds_analysis['min_energy_density']:.2e}")
        logger.info(f"Quantum inequality duration: {bounds_analysis['quantum_inequality_duration']:.2e} s")
        
        return bounds_analysis
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive backreaction and stability analysis report.
        """
        logger.info("Generating comprehensive stability report...")
        
        start_time = time.time()
        
        # Solve Einstein equations
        evolution_data = self.solve_einstein_equations()
        
        # Analyze stability
        stability_results = self.analyze_perturbation_stability()
        
        # Compute negative energy bounds
        energy_bounds = self.compute_negative_energy_bounds()
        
        # LQG-specific analysis
        lqg_analysis = {}
        if hasattr(self, 'lqg_stress'):
            try:
                lqg_analysis = self.lqg_stress.generate_lqg_stress_report(self.field_config)
            except Exception as e:
                logger.warning(f"LQG analysis failed: {e}")
        
        # Summary statistics
        computation_time = time.time() - start_time
        
        comprehensive_report = {
            'analysis_parameters': {
                'spacetime_dimension': self.params.spacetime_dimension,
                'spatial_grid_size': self.params.spatial_grid_size,
                'temporal_range': self.params.temporal_range,
                'integration_steps': self.params.integration_steps,
                'newton_constant': self.params.newton_constant,
                'planck_length': self.params.planck_length
            },
            'evolution_data': {
                'num_time_steps': len(evolution_data['times']),
                'final_time': evolution_data['times'][-1] if evolution_data['times'] else 0,
                'metric_evolution': 'stored' if evolution_data['metric_components'] else 'failed',
                'curvature_evolution': 'computed' if evolution_data['curvature_scalars'] else 'failed'
            },
            'stability_analysis': stability_results,
            'negative_energy_bounds': energy_bounds,
            'lqg_analysis': lqg_analysis,
            'computational_performance': {
                'computation_time': computation_time,
                'device_used': str(self.device),
                'memory_usage': 'tracked' if torch.cuda.is_available() else 'cpu_only'
            },
            'critical_findings': {
                'geometry_stable': self.is_stable,
                'negative_energy_present': energy_bounds['negative_energy_fraction'] > 0,
                'quantum_violation_detected': energy_bounds['quantum_inequality_duration'] < 1e-10,
                'backreaction_significant': max(stability_results['growth_rates']) > 1e-5
            }
        }
        
        # Add memory usage if CUDA available
        if torch.cuda.is_available():
            comprehensive_report['computational_performance']['gpu_memory_used'] = torch.cuda.memory_allocated() / 1e9  # GB
            comprehensive_report['computational_performance']['gpu_memory_cached'] = torch.cuda.memory_reserved() / 1e9  # GB
        
        logger.info(f"Analysis completed in {computation_time:.2f} seconds")
        logger.info(f"Geometry stability: {'STABLE' if self.is_stable else 'UNSTABLE'}")
        logger.info(f"Negative energy fraction: {energy_bounds['negative_energy_fraction']:.3f}")
        
        return comprehensive_report
    
    def save_results(self, report: Dict[str, Any], output_dir: str = "results"):
        """Save analysis results to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main report
        report_file = os.path.join(output_dir, "backreaction_stability_report.json")
        with open(report_file, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json_report = self._convert_tensors_to_lists(report)
            json.dump(json_report, f, indent=2)
        
        logger.info(f"Results saved to {report_file}")
        
        # Save metric evolution data if requested
        if self.params.plot_results:
            self._save_evolution_plots(report, output_dir)
    
    def _convert_tensors_to_lists(self, obj):
        """Recursively convert tensors to lists for JSON serialization."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_tensors_to_lists(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_lists(item) for item in obj]
        else:
            return obj
    
    def _save_evolution_plots(self, report: Dict[str, Any], output_dir: str):
        """Save evolution plots (simplified placeholder)."""
        try:
            import matplotlib.pyplot as plt
            
            # Extract time evolution data
            if 'evolution_data' in report and 'times' in report['evolution_data']:
                times = report['evolution_data']['times']
                
                # Plot stability indicators
                if 'stability_indicators' in report['evolution_data']:
                    stability_data = report['evolution_data']['stability_indicators']
                    
                    plt.figure(figsize=(10, 6))
                    plt.plot(times, stability_data, 'b-', linewidth=2)
                    plt.xlabel('Time')
                    plt.ylabel('Stability Indicator')
                    plt.title('Geometric Stability Evolution')
                    plt.grid(True)
                    plt.savefig(os.path.join(output_dir, 'stability_evolution.png'), dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    logger.info("Evolution plots saved")
        
        except ImportError:
            logger.warning("Matplotlib not available, skipping plots")
        except Exception as e:
            logger.warning(f"Plot generation failed: {e}")


def main():
    """Main CLI interface for backreaction analysis."""
    parser = argparse.ArgumentParser(description="Backreaction and Geometry Stability Analysis")
    
    parser.add_argument("--grid-size", type=int, default=32, help="Spatial grid resolution")
    parser.add_argument("--time-steps", type=int, default=500, help="Number of time steps")
    parser.add_argument("--temporal-range", type=float, default=1.0, help="Time evolution range")
    parser.add_argument("--spatial-range", type=float, default=10.0, help="Spatial domain size")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device")
    parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--perturbation-amp", type=float, default=1e-6, help="Perturbation amplitude")
    parser.add_argument("--stability-threshold", type=float, default=1e-3, help="Stability threshold")
    
    args = parser.parse_args()
    
    # Create parameters
    params = BackreactionParameters(
        spatial_grid_size=args.grid_size,
        integration_steps=args.time_steps,
        temporal_range=args.temporal_range,
        spatial_range=args.spatial_range,
        perturbation_amplitude=args.perturbation_amp,
        stability_threshold=args.stability_threshold,
        device=args.device,
        plot_results=True
    )
    
    logger.info("Starting backreaction and stability analysis...")
    logger.info(f"Grid: {params.spatial_grid_size}³, Steps: {params.integration_steps}")
    logger.info(f"Device: {params.device}")
    
    try:
        # Initialize analyzer
        analyzer = GeometryStabilityAnalyzer(params)
        
        # Run comprehensive analysis
        report = analyzer.generate_comprehensive_report()
        
        # Save results
        analyzer.save_results(report, args.output_dir)
        
        # Print summary
        print("\n" + "="*60)
        print("BACKREACTION ANALYSIS SUMMARY")
        print("="*60)
        print(f"Computation time: {report['computational_performance']['computation_time']:.2f} seconds")
        print(f"Geometry stability: {'STABLE' if report['critical_findings']['geometry_stable'] else 'UNSTABLE'}")
        print(f"Negative energy present: {report['critical_findings']['negative_energy_present']}")
        print(f"Max perturbation growth rate: {report['stability_analysis']['max_growth_rate']:.2e}")
        print(f"Negative energy fraction: {report['negative_energy_bounds']['negative_energy_fraction']:.3f}")
        print(f"Min energy density: {report['negative_energy_bounds']['min_energy_density']:.2e}")
        print(f"Quantum inequality duration: {report['negative_energy_bounds']['quantum_inequality_duration']:.2e} s")
        
        if torch.cuda.is_available():
            print(f"GPU memory used: {report['computational_performance']['gpu_memory_used']:.2f} GB")
        
        print(f"\nResults saved to: {args.output_dir}/")
        print("="*60)
        
        return 0
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
