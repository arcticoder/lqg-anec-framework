"""
Ghost-Condensate Effective Field Theory Module

UV-complete ghost-condensate EFT with ANEC integral computation and stability analysis.
Implements controlled NEC violation through ghost fields with cutoff regularization.

Key Features:
- Ghost scalar field with derivative self-interactions
- UV completion through higher-derivative terms
- ANEC integral computation with violation mechanisms
- Stability analysis for ghost condensate configurations
- GPU-optimized tensor operations for massive parameter sweeps

Theory Background:
- Ghost condensate: φ field with kinetic term -1/2 (∂φ)²
- Self-interaction: P(X) where X = -1/2 (∂φ)²
- ANEC violation through negative pressure states
- UV completion suppresses ghost instabilities at high energy
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class GhostEFTParameters:
    """Parameters for ghost-condensate EFT configuration."""
    # Field parameters
    phi_0: float = 1.0              # Background field value
    lambda_ghost: float = 0.1       # Ghost coupling strength
    
    # UV completion parameters
    cutoff_scale: float = 10.0      # UV cutoff scale Λ
    higher_deriv_coeff: float = 0.01 # Higher derivative coefficient
    
    # Spacetime parameters
    temporal_range: float = 1.0     # Time coordinate range
    spatial_range: float = 2.0      # Spatial coordinate range
    grid_size: int = 128            # Grid resolution
    
    # ANEC parameters
    anec_kernel_width: float = 0.1  # ANEC averaging kernel width
    anec_boost_velocity: float = 0.5 # Boost velocity for ANEC
    
    # Numerical parameters
    dt: float = 0.01               # Time step
    dx: float = 0.01               # Spatial step
    device: str = "cuda"           # Computation device


class GhostCondensateEFT:
    """
    Ghost-condensate effective field theory implementation.
    
    Implements UV-complete ghost scalar theory with controlled ANEC violation.
    Optimized for GPU computation with vectorized operations.
    """
    
    def __init__(self, params: GhostEFTParameters):
        """Initialize ghost condensate EFT system."""
        self.params = params
        self.device = torch.device(params.device if torch.cuda.is_available() else "cpu")
        
        # Initialize coordinate grids
        self._setup_coordinates()
        
        # Initialize field configuration
        self._setup_fields()
        
        # Precompute derivatives and operators
        self._setup_operators()
        
        logger.info(f"Ghost-condensate EFT initialized on {self.device}")
        logger.info(f"Grid: {params.grid_size}³, Cutoff: {params.cutoff_scale}")
    
    def _setup_coordinates(self):
        """Setup spacetime coordinate grids."""
        # Time coordinate
        t_vals = torch.linspace(-self.params.temporal_range/2, 
                               self.params.temporal_range/2,
                               self.params.grid_size, device=self.device)
        
        # Spatial coordinates (2+1D)
        x_vals = torch.linspace(-self.params.spatial_range/2,
                               self.params.spatial_range/2,
                               self.params.grid_size, device=self.device)
        y_vals = torch.linspace(-self.params.spatial_range/2,
                               self.params.spatial_range/2,
                               self.params.grid_size, device=self.device)
        
        # Create meshgrids
        self.T, self.X, self.Y = torch.meshgrid(t_vals, x_vals, y_vals, indexing='ij')
        
        # Store coordinate arrays
        self.coords = {
            't': self.T,
            'x': self.X, 
            'y': self.Y
        }
    
    def _setup_fields(self):
        """Initialize ghost scalar field configuration."""
        # Background ghost condensate
        self.phi_bg = torch.full_like(self.T, self.params.phi_0)
        
        # Ghost field fluctuations (initially zero)
        self.phi_fluct = torch.zeros_like(self.T)
        
        # Total field
        self.phi = self.phi_bg + self.phi_fluct
        
        # Field derivatives (computed dynamically)
        self.dphi_dt = None
        self.dphi_dx = None
        self.dphi_dy = None
        self.X_kinetic = None  # Kinetic invariant X = -1/2 (∂φ)²
    
    def _setup_operators(self):
        """Setup differential operators for field equations."""
        # Finite difference operators
        self.dt = self.params.dt
        self.dx = self.params.dx
        self.dy = self.params.dx  # Assuming isotropic spacing
        
        # Laplacian stencils for spatial derivatives
        self.laplacian_kernel = torch.tensor([[[0, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 0]],
                                             [[0, 1, 0],
                                              [1, -6, 1],
                                              [0, 1, 0]],
                                             [[0, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 0]]], 
                                            dtype=torch.float32, device=self.device)
        
        # Gradient kernels
        self.grad_x_kernel = torch.tensor([[[0, 0, 0]],
                                          [[-0.5, 0, 0.5]],
                                          [[0, 0, 0]]], 
                                         dtype=torch.float32, device=self.device) / self.dx        
        self.grad_y_kernel = torch.tensor([[[0, -0.5, 0]],
                                          [[0, 0, 0]],
                                          [[0, 0.5, 0]]], 
                                         dtype=torch.float32, device=self.device) / self.dy
    
    def compute_field_derivatives(self):
        """Compute field derivatives using finite differences."""
        # Time derivative (central difference)
        self.dphi_dt = torch.gradient(self.phi, dim=0)[0] / self.dt
        
        # Spatial derivatives (central difference)
        self.dphi_dx = torch.gradient(self.phi, dim=1)[0] / self.dx
        self.dphi_dy = torch.gradient(self.phi, dim=2)[0] / self.dy
        
        # Kinetic invariant X = -1/2 (∂μφ ∂^μ φ) = -1/2 [-(∂φ/∂t)² + (∇φ)²]
        self.X_kinetic = -0.5 * (-self.dphi_dt**2 + self.dphi_dx**2 + self.dphi_dy**2)
    
    def ghost_lagrangian_density(self) -> torch.Tensor:
        """
        Compute ghost-condensate Lagrangian density.
        
        L = X + P(X) + UV completion terms
        where X = -1/2 (∂φ)² and P(X) is the self-interaction.
        """
        if self.X_kinetic is None:
            self.compute_field_derivatives()
        
        # Base kinetic term
        L_kinetic = self.X_kinetic
          # Ghost self-interaction P(X)
        # Use P(X) = λ X^2 for simplicity (quartic self-interaction)
        P_X = self.params.lambda_ghost * self.X_kinetic**2
        
        # UV completion: higher derivative terms
        # Simple Laplacian approximation using second derivatives
        d2phi_dx2 = torch.gradient(torch.gradient(self.phi, dim=1)[0], dim=1)[0] / self.dx**2
        d2phi_dy2 = torch.gradient(torch.gradient(self.phi, dim=2)[0], dim=2)[0] / self.dy**2
        phi_laplacian = d2phi_dx2 + d2phi_dy2
        
        L_uv = self.params.higher_deriv_coeff * phi_laplacian**2 / self.params.cutoff_scale**2
        
        return L_kinetic + P_X + L_uv
    
    def stress_energy_tensor(self) -> Dict[str, torch.Tensor]:
        """
        Compute stress-energy tensor components for ghost field.
        
        T_μν = ∂L/∂(∂_μφ) ∂_νφ - η_μν L
        
        Returns dictionary with components T_00, T_01, T_11, etc.
        """
        if self.X_kinetic is None:
            self.compute_field_derivatives()
        
        L_density = self.ghost_lagrangian_density()
        
        # Compute ∂L/∂(∂_μφ) derivatives
        # For L = X + P(X), ∂L/∂(∂_μφ) = -∂_μφ (1 + P'(X))
        P_prime = 2 * self.params.lambda_ghost * self.X_kinetic  # P'(X) for P(X) = λX²
        
        factor = -(1 + P_prime)
        
        # Stress-energy components
        T_00 = factor * self.dphi_dt * self.dphi_dt - L_density
        T_01 = factor * self.dphi_dt * self.dphi_dx
        T_02 = factor * self.dphi_dt * self.dphi_dy
        T_11 = factor * self.dphi_dx * self.dphi_dx + L_density
        T_12 = factor * self.dphi_dx * self.dphi_dy
        T_22 = factor * self.dphi_dy * self.dphi_dy + L_density
        
        return {
            'T_00': T_00,  # Energy density
            'T_01': T_01,  # Energy flux x
            'T_02': T_02,  # Energy flux y
            'T_11': T_11,  # Stress xx
            'T_12': T_12,  # Stress xy
            'T_22': T_22   # Stress yy
        }
    
    def compute_anec_integrand(self, boost_velocity: Optional[float] = None) -> torch.Tensor:
        """
        Compute ANEC integrand ρ(v,λ) = T_μν k^μ k^ν along null geodesics.
        
        For null vector k^μ = (1, v, 0) with boost velocity v.
        """
        if boost_velocity is None:
            boost_velocity = self.params.anec_boost_velocity
        
        stress_tensor = self.stress_energy_tensor()
        
        # Null vector components k^μ = (1, v, 0) 
        k_t, k_x, k_y = 1.0, boost_velocity, 0.0
        
        # ANEC integrand: T_μν k^μ k^ν
        anec_integrand = (stress_tensor['T_00'] * k_t * k_t + 
                         2 * stress_tensor['T_01'] * k_t * k_x +
                         2 * stress_tensor['T_02'] * k_t * k_y +
                         stress_tensor['T_11'] * k_x * k_x +
                         2 * stress_tensor['T_12'] * k_x * k_y +
                         stress_tensor['T_22'] * k_y * k_y)
        
        return anec_integrand
    
    def compute_anec_integral(self, 
                             boost_velocity: Optional[float] = None,
                             kernel_width: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Compute ANEC integral with averaging kernel.
        
        ANEC = ∫ dλ ρ(v,λ) f(λ) where f(λ) is averaging kernel.
        """
        if boost_velocity is None:
            boost_velocity = self.params.anec_boost_velocity
        if kernel_width is None:
            kernel_width = self.params.anec_kernel_width
        
        # Compute ANEC integrand
        anec_integrand = self.compute_anec_integrand(boost_velocity)
        
        # Averaging kernel (Gaussian)
        lambda_coord = self.X  # Use x-coordinate as affine parameter λ
        kernel = torch.exp(-0.5 * (lambda_coord / kernel_width)**2)
        kernel = kernel / torch.trapz(torch.trapz(torch.trapz(kernel, dim=2), dim=1), dim=0)
        
        # ANEC integral over spatial slice
        anec_weighted = anec_integrand * kernel
        
        # Integrate over spatial dimensions for each time slice
        anec_spatial = torch.trapz(torch.trapz(anec_weighted, dim=2), dim=1)
        
        # Total ANEC (integrate over time)
        anec_total = torch.trapz(anec_spatial, dim=0)
        
        return {
            'anec_total': anec_total,
            'anec_spatial': anec_spatial,
            'anec_integrand': anec_integrand,
            'averaging_kernel': kernel
        }
    
    def stability_analysis(self) -> Dict[str, torch.Tensor]:
        """
        Analyze stability of ghost condensate configuration.
        
        Checks for:
        - Ghost instabilities (negative kinetic energy)
        - UV completion effectiveness
        - Field gradients and potential runaway behavior
        """
        if self.X_kinetic is None:
            self.compute_field_derivatives()
        
        stress_tensor = self.stress_energy_tensor()
        
        # Energy density analysis
        energy_density = stress_tensor['T_00']
        negative_energy_fraction = torch.mean((energy_density < 0).float())
        
        # Ghost kinetic energy (should be negative for ghost)
        ghost_kinetic = -0.5 * self.dphi_dt**2  # Ghost kinetic term
        normal_kinetic = 0.5 * (self.dphi_dx**2 + self.dphi_dy**2)  # Spatial gradients
        
        # UV cutoff effectiveness
        field_scale = torch.sqrt(torch.mean(self.phi**2))
        cutoff_ratio = field_scale / self.params.cutoff_scale
        
        # Gradient stability
        grad_magnitude = torch.sqrt(self.dphi_dx**2 + self.dphi_dy**2)
        max_gradient = torch.max(grad_magnitude)
        
        # Effective potential minimum analysis
        potential_value = self.params.lambda_ghost * self.X_kinetic**2
        potential_minimum = torch.min(potential_value)
        
        return {
            'negative_energy_fraction': negative_energy_fraction,
            'ghost_kinetic_energy': ghost_kinetic,
            'normal_kinetic_energy': normal_kinetic,
            'field_scale_ratio': cutoff_ratio,
            'max_gradient': max_gradient,
            'potential_minimum': potential_minimum,
            'total_energy': torch.mean(energy_density),
            'energy_variance': torch.var(energy_density)
        }
    
    def evolve_field_configuration(self, 
                                  initial_perturbation: Optional[torch.Tensor] = None,
                                  evolution_steps: int = 100) -> Dict[str, List[torch.Tensor]]:
        """
        Evolve ghost field configuration in time using equations of motion.
        
        Implements simplified time evolution for stability testing.
        """
        if initial_perturbation is not None:
            self.phi_fluct = initial_perturbation
            self.phi = self.phi_bg + self.phi_fluct
        
        # Storage for evolution history
        evolution_history = {
            'phi': [],
            'energy_density': [],
            'anec_violation': []
        }
        
        for step in range(evolution_steps):
            # Compute current derivatives and stress tensor
            self.compute_field_derivatives()
            stress_tensor = self.stress_energy_tensor()
            
            # Simple time evolution (Euler method)
            # ∂²φ/∂t² = equation of motion from Lagrangian
            phi_ddot = self._compute_equation_of_motion()
            
            # Update field
            self.phi = self.phi + self.dt * self.dphi_dt + 0.5 * self.dt**2 * phi_ddot
            
            # Store evolution data
            if step % 10 == 0:  # Sample every 10 steps
                evolution_history['phi'].append(self.phi.clone())
                evolution_history['energy_density'].append(stress_tensor['T_00'].clone())
                
                # ANEC violation check
                anec_result = self.compute_anec_integral()
                evolution_history['anec_violation'].append(anec_result['anec_total'].clone())
        
        return evolution_history
    
    def _compute_equation_of_motion(self) -> torch.Tensor:
        """
        Compute equation of motion for ghost field.
        
        Derived from varying the action: δS/δφ = 0
        """
        if self.X_kinetic is None:
            self.compute_field_derivatives()        
        # Simplified EOM: □φ + interaction terms = 0
        # For P(X) = λX², this gives complex coupled system
        
        # Spatial Laplacian using gradient approach
        d2phi_dx2 = torch.gradient(torch.gradient(self.phi, dim=1)[0], dim=1)[0] / self.dx**2
        d2phi_dy2 = torch.gradient(torch.gradient(self.phi, dim=2)[0], dim=2)[0] / self.dy**2
        phi_laplacian = d2phi_dx2 + d2phi_dy2
        
        # Interaction contribution (simplified)
        interaction_term = -2 * self.params.lambda_ghost * self.X_kinetic * (
            -self.dphi_dt**2 + self.dphi_dx**2 + self.dphi_dy**2
        )
        
        # UV completion contribution
        uv_term = -self.params.higher_deriv_coeff * phi_laplacian / self.params.cutoff_scale**2
        
        return phi_laplacian + interaction_term + uv_term
    
    def generate_anec_violation_report(self) -> Dict[str, Union[float, torch.Tensor]]:
        """
        Generate comprehensive ANEC violation analysis report.
        """
        logger.info("Generating ANEC violation report...")
        
        # Compute ANEC for multiple boost velocities
        boost_velocities = torch.linspace(0.1, 0.9, 9, device=self.device)
        anec_violations = []
        
        for v in boost_velocities:
            anec_result = self.compute_anec_integral(boost_velocity=v.item())
            anec_violations.append(anec_result['anec_total'].item())
        
        # Stability analysis
        stability = self.stability_analysis()
        
        # Summary statistics
        min_anec = min(anec_violations)
        max_anec = max(anec_violations)
        mean_anec = np.mean(anec_violations)
        
        report = {
            'anec_violations_by_velocity': dict(zip(boost_velocities.cpu().numpy(), anec_violations)),
            'min_anec_violation': min_anec,
            'max_anec_violation': max_anec,
            'mean_anec_violation': mean_anec,
            'negative_energy_fraction': stability['negative_energy_fraction'].item(),
            'field_scale_ratio': stability['field_scale_ratio'].item(),
            'total_energy': stability['total_energy'].item(),
            'ghost_parameters': {
                'lambda_ghost': self.params.lambda_ghost,
                'cutoff_scale': self.params.cutoff_scale,
                'phi_background': self.params.phi_0
            }
        }
        
        logger.info(f"ANEC violation range: [{min_anec:.2e}, {max_anec:.2e}]")
        logger.info(f"Negative energy fraction: {stability['negative_energy_fraction'].item():.3f}")
        
        return report


def scan_ghost_eft_parameters(parameter_ranges: Dict[str, Tuple[float, float]],
                             num_samples: int = 50,
                             device: str = "cuda") -> List[Dict]:
    """
    Systematic parameter scan for ghost-condensate EFT ANEC violations.
    
    Args:
        parameter_ranges: Dictionary of parameter ranges to scan
        num_samples: Number of samples per parameter
        device: Computation device
    
    Returns:
        List of results for each parameter combination
    """
    logger.info(f"Starting ghost EFT parameter scan with {num_samples} samples per parameter")
    
    results = []
    
    # Generate parameter combinations
    param_combinations = []
    for param_name, (min_val, max_val) in parameter_ranges.items():
        param_values = np.linspace(min_val, max_val, num_samples)
        param_combinations.append([(param_name, val) for val in param_values])
    
    # Scan parameters (simplified for demonstration)
    base_params = GhostEFTParameters(device=device)
    
    for i in range(min(num_samples, 20)):  # Limit for demo
        # Create modified parameters
        current_params = GhostEFTParameters(device=device)
        
        # Modify one parameter at a time
        for param_name, (min_val, max_val) in parameter_ranges.items():
            if hasattr(current_params, param_name):
                value = min_val + (max_val - min_val) * i / (num_samples - 1)
                setattr(current_params, param_name, value)
        
        try:
            # Initialize and analyze EFT
            ghost_eft = GhostCondensateEFT(current_params)
            
            # Generate violation report
            report = ghost_eft.generate_anec_violation_report()
            
            # Add parameter configuration
            report['parameter_config'] = {
                param: getattr(current_params, param) 
                for param in parameter_ranges.keys()
                if hasattr(current_params, param)
            }
            
            results.append(report)
            
        except Exception as e:
            logger.warning(f"Parameter combination {i} failed: {e}")
            continue
    
    logger.info(f"Completed parameter scan with {len(results)} successful configurations")
    return results


# Example usage and testing functions
def test_ghost_condensate_basic():
    """Basic test of ghost-condensate EFT functionality."""
    print("Testing Ghost-Condensate EFT...")
    
    # Create test parameters
    params = GhostEFTParameters(
        lambda_ghost=0.1,
        cutoff_scale=5.0,
        grid_size=64,  # Smaller for testing
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Initialize EFT
    ghost_eft = GhostCondensateEFT(params)
    
    # Generate ANEC violation report
    report = ghost_eft.generate_anec_violation_report()
    
    print(f"ANEC violation range: [{report['min_anec_violation']:.2e}, {report['max_anec_violation']:.2e}]")
    print(f"Negative energy fraction: {report['negative_energy_fraction']:.3f}")
    
    return report


if __name__ == "__main__":
    # Run basic test
    test_report = test_ghost_condensate_basic()
    print("Ghost-condensate EFT test completed successfully!")
