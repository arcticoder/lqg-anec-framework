# src/polymer_quantization.py

import numpy as np
from typing import Union, Optional
from math import sin, cos, sinh, cosh

def polymer_correction(value: float, mu: float) -> float:
    """
    Apply LQG polymer modification to a classical quantity.
    Typically: sin(mu·value)/(mu) for bounded operators
    
    :param value: Classical quantity 
    :param mu: Polymer scale parameter
    :return: Polymer-corrected value
    """
    if mu == 0:
        return value
    
    mu_value = mu * value
    if abs(mu_value) < 1e-10:
        # Taylor expansion for small arguments
        return value * (1 - (mu_value)**2/6 + (mu_value)**4/120)
    
    return sin(mu_value) / mu

def polymer_sine(x: float, mu: float) -> float:
    """
    Polymer sine function: sin(μx)/μ
    
    :param x: Input value
    :param mu: Polymer parameter
    :return: Polymer sine value
    """
    return polymer_correction(x, mu)

def polymer_cosine(x: float, mu: float) -> float:
    """
    Polymer cosine function for holonomy corrections.
    
    :param x: Input value  
    :param mu: Polymer parameter
    :return: Polymer cosine value
    """
    if mu == 0:
        return 1.0
    return cos(mu * x)

def inverse_polymer_correction(corrected_value: float, mu: float, 
                              max_iter: int = 100, tol: float = 1e-12) -> float:
    """
    Invert the polymer correction: given sin(μx)/μ, find x
    
    :param corrected_value: The polymer-corrected value sin(μx)/μ
    :param mu: Polymer parameter
    :param max_iter: Maximum Newton-Raphson iterations
    :param tol: Convergence tolerance
    :return: Original value x
    """
    if mu == 0:
        return corrected_value
    
    # Initial guess
    x = corrected_value
    
    for _ in range(max_iter):
        fx = sin(mu * x) / mu - corrected_value
        fpx = cos(mu * x)  # derivative
        
        if abs(fx) < tol:
            break
            
        x_new = x - fx / fpx
        if abs(x_new - x) < tol:
            break
        x = x_new
    
    return x

def polymer_volume_correction(classical_volume: float, mu_bar: float) -> float:
    """
    Volume correction in LQG: |q|^{1/2} → polymer-corrected volume
    
    :param classical_volume: Classical volume √|det(q)|
    :param mu_bar: Dimensionless polymer parameter
    :return: Polymer-corrected volume
    """
    if mu_bar == 0:
        return classical_volume
    
    # Improved prescription: (sin(μ̄√|q|))/(μ̄)
    sqrt_vol = np.sqrt(abs(classical_volume))
    return (sin(mu_bar * sqrt_vol) / mu_bar)**2

def polymer_momentum_correction(momentum: float, mu: float) -> float:
    """
    Momentum operator correction in polymer representation.
    
    :param momentum: Classical momentum
    :param mu: Polymer parameter
    :return: Polymer-corrected momentum
    """
    # For momentum operators: p → (sin(μp))/μ
    return polymer_correction(momentum, mu)

def polymer_kinetic_energy(momentum: float, mass: float, mu: float) -> float:
    """
    Kinetic energy with polymer corrections.
    
    Classical: p²/(2m)
    Polymer: [sin(μp)/μ]²/(2m)
    
    :param momentum: Momentum value
    :param mass: Particle mass
    :param mu: Polymer parameter
    :return: Polymer-corrected kinetic energy
    """
    if mass <= 0:
        raise ValueError("Mass must be positive")
    
    corrected_momentum = polymer_correction(momentum, mu)
    return corrected_momentum**2 / (2 * mass)

def polymer_quantum_inequality_bound(tau: float, mu: float, 
                                   dimension: int = 4) -> float:
    """
    Polymer-modified quantum inequality bound.
    
    Modified Ford-Roman bound with polymer corrections:
    ∫ ⟨T₀₀⟩ f(t) dt ≥ -C/(τ²) × polymer_factor(μ)
    
    :param tau: Sampling timescale
    :param mu: Polymer parameter
    :param dimension: Spacetime dimension
    :return: Modified bound (negative)
    """
    # Classical Ford-Roman constant
    if dimension == 4:
        C_classical = 3.0 / (32 * np.pi**2)
    else:
        # Generalized bound
        C_classical = 1.0 / (8 * np.pi**(dimension/2))
    
    # Polymer modification factor
    if mu == 0:
        polymer_factor = 1.0
    else:
        # Model: sinc(πμ) modification
        polymer_factor = sin(np.pi * mu) / (np.pi * mu)
    
    return -C_classical / tau**2 * polymer_factor

def effective_polymer_hamiltonian(q: np.ndarray, p: np.ndarray, 
                                mu: float, potential_func) -> float:
    """
    Effective Hamiltonian with polymer corrections.
    
    H = Σᵢ [sin(μpᵢ)/μ]²/(2m) + V(q)
    
    :param q: Configuration coordinates
    :param p: Momentum coordinates  
    :param mu: Polymer parameter
    :param potential_func: Potential energy function V(q)
    :return: Total Hamiltonian value
    """
    # Kinetic energy with polymer corrections
    kinetic = 0.0
    for pi in p:
        kinetic += polymer_kinetic_energy(pi, mass=1.0, mu=mu)
    
    # Potential energy (unchanged)
    potential = potential_func(q)
    
    return kinetic + potential

def polymer_scale_hierarchy(planck_length: float = 1.0, 
                          phenomenology_scale: float = 1e-35) -> dict:
    """
    Generate hierarchy of polymer scales for different physics.
    
    :param planck_length: Planck length (default = 1 in Planck units)
    :param phenomenology_scale: Phenomenological scale in meters
    :return: Dictionary of polymer parameters for different scales
    """
    return {
        'planck_scale': 1.0,  # μ ~ 1 at Planck scale
        'phenomenological': phenomenology_scale / planck_length,
        'cosmological': 1e-60,  # For cosmological applications
        'black_hole': 0.1,      # For black hole interiors
        'quantum_bounce': 0.01  # For bounce scenarios
    }

class PolymerOperator:
    """
    Generic polymer operator with configurable prescription.
    """
    
    def __init__(self, mu: float, prescription: str = "sine"):
        """
        :param mu: Polymer parameter
        :param prescription: Type of polymer modification ("sine", "cosine", "tan")
        """
        self.mu = mu
        self.prescription = prescription
    
    def apply(self, value: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """Apply polymer modification to value(s)."""
        if self.prescription == "sine":
            return polymer_correction(value, self.mu)
        elif self.prescription == "cosine":
            return polymer_cosine(value, self.mu)
        elif self.prescription == "tan":
            if self.mu == 0:
                return value
            return np.tan(self.mu * value) / self.mu
        else:
            raise ValueError(f"Unknown prescription: {self.prescription}")
    
    def expectation_value(self, wavefunction, operator_matrix):
        """
        Compute expectation value with polymer corrections.
        
        :param wavefunction: Quantum state
        :param operator_matrix: Operator matrix elements
        :return: Polymer-corrected expectation value
        """
        # Apply polymer correction to operator matrix elements
        corrected_matrix = self.apply(operator_matrix)
        
        # Standard quantum expectation value
        return np.conj(wavefunction).T @ corrected_matrix @ wavefunction
