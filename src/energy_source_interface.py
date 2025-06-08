#!/usr/bin/env python3
"""
Energy Source Interface for Warp Bubble Simulations

This module defines the interface for negative energy sources used in
3D mesh-based warp bubble validation, including Ghost/Phantom EFT
and Metamaterial Casimir sources.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, Optional
import warnings

class EnergySource(ABC):
    """
    Abstract base class for negative energy sources.
    
    Provides interface for computing energy density profiles
    on 3D meshes for warp bubble validation.
    """
    
    def __init__(self, name: str, parameters: Dict[str, Any]):
        """
        Initialize energy source.
        
        Args:
            name: Human-readable name for the source
            parameters: Configuration parameters for the source
        """
        self.name = name
        self.parameters = parameters
        self._is_initialized = False
        
    @abstractmethod
    def energy_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute energy density at given coordinates.
        
        Args:
            x, y, z: Coordinate arrays (same shape)
            
        Returns:
            Energy density array (same shape as input coordinates)
        """
        pass
    
    @abstractmethod
    def total_energy(self, volume: float) -> float:
        """
        Compute total integrated energy over given volume.
        
        Args:
            volume: Integration volume (m³)
            
        Returns:
            Total energy (J)
        """
        pass
    
    def validate_parameters(self) -> bool:
        """
        Validate source parameters for physical consistency.
        
        Returns:
            True if parameters are valid, False otherwise
        """
        return True  # Default implementation
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get source information summary.
        
        Returns:
            Dictionary with source information
        """
        return {
            'name': self.name,
            'parameters': self.parameters,
            'initialized': self._is_initialized,
            'valid': self.validate_parameters()
        }

class GhostCondensateEFT(EnergySource):
    """
    Ghost/Phantom Effective Field Theory negative energy source.
    
    Based on Discovery 21 optimal parameters:
    M=1000, α=0.01, β=0.1
    """
    
    def __init__(self, M: float = 1000, alpha: float = 0.01, beta: float = 0.1,
                 R0: float = 5.0, sigma: float = 0.2):
        """
        Initialize Ghost EFT source.
        
        Args:
            M: Ghost mass scale (Discovery 21 optimal: 1000)
            alpha: Coupling parameter α (Discovery 21 optimal: 0.01)
            beta: Coupling parameter β (Discovery 21 optimal: 0.1)
            R0: Characteristic radius (m)
            sigma: Gaussian width parameter (m)
        """
        parameters = {
            'M': M,
            'alpha': alpha,
            'beta': beta,
            'R0': R0,
            'sigma': sigma
        }
        super().__init__("Ghost/Phantom EFT", parameters)
        
        # Store parameters as attributes
        self.M = M
        self.alpha = alpha
        self.beta = beta
        self.R0 = R0
        self.sigma = sigma
        
        # Compute amplitude based on Discovery 21 results
        # Peak ANEC violation: -1.418×10⁻¹² W
        self.amplitude = 1.418e-12  # Base amplitude (W)
        
        self._is_initialized = True
    
    def energy_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute Ghost EFT energy density profile.
        
        Uses Gaussian profile centered at bubble wall:
        ρ(r) = -A * exp(-(r-R0)²/(2σ²))
        
        Args:
            x, y, z: Coordinate arrays
            
        Returns:
            Energy density array (J/m³)
        """
        # Compute radial distance
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Gaussian shell profile
        energy_density = -self.amplitude * np.exp(
            -((r - self.R0)**2) / (2 * self.sigma**2)
        )
        
        # Apply EFT corrections based on parameters
        eft_factor = (self.alpha * self.beta) / (1 + (r / self.M)**2)
        energy_density *= eft_factor
        
        return energy_density
    
    def total_energy(self, volume: float) -> float:
        """
        Compute total integrated Ghost EFT energy.
        
        Args:
            volume: Integration volume (m³)
            
        Returns:
            Total energy (J)
        """
        # Analytical integration for Gaussian shell
        # For spherical shell: E_total ≈ -A * σ * sqrt(2π) * 4π * R0²
        eft_factor = (self.alpha * self.beta) / (1 + (self.R0 / self.M)**2)
        total_energy = (-self.amplitude * self.sigma * np.sqrt(2 * np.pi) * 
                       4 * np.pi * self.R0**2 * eft_factor)
        
        return total_energy
    
    def validate_parameters(self) -> bool:
        """
        Validate Ghost EFT parameters.
        
        Returns:
            True if parameters are physically reasonable
        """
        checks = [
            self.M > 0,
            self.alpha > 0,
            self.beta > 0,
            self.R0 > 0,
            self.sigma > 0,
            self.sigma < self.R0,  # Shell width should be < radius
            self.alpha < 1.0,      # Coupling should be perturbative
            self.beta < 1.0
        ]
        return all(checks)

class MetamaterialCasimirSource(EnergySource):
    """
    Metamaterial-enhanced Casimir effect negative energy source.
    
    Based on negative-index metamaterial configurations
    from vacuum engineering discoveries.
    """
    
    def __init__(self, epsilon: float = -2.0, mu: float = -1.5, 
                 cell_size: float = 50e-9, n_layers: int = 100,
                 R0: float = 5.0, shell_thickness: float = 0.1):
        """
        Initialize Metamaterial Casimir source.
        
        Args:
            epsilon: Relative permittivity (negative for metamaterial)
            mu: Relative permeability (negative for metamaterial)
            cell_size: Unit cell size (m)
            n_layers: Number of metamaterial layers
            R0: Shell radius (m)
            shell_thickness: Shell thickness (m)
        """
        parameters = {
            'epsilon': epsilon,
            'mu': mu,
            'cell_size': cell_size,
            'n_layers': n_layers,
            'R0': R0,
            'shell_thickness': shell_thickness
        }
        super().__init__("Metamaterial Casimir", parameters)
        
        # Store parameters
        self.epsilon = epsilon
        self.mu = mu
        self.cell_size = cell_size
        self.n_layers = n_layers
        self.R0 = R0
        self.shell_thickness = shell_thickness
        
        # Compute Casimir energy density scaling
        # Based on vacuum engineering results: ~10⁻⁶ J/m³ base level
        self.base_density = 1e-6  # J/m³
        
        # Enhancement factor from metamaterial negative index
        self.enhancement = abs(epsilon * mu) * n_layers
        
        self._is_initialized = True
    
    def energy_density(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Compute metamaterial Casimir energy density.
        
        Uses spherical shell with enhanced Casimir effect:
        ρ(r) = -A * enhancement * (1 - ((r-R0)/thickness)²) for |r-R0| < thickness
        
        Args:
            x, y, z: Coordinate arrays
            
        Returns:
            Energy density array (J/m³)
        """
        # Compute radial distance
        r = np.sqrt(x**2 + y**2 + z**2)
        
        # Shell profile: negative energy only within shell thickness
        r_shell = np.abs(r - self.R0)
        in_shell = r_shell <= self.shell_thickness
        
        # Parabolic profile within shell
        profile = np.zeros_like(r)
        shell_coord = r_shell[in_shell] / self.shell_thickness
        profile[in_shell] = 1.0 - shell_coord**2
        
        # Apply metamaterial enhancement
        energy_density = -self.base_density * self.enhancement * profile
        
        return energy_density
    
    def total_energy(self, volume: float) -> float:
        """
        Compute total metamaterial Casimir energy.
        
        Args:
            volume: Integration volume (m³)
            
        Returns:
            Total energy (J)
        """
        # Shell volume: 4π * R0² * shell_thickness (thin shell approximation)
        shell_volume = 4 * np.pi * self.R0**2 * self.shell_thickness
        
        # Average energy density in shell (integral of parabolic profile)
        avg_density = (2/3) * self.base_density * self.enhancement
        
        total_energy = -avg_density * shell_volume
        return total_energy
    
    def validate_parameters(self) -> bool:
        """
        Validate metamaterial parameters.
        
        Returns:
            True if parameters are physically reasonable
        """
        checks = [
            self.epsilon < 0,  # Must be negative for metamaterial
            self.mu < 0,       # Must be negative for metamaterial
            self.cell_size > 0,
            self.n_layers > 0,
            self.R0 > 0,
            self.shell_thickness > 0,
            self.shell_thickness < self.R0,  # Shell must fit inside radius
            self.cell_size < self.shell_thickness,  # Unit cells must fit in shell
            abs(self.epsilon) < 100,  # Reasonable material parameters
            abs(self.mu) < 100
        ]
        return all(checks)

def create_energy_source(source_type: str, **kwargs) -> EnergySource:
    """
    Factory function to create energy sources.
    
    Args:
        source_type: Type of source ('ghost', 'metamaterial')
        **kwargs: Parameters for the specific source type
        
    Returns:
        Initialized energy source
    """
    if source_type.lower() in ['ghost', 'ghost_eft', 'phantom']:
        return GhostCondensateEFT(**kwargs)
    elif source_type.lower() in ['metamaterial', 'casimir', 'meta']:
        return MetamaterialCasimirSource(**kwargs)
    else:
        raise ValueError(f"Unknown energy source type: {source_type}")

# Example usage and validation
if __name__ == "__main__":
    # Test Ghost EFT with Discovery 21 optimal parameters
    ghost = GhostCondensateEFT(M=1000, alpha=0.01, beta=0.1)
    print(f"Ghost EFT Info: {ghost.get_info()}")
    
    # Test metamaterial source
    meta = MetamaterialCasimirSource(epsilon=-2.0, mu=-1.5, n_layers=100)
    print(f"Metamaterial Info: {meta.get_info()}")
    
    # Test energy computation on a small grid
    x = np.linspace(-10, 10, 21)
    y = np.linspace(-10, 10, 21)
    z = np.linspace(-10, 10, 21)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    ghost_energy = ghost.energy_density(X, Y, Z)
    meta_energy = meta.energy_density(X, Y, Z)
    
    print(f"Ghost EFT energy range: [{ghost_energy.min():.2e}, {ghost_energy.max():.2e}] J/m³")
    print(f"Metamaterial energy range: [{meta_energy.min():.2e}, {meta_energy.max():.2e}] J/m³")
    
    print(f"Ghost EFT total energy: {ghost.total_energy(1000):.2e} J")
    print(f"Metamaterial total energy: {meta.total_energy(1000):.2e} J")
