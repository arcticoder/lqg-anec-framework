"""
3+1D Polymer-Quantized Matter Coupling module.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass


@dataclass
class Field3DConfig:
    """Configuration for 3D field evolution."""
    grid_size: Tuple[int, int, int] = (64, 64, 64)
    dx: float = 0.05
    dt: float = 0.001
    epsilon: float = 0.01  # Polymer scale
    mass: float = 1.0
    total_time: float = 0.2


class PolymerField3D:
    """3+1D Polymer-corrected scalar field implementation."""

    def __init__(self, config: Field3DConfig):
        self.config = config
        self.nx, self.ny, self.nz = config.grid_size
        self.dx = config.dx
        self.dt = config.dt
        self.epsilon = config.epsilon
        self.mass = config.mass

        # Initialize coordinate arrays
        x = np.linspace(-1, 1, self.nx) * config.dx * self.nx / 2
        y = np.linspace(-1, 1, self.ny) * config.dx * self.ny / 2
        z = np.linspace(-1, 1, self.nz) * config.dx * self.nz / 2
        self.X, self.Y, self.Z = np.meshgrid(x, y, z, indexing='ij')

    def initialize_fields(self, initial_profile: callable) -> Tuple[np.ndarray, np.ndarray]:
        """Initialize scalar field and momentum."""
        phi = initial_profile(self.X, self.Y, self.Z)
        pi = np.zeros_like(phi)  # Start with zero momentum
        return phi, pi

    def compute_laplacian(self, field: np.ndarray) -> np.ndarray:
        """Compute discrete Laplacian."""
        laplacian = np.zeros_like(field)

        # Interior points
        laplacian[1:-1, 1:-1, 1:-1] = (
            (field[2:, 1:-1, 1:-1] - 2 * field[1:-1, 1:-1, 1:-1] + field[:-2, 1:-1, 1:-1]) / self.dx**2 +
            (field[1:-1, 2:, 1:-1] - 2 * field[1:-1, 1:-1, 1:-1] + field[1:-1, :-2, 1:-1]) / self.dx**2 +
            (field[1:-1, 1:-1, 2:] - 2 * field[1:-1, 1:-1, 1:-1] + field[1:-1, 1:-1, :-2]) / self.dx**2
        )

        return laplacian

    def evolve_step(self, phi: np.ndarray, pi: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Evolve fields by one time step."""
        # Compute Laplacian
        laplacian_phi = self.compute_laplacian(phi)

        # Polymer-corrected kinetic term
        phi_polymer = 2 * np.sin(phi / self.epsilon) / self.epsilon

        # Update equations (simplified leapfrog)
        phi_new = phi + self.dt * pi
        pi_new = pi + self.dt * (laplacian_phi - self.mass**2 * phi - phi_polymer)

        return phi_new, pi_new

    def compute_stress_energy(self, phi: np.ndarray, pi: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute stress-energy tensor components."""
        # T00 component (energy density)
        grad_phi_sq = (
            np.gradient(phi, self.dx, axis=0)**2 +
            np.gradient(phi, self.dx, axis=1)**2 +
            np.gradient(phi, self.dx, axis=2)**2
        )

        T00 = 0.5 * (pi**2 + grad_phi_sq + self.mass**2 * phi**2)

        return {"T00": T00, "mean_T00": np.mean(T00)}
