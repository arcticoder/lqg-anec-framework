#!/usr/bin/env python3
"""
3D Mesh-Based Warp Bubble Solver

This module provides tools for validating warp bubble configurations
using 3D meshes and finite element methods. Integrates with the
energy source interface to test different negative energy sources.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
import warnings
from dataclasses import dataclass
import time

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False
    warnings.warn("PyVista not available. 3D visualization will be limited.")

try:
    # Try to import FEniCS components
    from dolfin import *
    HAS_FENICS = True
except ImportError:
    HAS_FENICS = False
    warnings.warn("FEniCS not available. Using fallback mesh generation.")

# Import our energy source interface
try:
    from .energy_source_interface import EnergySource
except ImportError:
    from energy_source_interface import EnergySource

@dataclass
class WarpBubbleResult:
    """Results from warp bubble simulation."""
    success: bool
    energy_total: float
    stability: float
    bubble_radius: float
    max_negative_density: float
    min_negative_density: float
    execution_time: float
    mesh_nodes: int
    source_name: str
    parameters: Dict[str, Any]
    energy_profile: Optional[np.ndarray] = None
    coordinates: Optional[np.ndarray] = None

class SimpleMeshGenerator:
    """
    Fallback mesh generator when FEniCS is not available.
    Creates structured spherical grids for warp bubble analysis.
    """
    
    @staticmethod
    def create_spherical_mesh(radius: float, n_radial: int = 50, 
                            n_theta: int = 30, n_phi: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create a structured spherical mesh.
        
        Args:
            radius: Outer radius of sphere
            n_radial: Number of radial divisions
            n_theta: Number of theta (polar) divisions
            n_phi: Number of phi (azimuthal) divisions
            
        Returns:
            Tuple of (coordinates, connectivity)
        """
        # Create spherical coordinates
        r = np.linspace(0.1, radius, n_radial)  # Avoid r=0 singularity
        theta = np.linspace(0, np.pi, n_theta)
        phi = np.linspace(0, 2*np.pi, n_phi)
        
        # Create coordinate arrays
        R, THETA, PHI = np.meshgrid(r, theta, phi, indexing='ij')
        
        # Convert to Cartesian
        X = R * np.sin(THETA) * np.cos(PHI)
        Y = R * np.sin(THETA) * np.sin(PHI)
        Z = R * np.cos(THETA)
        
        # Flatten coordinates
        coords = np.column_stack([
            X.flatten(),
            Y.flatten(), 
            Z.flatten()
        ])
        
        # Simple connectivity (for visualization)
        n_points = coords.shape[0]
        connectivity = np.arange(n_points).reshape(-1, 1)
        
        return coords, connectivity

class WarpBubbleSolver:
    """
    3D mesh-based warp bubble solver with multiple energy source support.
    
    Provides validation and analysis of warp bubble configurations
    using finite element methods or structured grids.
    """
    
    def __init__(self, metric_ansatz: str = "4d", use_fenics: bool = True):
        """
        Initialize the warp bubble solver.
        
        Args:
            metric_ansatz: Type of metric ansatz ("4d", "alcubierre", "simple")
            use_fenics: Whether to use FEniCS for advanced meshing
        """
        self.metric_ansatz = metric_ansatz
        self.use_fenics = use_fenics and HAS_FENICS
        
        # Solver state
        self.mesh_coords = None
        self.mesh_connectivity = None
        self.energy_profile = None
        self.last_result = None
        
    def generate_mesh(self, radius: float, resolution: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate 3D mesh for warp bubble domain.
        
        Args:
            radius: Domain radius (m)
            resolution: Mesh resolution parameter
            
        Returns:
            Tuple of (coordinates, connectivity)
        """
        if self.use_fenics:
            return self._generate_fenics_mesh(radius, resolution)
        else:
            return self._generate_simple_mesh(radius, resolution)
    
    def _generate_fenics_mesh(self, radius: float, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh using FEniCS/mshr."""
        try:
            from mshr import Sphere, generate_mesh
            
            # Create sphere domain
            domain = Sphere(Point(0, 0, 0), radius)
            
            # Generate mesh
            mesh = generate_mesh(domain, resolution)
            
            # Extract coordinates and connectivity
            coords = mesh.coordinates()
            cells = mesh.cells()
            
            self.mesh_coords = coords
            self.mesh_connectivity = cells
            
            return coords, cells
            
        except ImportError:
            warnings.warn("FEniCS/mshr not available, falling back to simple mesh")
            return self._generate_simple_mesh(radius, resolution)
    
    def _generate_simple_mesh(self, radius: float, resolution: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate mesh using simple structured grid."""
        generator = SimpleMeshGenerator()
        coords, connectivity = generator.create_spherical_mesh(
            radius, n_radial=resolution, n_theta=resolution//2, n_phi=resolution//2
        )
        
        self.mesh_coords = coords
        self.mesh_connectivity = connectivity
        
        return coords, connectivity
    
    def compute_energy_profile(self, energy_source: EnergySource,
                             coords: np.ndarray) -> np.ndarray:
        """
        Compute energy density profile on mesh.
        
        Args:
            energy_source: Energy source to evaluate
            coords: Mesh coordinates (N x 3)
            
        Returns:
            Energy density array (N,)
        """
        x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
        energy_density = energy_source.energy_density(x, y, z)
        
        self.energy_profile = energy_density
        return energy_density
    
    def stability_analysis(self, energy_profile: np.ndarray,
                          coords: np.ndarray) -> float:
        """
        Perform simplified stability analysis.
        
        Args:
            energy_profile: Energy density values
            coords: Mesh coordinates
            
        Returns:
            Stability metric (0-1, higher is more stable)
        """
        # Simple stability metric based on energy gradients
        # In a real implementation, this would solve Einstein equations
        
        # Compute energy gradient magnitude
        if len(energy_profile) < 10:
            return 0.5  # Insufficient data
        
        # Remove any NaN or infinite values
        valid_mask = np.isfinite(energy_profile)
        if not np.any(valid_mask):
            return 0.0  # No valid data
            
        valid_coords = coords[valid_mask]
        valid_energy = energy_profile[valid_mask]
        
        if len(valid_energy) < 3:
            return 0.1  # Insufficient valid data
        
        # For shell-like profiles, use a different approach
        # Check if energy profile looks like a shell (mostly zeros with some negative values)
        negative_fraction = np.sum(valid_energy < 0) / len(valid_energy)
        
        if negative_fraction <= 0.1:  # Shell-like profile (including exactly 0.1)
            # For shells, stability depends on energy confinement
            if np.any(valid_energy < 0):
                energy_range = np.max(valid_energy) - np.min(valid_energy)
                if energy_range > 0:
                    # Stability based on energy localization
                    negative_energy = valid_energy[valid_energy < 0]
                    energy_std = np.std(negative_energy)
                    energy_mean = np.abs(np.mean(negative_energy))
                    
                    if energy_mean > 0:
                        stability = 1.0 / (1.0 + energy_std / energy_mean)
                        return max(0.1, min(1.0, stability))
                return 0.3  # Some negative energy but poorly characterized
            else:
                return 0.1  # No negative energy
        
        # For continuous profiles, use gradient analysis
        # Approximate gradient using nearest neighbors
        r = np.sqrt(np.sum(valid_coords**2, axis=1))
        
        # Sort by radius for gradient computation
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        energy_sorted = valid_energy[sort_idx]
        
        # Handle duplicate radii to avoid divide by zero
        unique_r, unique_idx = np.unique(r_sorted, return_index=True)
        if len(unique_r) < 3:
            return 0.2  # Too few unique radii
            
        unique_energy = energy_sorted[unique_idx]
        
        try:
            # Compute gradient with finite differences
            if len(unique_r) >= 3:
                grad_energy = np.gradient(unique_energy, unique_r)
                # Remove infinite or NaN gradients
                valid_grad = grad_energy[np.isfinite(grad_energy)]
                
                if len(valid_grad) == 0:
                    return 0.1
                    
                # Stability metric: inverse of maximum gradient magnitude
                max_grad = np.max(np.abs(valid_grad))
                if max_grad == 0:
                    return 1.0  # Perfect stability
                    
                stability = 1.0 / (1.0 + max_grad * 1e12)  # Scale factor
                return max(0.0, min(1.0, stability))  # Clamp to [0,1]
            else:
                return 0.3  # Insufficient data for gradient
                
        except Exception:
            return 0.1  # Fallback stability
    
    def solve_poisson_equation(self, energy_profile: np.ndarray,
                              coords: np.ndarray) -> np.ndarray:
        """
        Solve simplified Poisson equation as metric proxy.
        
        ∇²Φ = κ ρ
        
        Args:
            energy_profile: Source term (energy density)
            coords: Mesh coordinates
            
        Returns:
            Solution field Φ
        """
        # For structured mesh, use finite differences
        # This is a simplified version - real implementation would use FEM
        
        n_points = len(energy_profile)
        
        # Simple approach: assume spherical symmetry
        r = np.sqrt(np.sum(coords**2, axis=1))
        
        # Sort by radius
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        rho_sorted = energy_profile[sort_idx]
        
        # Solve 1D Poisson in spherical coordinates
        # d²Φ/dr² + (2/r)dΦ/dr = κρ
        
        dr = np.diff(r_sorted)
        dr = np.append(dr, dr[-1])  # Extend for last point
        
        phi = np.zeros_like(r_sorted)
        kappa = 1.0  # Coupling constant
        
        # Simple integration (Euler method)
        dphi_dr = 0.0
        for i in range(1, len(phi)):
            d2phi_dr2 = kappa * rho_sorted[i] - (2/r_sorted[i]) * dphi_dr
            dphi_dr += d2phi_dr2 * dr[i-1]
            phi[i] = phi[i-1] + dphi_dr * dr[i-1]
        
        # Unsort to match original coordinate order
        phi_unsorted = np.zeros_like(phi)
        phi_unsorted[sort_idx] = phi
        
        return phi_unsorted
    
    def simulate(self, energy_source: EnergySource, 
                radius: float = 10.0, resolution: int = 50,
                speed: Optional[float] = None) -> WarpBubbleResult:
        """
        Run complete warp bubble simulation.
        
        Args:
            energy_source: Negative energy source to test
            radius: Simulation domain radius (m)
            resolution: Mesh resolution
            speed: Desired warp speed (unused in current implementation)
            
        Returns:
            Simulation results
        """
        start_time = time.time()
        
        try:
            # Generate mesh
            coords, connectivity = self.generate_mesh(radius, resolution)
            
            # Compute energy profile
            energy_profile = self.compute_energy_profile(energy_source, coords)
            
            # Analyze stability
            stability = self.stability_analysis(energy_profile, coords)
            
            # Compute total energy
            # Simple integration using mesh volume approximation
            total_energy = energy_source.total_energy(
                (4/3) * np.pi * radius**3
            )
            
            # Solve simplified field equation
            metric_field = self.solve_poisson_equation(energy_profile, coords)
              # Determine success criteria
            max_negative = np.min(energy_profile)
            negative_mask = energy_profile < 0
            min_negative = np.max(energy_profile[negative_mask]) if np.any(negative_mask) else 0.0            
            has_negative = max_negative < -1e-16  # Significant negative energy
            is_stable = stability > 0.1
            params_valid = energy_source.validate_parameters()
            
            success = has_negative and is_stable and params_valid
            
            execution_time = time.time() - start_time
            
            # Create result
            result = WarpBubbleResult(
                success=success,
                energy_total=total_energy,
                stability=stability,
                bubble_radius=radius,
                max_negative_density=max_negative,
                min_negative_density=min_negative if np.any(energy_profile < 0) else 0.0,
                execution_time=execution_time,
                mesh_nodes=len(coords),
                source_name=energy_source.name,
                parameters=energy_source.parameters,
                energy_profile=energy_profile,
                coordinates=coords
            )
            
            self.last_result = result
            return result
            
        except Exception as e:
            warnings.warn(f"Simulation failed: {e}")
            
            # Return failed result
            return WarpBubbleResult(
                success=False,
                energy_total=0.0,
                stability=0.0,
                bubble_radius=radius,
                max_negative_density=0.0,
                min_negative_density=0.0,
                execution_time=time.time() - start_time,
                mesh_nodes=0,
                source_name=energy_source.name,
                parameters=energy_source.parameters
            )
    
    def visualize_result(self, result: WarpBubbleResult, 
                        save_path: Optional[str] = None) -> None:
        """
        Visualize simulation results.
        
        Args:
            result: Simulation result to visualize
            save_path: Optional path to save visualization
        """
        if not HAS_PYVISTA or result.coordinates is None:
            self._plot_matplotlib(result, save_path)
        else:
            self._plot_pyvista(result, save_path)
    
    def _plot_matplotlib(self, result: WarpBubbleResult, 
                        save_path: Optional[str] = None) -> None:
        """Fallback matplotlib visualization."""
        if result.coordinates is None or result.energy_profile is None:
            print(f"No data to plot for {result.source_name}")
            return
            
        coords = result.coordinates
        energy = result.energy_profile
        
        # Create radial plot
        r = np.sqrt(np.sum(coords**2, axis=1))
        
        plt.figure(figsize=(12, 5))
        
        # Energy vs radius
        plt.subplot(1, 2, 1)
        plt.scatter(r, energy, alpha=0.6, s=1)
        plt.xlabel('Radius (m)')
        plt.ylabel('Energy Density (J/m³)')
        plt.title(f'{result.source_name}: Energy Profile')
        plt.grid(True)
        
        # Summary statistics
        plt.subplot(1, 2, 2)
        stats = [
            f"Total Energy: {result.energy_total:.2e} J",
            f"Stability: {result.stability:.3f}",
            f"Max Negative: {result.max_negative_density:.2e} J/m³",
            f"Execution Time: {result.execution_time:.3f} s",
            f"Mesh Nodes: {result.mesh_nodes}",
            f"Success: {result.success}"
        ]
        
        plt.text(0.1, 0.9, '\n'.join(stats), transform=plt.gca().transAxes,
                verticalalignment='top', fontfamily='monospace')
        plt.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        plt.show()
    
    def _plot_pyvista(self, result: WarpBubbleResult,
                     save_path: Optional[str] = None) -> None:
        """PyVista 3D visualization."""
        coords = result.coordinates
        energy = result.energy_profile
        
        # Create PyVista mesh
        grid = pv.PolyData(coords)
        grid["Energy_Density"] = energy
        
        # Create plotter
        p = pv.Plotter()
        p.add_mesh(grid, scalars="Energy_Density", 
                  point_size=5, render_points_as_spheres=True,
                  cmap='RdBu_r', clim=[energy.min(), 0])
        
        p.add_title(f"{result.source_name}: Energy Density")
        p.show_grid()
        
        if save_path:
            p.screenshot(save_path)
            
        p.show()

def compare_energy_sources(sources: List[EnergySource], 
                         radius: float = 10.0,
                         resolution: int = 50) -> Dict[str, WarpBubbleResult]:
    """
    Compare multiple energy sources side by side.
    
    Args:
        sources: List of energy sources to compare
        radius: Simulation domain radius
        resolution: Mesh resolution
        
    Returns:
        Dictionary mapping source names to results
    """
    solver = WarpBubbleSolver()
    results = {}
    
    print("Comparing energy sources...")
    print("=" * 60)
    
    for source in sources:
        print(f"Testing {source.name}...")
        result = solver.simulate(source, radius, resolution)
        results[source.name] = result
        
        print(f"  Success: {result.success}")
        print(f"  Total Energy: {result.energy_total:.2e} J")
        print(f"  Stability: {result.stability:.3f}")
        print(f"  Max Negative Density: {result.max_negative_density:.2e} J/m³")
        print(f"  Execution Time: {result.execution_time:.3f} s")
        print()
    
    return results

# Example usage
if __name__ == "__main__":
    from .energy_source_interface import GhostCondensateEFT, MetamaterialCasimirSource
    
    # Create energy sources
    ghost = GhostCondensateEFT(M=1000, alpha=0.01, beta=0.1)
    meta = MetamaterialCasimirSource(epsilon=-2.0, mu=-1.5, n_layers=100)
    
    # Compare sources
    results = compare_energy_sources([ghost, meta], radius=10.0, resolution=30)
    
    # Visualize best result
    best_source = max(results.keys(), key=lambda k: results[k].stability)
    print(f"\nBest performing source: {best_source}")
    
    solver = WarpBubbleSolver()
    solver.visualize_result(results[best_source])
