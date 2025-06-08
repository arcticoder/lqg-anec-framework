"""
Warp Bubble Optimizer - Advanced spacetime manipulation framework

This package provides tools for simulating and optimizing warp bubble geometries
using various negative energy sources including Ghost/Phantom EFT and metamaterial
Casimir configurations.
"""

from .solver import WarpBubbleSolver
from .energy_sources import EnergySource, GhostCondensateEFT, MetamaterialCasimirSource
from .mesh_generator import SphericalMeshGenerator
from .stability_analyzer import StabilityAnalyzer
from .visualization import WarpBubbleVisualizer

__version__ = "1.0.0"
__author__ = "LQG-ANEC Research Team"

__all__ = [
    'WarpBubbleSolver',
    'EnergySource', 
    'GhostCondensateEFT',
    'MetamaterialCasimirSource',
    'SphericalMeshGenerator',
    'StabilityAnalyzer',
    'WarpBubbleVisualizer'
]
