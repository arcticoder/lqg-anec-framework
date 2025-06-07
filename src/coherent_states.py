# src/coherent_states.py

import numpy as np
from typing import Dict, Any, Optional
import networkx as nx

class CoherentState:
    """
    Build weave/heat-kernel coherent states peaked on flat geometry.
    
    This class implements the construction of semi-classical coherent states
    in Loop Quantum Gravity that approximate flat Minkowski spacetime with
    small perturbations.
    """

    def __init__(self, graph: 'SpinNetwork', alpha: float, hbar: float = 1.0):
        """
        Initialize coherent state on a spin network graph.
        
        :param graph: spin-network graph object
        :param alpha: spread parameter (ℏ-scale) controlling coherence width
        :param hbar: reduced Planck constant (units)
        """
        self.graph = graph
        self.alpha = alpha
        self.hbar = hbar
        self._amplitudes = {}
        self._normalization = 1.0

    def weave_state(self) -> Dict[Any, complex]:
        """
        Construct the weave state amplitudes on each edge.
        
        The weave state is a semiclassical approximation where edge lengths
        are peaked around their classical values with Gaussian spreads.
        
        Returns:
            Dict mapping edge_id → complex amplitude
        """
        amplitudes = {}
        total_weight = 0.0
        
        for edge in self.graph.edges:
            # Gaussian weight around spatial length = 1 (Planck units)
            L = self.graph.edge_length(edge)
            weight = np.exp(-(L - 1.0)**2 / (2 * self.alpha**2))
            
            # Add phase factor for quantum coherence
            phase = np.exp(1j * L / self.hbar)
            amplitudes[edge] = weight * phase
            total_weight += abs(weight)**2
        
        # Store normalization
        self._normalization = np.sqrt(total_weight)
        
        return amplitudes

    def heat_kernel_state(self, t: float = 0.1) -> Dict[Any, complex]:
        """
        Construct heat kernel coherent state using diffusion kernel.
        
        :param t: diffusion time parameter
        :return: Dict mapping edge_id → complex amplitude
        """
        amplitudes = {}
        
        for edge in self.graph.edges:
            # Heat kernel on discrete graph
            neighbors = list(self.graph.get_edge_neighbors(edge))
            n_neighbors = len(neighbors)
            
            # Discrete heat kernel approximation
            heat_weight = np.exp(-n_neighbors * t / self.alpha**2)
            amplitudes[edge] = heat_weight
            
        return amplitudes

    def peak_on_flat(self) -> 'SpinNetwork':
        """
        Adjust node labels/coherent data to approximate flat metric.
        
        Returns:
            Modified spin network with coherent state amplitudes
        """
        weave = self.weave_state()
        
        # Normalize amplitudes
        for edge in weave:
            weave[edge] /= self._normalization
            
        # Assign to graph
        self.graph.assign_amplitudes(weave)
        
        # Set flat metric data at nodes
        for node in self.graph.nodes:
            self.graph.set_metric_data(node, self._flat_metric_data())
            
        return self.graph

    def expectation_value(self, operator) -> complex:
        """
        Compute expectation value of an operator in this coherent state.
        
        :param operator: Operator object with apply() method
        :return: Complex expectation value
        """
        if not hasattr(self, '_amplitudes') or not self._amplitudes:
            self._amplitudes = self.weave_state()
            
        return operator.apply(self.graph, self._amplitudes)

    def overlap(self, other: 'CoherentState') -> complex:
        """
        Compute overlap between two coherent states.
        
        :param other: Another CoherentState
        :return: Complex overlap <ψ₁|ψ₂>
        """
        if not self._amplitudes:
            self._amplitudes = self.weave_state()
        if not other._amplitudes:
            other._amplitudes = other.weave_state()
            
        overlap = 0.0
        for edge in self.graph.edges:
            if edge in other._amplitudes:
                overlap += np.conj(self._amplitudes[edge]) * other._amplitudes[edge]
                
        return overlap / (self._normalization * other._normalization)

    def _flat_metric_data(self) -> Dict[str, float]:
        """Generate metric data corresponding to flat spacetime."""
        return {
            'g00': -1.0,
            'g11': 1.0, 
            'g22': 1.0,
            'g23': 1.0,
            'det_g': -1.0
        }

    def coherence_length(self) -> float:
        """
        Estimate the coherence length scale of this state.
        
        Returns:
            Characteristic length scale in Planck units
        """
        return self.alpha * np.sqrt(self.hbar)

    def classical_limit(self, scaling_factor: float = 1e-3) -> 'CoherentState':
        """
        Take classical limit by scaling ℏ → 0.
        
        :param scaling_factor: Factor to scale hbar by
        :return: New CoherentState in classical limit
        """
        classical_hbar = self.hbar * scaling_factor
        return CoherentState(self.graph, self.alpha, classical_hbar)
