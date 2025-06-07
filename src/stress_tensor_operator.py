# src/stress_tensor_operator.py

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import cmath

class StressTensorOperator(ABC):
    """
    Abstract base class for defining T_ab operators on spin-network states.
    
    In Loop Quantum Gravity, the stress-energy tensor emerges from
    matter field operators defined on the discrete spin network geometry.
    """

    @abstractmethod
    def apply(self, spin_network, amplitudes: Optional[Dict] = None) -> Dict[Any, complex]:
        """
        Apply stress-tensor operator to a spin network state.
        
        :param spin_network: SpinNetwork object with geometric data
        :param amplitudes: Optional coherent state amplitudes
        :return: Dict mapping locations → T_ab expectation values
        """
        pass

    @abstractmethod  
    def component(self, mu: int, nu: int):
        """
        Get specific component T_μν of the stress tensor.
        
        :param mu, nu: Spacetime indices (0,1,2,3)
        :return: Component operator
        """
        pass


class ScalarFieldStressTensor(StressTensorOperator):
    """
    Stress-energy tensor for a scalar field on spin networks.
    
    For a scalar field φ with Lagrangian L = ½(∂φ)² - ½m²φ²,
    the stress tensor is:
    T_μν = ∂_μφ ∂_νφ - ½g_μν[(∂φ)² + m²φ²]
    """
    
    def __init__(self, mass: float = 0.0, coupling: float = 1.0):
        """
        Initialize scalar field stress tensor.
        
        :param mass: Scalar field mass
        :param coupling: Coupling constant
        """
        self.mass = mass
        self.coupling = coupling
        self.field_values = {}  # vertex → field value φ
        self.field_gradients = {}  # edge → gradient ∂φ
        
    def set_field_configuration(self, field_config: Dict[Any, complex]):
        """
        Set scalar field values on the spin network.
        
        :param field_config: Dict mapping vertex → field value φ
        """
        self.field_values = field_config.copy()
        self._compute_gradients()
        
    def _compute_gradients(self):
        """Compute field gradients on edges from vertex values."""
        self.field_gradients.clear()
        
        # For each edge, compute finite difference gradient
        for spin_network in []:  # Will be passed in apply()
            for edge in spin_network.edges:
                if isinstance(edge, tuple) and len(edge) == 2:
                    v1, v2 = edge
                    phi1 = self.field_values.get(v1, 0.0)
                    phi2 = self.field_values.get(v2, 0.0)
                    
                    # Finite difference gradient
                    length = spin_network.edge_length(edge)
                    gradient = (phi2 - phi1) / length if length > 0 else 0.0
                    self.field_gradients[edge] = gradient
        
    def apply(self, spin_network, amplitudes: Optional[Dict] = None) -> Dict[Any, complex]:
        """
        Compute stress tensor expectation values at all vertices.
        
        :param spin_network: SpinNetwork object
        :param amplitudes: Coherent state amplitudes (optional)
        :return: Dict mapping vertex → T_00 expectation value
        """
        # Update gradients for this network
        self._compute_gradients_for_network(spin_network)
        
        stress_tensor = {}
        
        for vertex in spin_network.nodes:
            # Get field value at vertex
            phi = self.field_values.get(vertex, 0.0)
            
            # Compute kinetic and potential contributions
            kinetic_density = self._kinetic_density(vertex, spin_network)
            potential_density = 0.5 * self.mass**2 * abs(phi)**2
            
            # T_00 = kinetic + potential (energy density)
            T_00 = kinetic_density + potential_density
            
            # Apply coherent state weighting if provided
            if amplitudes:
                # Weight by nearby edge amplitudes
                weight = self._coherent_weight(vertex, spin_network, amplitudes)
                T_00 *= weight
                
            stress_tensor[vertex] = self.coupling * T_00
            
        return stress_tensor
        
    def _compute_gradients_for_network(self, spin_network):
        """Compute gradients for specific network."""
        self.field_gradients.clear()
        
        for edge in spin_network.edges:
            if isinstance(edge, tuple) and len(edge) == 2:
                v1, v2 = edge
                phi1 = self.field_values.get(v1, 0.0)
                phi2 = self.field_values.get(v2, 0.0)
                
                length = spin_network.edge_length(edge)
                gradient = (phi2 - phi1) / length if length > 0 else 0.0
                self.field_gradients[edge] = gradient
                
    def _kinetic_density(self, vertex, spin_network) -> float:
        """
        Compute kinetic energy density at a vertex.
        
        Kinetic density = ½ Σ (∂φ/∂x_i)²
        """
        kinetic = 0.0
        
        # Sum over edges incident to vertex
        for neighbor in spin_network.graph.neighbors(vertex):
            edge = (vertex, neighbor) if vertex < neighbor else (neighbor, vertex)
            
            if edge in self.field_gradients:
                grad = self.field_gradients[edge]
                kinetic += 0.5 * abs(grad)**2
                
        return kinetic
        
    def _coherent_weight(self, vertex, spin_network, amplitudes) -> complex:
        """
        Compute coherent state weighting factor at vertex.
        
        :param vertex: Vertex location
        :param spin_network: SpinNetwork
        :param amplitudes: Edge amplitudes  
        :return: Complex weight factor
        """
        weight = 1.0 + 0j
        count = 0
        
        # Average amplitudes on incident edges
        for neighbor in spin_network.graph.neighbors(vertex):
            edge = (vertex, neighbor) if vertex < neighbor else (neighbor, vertex)
            
            if edge in amplitudes:
                weight += amplitudes[edge]
                count += 1
                
        return weight / count if count > 0 else 1.0
        
    def component(self, mu: int, nu: int):
        """
        Get T_μν component.
        
        :param mu, nu: Spacetime indices
        :return: Component calculator function
        """
        def compute_component(spin_network, vertex):
            phi = self.field_values.get(vertex, 0.0)
            
            if mu == 0 and nu == 0:
                # T_00: energy density
                kinetic = self._kinetic_density(vertex, spin_network)
                potential = 0.5 * self.mass**2 * abs(phi)**2
                return kinetic + potential
                
            elif mu == nu and mu > 0:
                # T_ii: pressure (negative energy density for perfect fluid)
                return -self._kinetic_density(vertex, spin_network)
                
            else:
                # T_μν for μ≠ν: typically zero for scalar field
                return 0.0
                
        return compute_component


class LocalT00(StressTensorOperator):
    """
    Simple local T_00 operator built from flux and extrinsic curvature.
    
    This is a phenomenological model where T_00 ~ flux² - K²
    representing the difference between electric and magnetic-like
    contributions to the energy density.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 1.0):
        """
        Initialize local T_00 operator.
        
        :param alpha: Coefficient for flux² term
        :param beta: Coefficient for curvature² term  
        """
        self.alpha = alpha
        self.beta = beta

    def apply(self, spin_network, amplitudes: Optional[Dict] = None) -> Dict[Any, complex]:
        """
        Compute T_00 expectation values at all vertices.
        
        :param spin_network: SpinNetwork object
        :param amplitudes: Optional coherent state amplitudes  
        :return: Dict mapping vertex → T_00 expectation value
        """
        T00_values = {}
        
        for vertex in spin_network.nodes:
            # Compute flux magnitude squared
            flux = spin_network.compute_flux(vertex)
            flux_squared = np.dot(flux, flux)
            
            # Compute extrinsic curvature squared
            K = spin_network.compute_extrinsic_curvature(vertex)
            K_squared = K**2
            
            # T_00 = α|flux|² - β|K|²
            T00 = self.alpha * flux_squared - self.beta * K_squared
            
            # Apply coherent state weighting
            if amplitudes:
                weight = self._amplitude_weight(vertex, spin_network, amplitudes)
                T00 *= weight
                
            T00_values[vertex] = T00
            
        return T00_values
        
    def _amplitude_weight(self, vertex, spin_network, amplitudes) -> complex:
        """Compute amplitude-based weight factor."""
        total_weight = 0.0 + 0j
        count = 0
        
        for neighbor in spin_network.graph.neighbors(vertex):
            edge = (vertex, neighbor) if vertex < neighbor else (neighbor, vertex)
            if edge in amplitudes:
                total_weight += amplitudes[edge]
                count += 1
                
        return total_weight / count if count > 0 else 1.0
        
    def component(self, mu: int, nu: int):
        """Get T_μν component (only T_00 implemented)."""
        if mu == 0 and nu == 0:
            return lambda spin_network, vertex: self.apply(spin_network)[vertex]
        else:
            return lambda spin_network, vertex: 0.0


class QuantumCorrectedStressTensor(StressTensorOperator):
    """
    Stress tensor with Loop Quantum Gravity corrections.
    
    Includes polymer quantization corrections and discrete geometry effects.
    """
    
    def __init__(self, base_operator: StressTensorOperator, 
                 polymer_scale: float = 1.0, hbar: float = 1.0):
        """
        Initialize quantum-corrected stress tensor.
        
        :param base_operator: Base classical stress tensor
        :param polymer_scale: LQG polymer correction scale μ
        :param hbar: Reduced Planck constant
        """
        self.base_operator = base_operator
        self.mu = polymer_scale
        self.hbar = hbar
        
    def apply(self, spin_network, amplitudes: Optional[Dict] = None) -> Dict[Any, complex]:
        """
        Apply quantum-corrected stress tensor.
        
        :param spin_network: SpinNetwork object
        :param amplitudes: Optional coherent state amplitudes
        :return: Dict mapping vertex → corrected T_ab values
        """
        # Get classical values
        classical_values = self.base_operator.apply(spin_network, amplitudes)
        
        # Apply polymer corrections
        corrected_values = {}
        for vertex, T_classical in classical_values.items():
            # Polymer correction: sin(μx)/(μ)
            if abs(self.mu * T_classical) < 1e-10:
                correction = T_classical
            else:
                correction = np.sin(self.mu * T_classical) / self.mu
                
            # Add discrete geometry correction
            volume_correction = self._volume_correction(vertex, spin_network)
            
            corrected_values[vertex] = correction * volume_correction
            
        return corrected_values
        
    def _volume_correction(self, vertex, spin_network) -> float:
        """
        Compute volume-based correction factor from discrete geometry.
        
        :param vertex: Vertex location
        :param spin_network: SpinNetwork
        :return: Correction factor
        """
        # Get vertex volume from incident edge lengths
        incident_edges = [(vertex, n) if vertex < n else (n, vertex) 
                         for n in spin_network.graph.neighbors(vertex)]
        
        total_length = sum(spin_network.edge_length(edge) for edge in incident_edges)
        
        if total_length > 0:
            # Volume correction ~ 1/√V in discrete geometry
            return 1.0 / np.sqrt(total_length)
        else:
            return 1.0
            
    def component(self, mu: int, nu: int):
        """Get quantum-corrected T_μν component."""
        base_component = self.base_operator.component(mu, nu)
        
        def corrected_component(spin_network, vertex):
            classical_value = base_component(spin_network, vertex)
            
            # Apply polymer correction
            if abs(self.mu * classical_value) < 1e-10:
                return classical_value
            else:
                return np.sin(self.mu * classical_value) / self.mu
                
        return corrected_component
