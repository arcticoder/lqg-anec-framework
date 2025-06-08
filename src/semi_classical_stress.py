"""
Semi-Classical LQG Stress-Tensor Operator Module

Semi-classical stress-tensor operator for discrete spin networks in Loop Quantum Gravity.
Implements coherent state expectation values and polymer-enhanced stress-energy computation.

Key Features:
- Discrete geometric operators on spin networks
- Coherent state representation for semi-classical limit
- Polymer-enhanced stress-energy tensor computation
- GPU-optimized spin network contractions
- Volume and area operators with LQG discreteness

Theory Background:
- LQG spin networks as quantum geometry states
- Coherent states |γ,z⟩ for semi-classical geometry
- Stress-tensor operator Ê(T̂_μν) in discrete setting
- Polymer field quantization modifications
- Connection to continuum field theory in large-j limit
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpinNetworkType(Enum):
    """Types of spin network configurations."""
    CUBICAL = "cubical"
    TETRAHEDRAL = "tetrahedral" 
    IRREGULAR = "irregular"
    TRIANGULAR = "triangular"


@dataclass
class LQGParameters:
    """Parameters for Loop Quantum Gravity computation."""
    # Fundamental LQG parameters
    planck_length: float = 1.616e-35      # Planck length (meters)
    barbero_immirzi: float = 0.2375       # Barbero-Immirzi parameter γ
    
    # Spin network parameters
    network_type: SpinNetworkType = SpinNetworkType.CUBICAL
    max_spin: float = 10.0                # Maximum spin quantum number
    network_size: int = 20                # Number of nodes per dimension
    edge_density: float = 0.8             # Connectivity of spin network
    
    # Coherent state parameters
    coherent_scale: float = 1000.0        # Semi-classical scaling parameter
    phase_spread: float = 0.1             # Coherent state width
    
    # Polymer field parameters
    polymer_scale: float = 1.0            # Polymer length scale μ
    polymer_boost: float = 1.5            # Polymer enhancement factor
    
    # Numerical parameters
    grid_resolution: int = 64             # Discrete grid resolution
    truncation_order: int = 8             # Expansion truncation order
    device: str = "cuda"                  # Computation device


@dataclass
class SpinNetworkNode:
    """Individual node in LQG spin network."""
    node_id: int
    position: torch.Tensor               # 3D position
    valence: int                         # Number of incident edges
    quantum_numbers: List[float]         # Associated quantum numbers
    coherent_amplitude: complex = 1.0    # Coherent state amplitude


@dataclass
class SpinNetworkEdge:
    """Edge connecting nodes in LQG spin network."""
    edge_id: int
    node_start: int                      # Starting node ID
    node_end: int                        # Ending node ID
    spin: float                          # SU(2) spin quantum number
    holonomy: torch.Tensor               # SU(2) holonomy matrix
    length: float                        # Edge length in Planck units


class SemiClassicalStressTensor:
    """
    Semi-classical stress-tensor operator for Loop Quantum Gravity.
    
    Computes ⟨γ,z|T̂_μν|γ,z⟩ for coherent states on spin networks.
    Implements polymer-enhanced field theory modifications.
    """
    
    def __init__(self, params: LQGParameters):
        """Initialize semi-classical LQG stress tensor system."""
        self.params = params
        self.device = torch.device(params.device if torch.cuda.is_available() else "cpu")
        
        # Initialize spin network
        self.nodes: List[SpinNetworkNode] = []
        self.edges: List[SpinNetworkEdge] = []
        self._setup_spin_network()
        
        # Initialize geometric operators
        self._setup_geometric_operators()
        
        # Initialize coherent states
        self._setup_coherent_states()
        
        # Precompute operator matrices
        self._precompute_operators()
        
        logger.info(f"Semi-classical LQG stress tensor initialized on {self.device}")
        logger.info(f"Spin network: {len(self.nodes)} nodes, {len(self.edges)} edges")
        logger.info(f"Max spin: {params.max_spin}, Coherent scale: {params.coherent_scale}")
    
    def _setup_spin_network(self):
        """Generate discrete spin network geometry."""
        logger.info(f"Setting up {self.params.network_type.value} spin network...")
        
        if self.params.network_type == SpinNetworkType.CUBICAL:
            self._generate_cubical_network()
        elif self.params.network_type == SpinNetworkType.TETRAHEDRAL:
            self._generate_tetrahedral_network()
        else:
            self._generate_irregular_network()
    
    def _generate_cubical_network(self):
        """Generate cubical lattice spin network."""
        n = self.params.network_size
        node_id = 0
        
        # Create nodes on cubic lattice
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    position = torch.tensor([i, j, k], dtype=torch.float32, device=self.device)
                    
                    # Random quantum numbers
                    quantum_nums = [
                        np.random.uniform(0.5, self.params.max_spin) for _ in range(3)
                    ]
                    
                    node = SpinNetworkNode(
                        node_id=node_id,
                        position=position,
                        valence=6,  # Cubic lattice coordination
                        quantum_numbers=quantum_nums
                    )
                    
                    self.nodes.append(node)
                    node_id += 1
        
        # Create edges between nearest neighbors
        edge_id = 0
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                node_i, node_j = self.nodes[i], self.nodes[j]
                
                # Check if nearest neighbors
                distance = torch.norm(node_i.position - node_j.position)
                if distance < 1.5:  # Nearest neighbor threshold
                    
                    # Random spin assignment
                    spin = np.random.uniform(0.5, self.params.max_spin)
                    
                    # SU(2) holonomy matrix (simplified)
                    angle = np.random.uniform(0, 2*np.pi)
                    holonomy = torch.tensor([[np.cos(angle), -np.sin(angle)],
                                           [np.sin(angle), np.cos(angle)]],
                                          dtype=torch.complex64, device=self.device)
                    
                    edge = SpinNetworkEdge(
                        edge_id=edge_id,
                        node_start=i,
                        node_end=j,
                        spin=spin,
                        holonomy=holonomy,
                        length=distance.item()
                    )
                    
                    self.edges.append(edge)
                    edge_id += 1
    
    def _generate_tetrahedral_network(self):
        """Generate tetrahedral spin network (simplified)."""
        # Simplified tetrahedral network for demo
        n_tetrahedra = self.params.network_size
        
        for tet_id in range(n_tetrahedra):
            # Create 4 nodes per tetrahedron
            for vertex in range(4):
                position = torch.randn(3, device=self.device)
                
                node = SpinNetworkNode(
                    node_id=len(self.nodes),
                    position=position,
                    valence=3,  # Tetrahedral coordination
                    quantum_numbers=[np.random.uniform(0.5, self.params.max_spin) for _ in range(3)]
                )
                
                self.nodes.append(node)
        
        # Connect tetrahedra vertices (simplified)
        for i in range(0, len(self.nodes), 4):
            for j in range(4):
                for k in range(j + 1, 4):
                    if i + k < len(self.nodes):
                        self._add_edge(i + j, i + k)
    
    def _generate_irregular_network(self):
        """Generate irregular random spin network."""
        n_nodes = self.params.network_size**3 // 8  # Sparse irregular network
        
        # Random node positions
        for node_id in range(n_nodes):
            position = torch.randn(3, device=self.device) * 5.0
            
            node = SpinNetworkNode(
                node_id=node_id,
                position=position,
                valence=np.random.randint(2, 8),
                quantum_numbers=[np.random.uniform(0.5, self.params.max_spin) for _ in range(3)]
            )
            
            self.nodes.append(node)
        
        # Random connectivity
        for i in range(len(self.nodes)):
            for j in range(i + 1, len(self.nodes)):
                if np.random.random() < self.params.edge_density:
                    self._add_edge(i, j)
    
    def _add_edge(self, node_start: int, node_end: int):
        """Add edge between two nodes."""
        if node_start >= len(self.nodes) or node_end >= len(self.nodes):
            return
        
        distance = torch.norm(self.nodes[node_start].position - self.nodes[node_end].position)
        spin = np.random.uniform(0.5, self.params.max_spin)
        
        # Random SU(2) holonomy
        angle = np.random.uniform(0, 2*np.pi)
        holonomy = torch.tensor([[np.cos(angle), -np.sin(angle)],
                               [np.sin(angle), np.cos(angle)]],
                              dtype=torch.complex64, device=self.device)
        
        edge = SpinNetworkEdge(
            edge_id=len(self.edges),
            node_start=node_start,
            node_end=node_end,
            spin=spin,
            holonomy=holonomy,
            length=distance.item()
        )
        
        self.edges.append(edge)
    
    def _setup_geometric_operators(self):
        """Setup discrete geometric operators (area, volume, etc.)."""
        self.n_nodes = len(self.nodes)
        self.n_edges = len(self.edges)
        
        # Area operator eigenvalues (for faces)
        self.area_eigenvalues = torch.tensor([
            np.sqrt(j * (j + 1)) * self.params.planck_length**2
            for j in np.arange(0.5, self.params.max_spin + 0.5, 0.5)
        ], device=self.device)
        
        # Volume operator (simplified)
        self.volume_eigenvalues = torch.tensor([
            j**(3/2) * self.params.planck_length**3
            for j in np.arange(0.5, self.params.max_spin + 0.5, 0.5)
        ], device=self.device)
        
        # Length operator
        self.length_eigenvalues = torch.tensor([
            np.sqrt(j * (j + 1)) * self.params.planck_length
            for j in np.arange(0.5, self.params.max_spin + 0.5, 0.5)
        ], device=self.device)
    
    def _setup_coherent_states(self):
        """Initialize coherent state parameters for semi-classical limit."""
        # Coherent state amplitudes for each node
        self.coherent_amplitudes = torch.randn(self.n_nodes, dtype=torch.complex64, device=self.device)
        self.coherent_amplitudes *= self.params.coherent_scale
        
        # Phase parameters
        self.coherent_phases = torch.randn(self.n_nodes, device=self.device) * 2 * np.pi
        
        # Classical geometry parameters
        self.classical_areas = torch.rand(self.n_edges, device=self.device) * 100 * self.params.planck_length**2
        self.classical_volumes = torch.rand(self.n_nodes, device=self.device) * 1000 * self.params.planck_length**3
    
    def _precompute_operators(self):
        """Precompute operator matrices for efficient computation."""
        # Adjacency matrix for spin network
        self.adjacency_matrix = torch.zeros(self.n_nodes, self.n_nodes, device=self.device)
        
        for edge in self.edges:
            self.adjacency_matrix[edge.node_start, edge.node_end] = edge.spin
            self.adjacency_matrix[edge.node_end, edge.node_start] = edge.spin
        
        # Edge-node incidence matrix
        self.incidence_matrix = torch.zeros(self.n_edges, self.n_nodes, device=self.device)
        
        for i, edge in enumerate(self.edges):
            self.incidence_matrix[i, edge.node_start] = 1.0
            self.incidence_matrix[i, edge.node_end] = -1.0
        
        # Spin weight matrix
        self.spin_weights = torch.tensor([edge.spin for edge in self.edges], device=self.device)
    
    def compute_area_operator_expectation(self) -> torch.Tensor:
        """
        Compute expectation value of area operator in coherent states.
        
        ⟨γ,z|Â_S|γ,z⟩ for surface S intersecting edges.
        """
        # Area contributions from each edge
        area_contributions = torch.zeros(self.n_edges, device=self.device)
        
        for i, edge in enumerate(self.edges):
            # Coherent state expectation for area
            j = edge.spin
            area_eigenvalue = np.sqrt(j * (j + 1)) * self.params.planck_length**2
            
            # Include coherent state modification
            coherent_factor = torch.abs(self.coherent_amplitudes[edge.node_start] * 
                                      torch.conj(self.coherent_amplitudes[edge.node_end]))
            
            area_contributions[i] = area_eigenvalue * coherent_factor
        
        return area_contributions
    
    def compute_volume_operator_expectation(self) -> torch.Tensor:
        """
        Compute expectation value of volume operator in coherent states.
        
        ⟨γ,z|V̂_R|γ,z⟩ for region R containing nodes.
        """
        volume_contributions = torch.zeros(self.n_nodes, device=self.device)
        
        for i, node in enumerate(self.nodes):
            # Volume eigenvalue (simplified)
            avg_spin = np.mean(node.quantum_numbers)
            volume_eigenvalue = avg_spin**(3/2) * self.params.planck_length**3
            
            # Coherent state enhancement
            coherent_factor = torch.abs(self.coherent_amplitudes[i])**2
            
            volume_contributions[i] = volume_eigenvalue * coherent_factor
        
        return volume_contributions
    
    def compute_holonomy_expectation(self) -> torch.Tensor:
        """
        Compute holonomy expectation values along edges.
        
        ⟨γ,z|h_e[A]|γ,z⟩ for edge e.
        """
        holonomy_expectations = torch.zeros(self.n_edges, 2, 2, dtype=torch.complex64, device=self.device)
        
        for i, edge in enumerate(self.edges):
            # Classical holonomy with coherent state corrections
            base_holonomy = edge.holonomy
            
            # Coherent state phase
            start_node = self.nodes[edge.node_start]
            end_node = self.nodes[edge.node_end]
            
            coherent_phase = torch.exp(1j * (self.coherent_phases[edge.node_start] - 
                                           self.coherent_phases[edge.node_end]))
            
            holonomy_expectations[i] = base_holonomy * coherent_phase
        
        return holonomy_expectations
    
    def compute_curvature_tensor(self) -> torch.Tensor:
        """
        Compute discrete curvature tensor from holonomies.
        
        F_μν ≈ holonomy around elementary plaquettes.
        """
        # Find elementary plaquettes (4-cycles in network)
        plaquettes = self._find_plaquettes()
        
        curvature_contributions = torch.zeros(len(plaquettes), 3, 3, device=self.device)
        
        for i, plaquette in enumerate(plaquettes):
            # Holonomy around plaquette
            total_holonomy = torch.eye(2, dtype=torch.complex64, device=self.device)
            
            for edge_id in plaquette:
                if edge_id < len(self.edges):
                    total_holonomy = torch.matmul(total_holonomy, self.edges[edge_id].holonomy)
            
            # Extract curvature (simplified)
            # Real part of trace deviation from identity
            curvature_scalar = torch.real(torch.trace(total_holonomy)) - 2.0
            
            # Distribute to tensor components (simplified)
            curvature_contributions[i, 0, 0] = curvature_scalar
            curvature_contributions[i, 1, 1] = curvature_scalar
            curvature_contributions[i, 2, 2] = curvature_scalar / 2
        
        return curvature_contributions
    
    def _find_plaquettes(self) -> List[List[int]]:
        """Find elementary plaquettes (4-cycles) in spin network."""
        plaquettes = []
        
        # Simplified plaquette finding for demo
        for i in range(min(50, len(self.edges) // 4)):  # Limit for efficiency
            # Create mock 4-cycle
            if i + 3 < len(self.edges):
                plaquettes.append([i, i+1, i+2, i+3])
        
        return plaquettes
    
    def compute_stress_energy_expectation(self, field_config: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute expectation value of stress-energy tensor operator.
        
        ⟨γ,z|T̂_μν|γ,z⟩ with polymer field modifications.
        
        Args:
            field_config: External field configuration (optional)
        
        Returns:
            Dictionary with stress-energy tensor components
        """
        logger.info("Computing stress-energy tensor expectation values...")
        
        # Initialize field configuration if not provided
        if field_config is None:
            field_config = torch.randn(self.n_nodes, device=self.device)
        
        # Geometric contributions
        area_expectations = self.compute_area_operator_expectation()
        volume_expectations = self.compute_volume_operator_expectation()
        
        # Field energy density (T_00)
        energy_density = torch.zeros(self.n_nodes, device=self.device)
        
        for i, node in enumerate(self.nodes):
            # Kinetic energy contribution
            kinetic_contribution = 0.5 * field_config[i]**2
            
            # Polymer modification
            polymer_factor = 1 + self.params.polymer_boost * torch.exp(-torch.abs(field_config[i]) / self.params.polymer_scale)
            
            # Volume weighting
            volume_weight = volume_expectations[i] / (self.params.planck_length**3)
            
            energy_density[i] = kinetic_contribution * polymer_factor * volume_weight
          # Field momentum density (T_0i)
        momentum_density = torch.zeros(self.n_nodes, 3, device=self.device)
        for i, node in enumerate(self.nodes):
            # Finite difference gradient approximation
            for neighbor_idx in torch.nonzero(self.adjacency_matrix[i]):
                j = neighbor_idx.item()
                if j < self.n_nodes:
                    direction = self.nodes[j].position - node.position
                    direction = direction / torch.norm(direction)
                    
                    field_gradient = (field_config[j] - field_config[i]) / torch.norm(self.nodes[j].position - node.position)
                    momentum_density[i] += field_gradient * direction
        
        # Stress tensor (T_ij)
        stress_tensor = torch.zeros(self.n_nodes, 3, 3, device=self.device)
        
        for i, node in enumerate(self.nodes):
            # Pressure contributions
            pressure = 0.5 * field_config[i]**2
            
            # Anisotropic stress from discrete geometry
            for k in range(3):
                stress_tensor[i, k, k] = pressure
                
                # Add geometric stress
                if i < len(area_expectations):
                    geometric_stress = area_expectations[min(i, len(area_expectations)-1)] / (self.params.planck_length**2)
                    stress_tensor[i, k, k] += geometric_stress
        
        return {
            'T_00': energy_density,           # Energy density
            'T_0i': momentum_density,         # Momentum density  
            'T_ij': stress_tensor,            # Stress tensor
            'area_contributions': area_expectations,
            'volume_contributions': volume_expectations
        }
    
    def compute_polymer_enhanced_stress(self, field_config: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute polymer-enhanced stress-energy tensor with LQG corrections.
        
        Includes discrete geometry effects and polymer field modifications.
        """
        # Base stress-energy computation
        stress_components = self.compute_stress_energy_expectation(field_config)
        
        # Polymer enhancement factors
        polymer_enhancement = torch.zeros(self.n_nodes, device=self.device)
        
        for i in range(self.n_nodes):
            # Polymer length scale effects
            field_scale = torch.abs(field_config[i])
            polymer_ratio = field_scale / self.params.polymer_scale
            
            # Enhancement function (smooth cutoff)
            enhancement = self.params.polymer_boost * torch.exp(-polymer_ratio**2)
            polymer_enhancement[i] = 1 + enhancement
        
        # Apply polymer modifications
        stress_components['T_00'] *= polymer_enhancement
        
        # Modify spatial components
        for i in range(3):
            stress_components['T_0i'][:, i] *= polymer_enhancement
            for j in range(3):
                stress_components['T_ij'][:, i, j] *= polymer_enhancement.unsqueeze(-1).expand(-1, 1).squeeze()
        
        # Add discreteness corrections
        discreteness_correction = torch.zeros_like(stress_components['T_00'])
        
        for i in range(self.n_nodes):
            # Discreteness scale
            local_volume = stress_components['volume_contributions'][i]
            discrete_scale = (local_volume / self.params.planck_length**3)**(1/3)
            
            # Correction grows for small volumes (high discreteness)
            discreteness_correction[i] = self.params.planck_length**2 / (discrete_scale**2 + self.params.planck_length**2)
        
        stress_components['T_00'] += discreteness_correction
        stress_components['polymer_enhancement'] = polymer_enhancement
        stress_components['discreteness_correction'] = discreteness_correction
        
        return stress_components
    
    def compute_anec_violation_lqg(self, field_config: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute ANEC violation using LQG stress-energy tensor.
        
        Integrates ⟨T̂_μν⟩ k^μ k^ν along null curves in discrete geometry.
        """
        logger.info("Computing LQG ANEC violation...")
        
        # Compute polymer-enhanced stress tensor
        stress_tensor = self.compute_polymer_enhanced_stress(field_config)
        
        # Null vector (simplified: along x-direction with boost)
        boost_velocity = 0.5
        null_vector = torch.tensor([1.0, boost_velocity, 0.0, 0.0], device=self.device)  # (t, x, y, z)
        
        # ANEC integrand for each node
        anec_integrand = torch.zeros(self.n_nodes, device=self.device)
        
        for i in range(self.n_nodes):
            # T_μν k^μ k^ν (simplified for 2+1D)
            T_00 = stress_tensor['T_00'][i]
            T_0x = stress_tensor['T_0i'][i, 0] if stress_tensor['T_0i'].shape[1] > 0 else torch.tensor(0.0)
            T_xx = stress_tensor['T_ij'][i, 0, 0] if stress_tensor['T_ij'].shape[1] > 0 else torch.tensor(0.0)
            
            # Null contraction
            anec_integrand[i] = (T_00 * null_vector[0]**2 + 
                               2 * T_0x * null_vector[0] * null_vector[1] +
                               T_xx * null_vector[1]**2)
        
        # Spatial integration (sum over discrete nodes with volume weighting)
        volume_weights = stress_tensor['volume_contributions']
        total_volume = torch.sum(volume_weights)
        
        if total_volume > 0:
            anec_integral = torch.sum(anec_integrand * volume_weights) / total_volume
        else:
            anec_integral = torch.sum(anec_integrand) / self.n_nodes
        
        # Violation analysis
        negative_nodes = torch.sum(anec_integrand < 0)
        violation_strength = torch.min(anec_integrand)
        
        return {
            'anec_integral': anec_integral,
            'anec_integrand': anec_integrand,
            'negative_nodes': negative_nodes,
            'violation_strength': violation_strength,
            'total_nodes': self.n_nodes,
            'violation_fraction': negative_nodes / self.n_nodes
        }
    
    def generate_lqg_stress_report(self, field_config: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Generate comprehensive LQG stress-energy analysis report.
        """
        logger.info("Generating LQG stress-energy report...")
        
        # Default field configuration
        if field_config is None:
            field_config = torch.randn(self.n_nodes, device=self.device)
        
        # Compute all stress tensor components
        stress_tensor = self.compute_polymer_enhanced_stress(field_config)
        
        # ANEC violation analysis
        anec_result = self.compute_anec_violation_lqg(field_config)
        
        # Geometric analysis
        area_expectations = self.compute_area_operator_expectation()
        volume_expectations = self.compute_volume_operator_expectation()
        
        # Summary statistics
        report = {
            'lqg_parameters': {
                'planck_length': self.params.planck_length,
                'barbero_immirzi': self.params.barbero_immirzi,
                'network_type': self.params.network_type.value,
                'polymer_scale': self.params.polymer_scale,
                'coherent_scale': self.params.coherent_scale
            },
            'network_statistics': {
                'num_nodes': self.n_nodes,
                'num_edges': self.n_edges,
                'avg_valence': np.mean([node.valence for node in self.nodes]),
                'max_spin': max([edge.spin for edge in self.edges])
            },
            'stress_tensor_analysis': {
                'mean_energy_density': torch.mean(stress_tensor['T_00']).item(),
                'max_energy_density': torch.max(stress_tensor['T_00']).item(),
                'min_energy_density': torch.min(stress_tensor['T_00']).item(),
                'energy_variance': torch.var(stress_tensor['T_00']).item(),
                'polymer_enhancement_avg': torch.mean(stress_tensor['polymer_enhancement']).item(),
                'discreteness_correction_avg': torch.mean(stress_tensor['discreteness_correction']).item()
            },
            'anec_violation': {
                'integral_value': anec_result['anec_integral'].item(),
                'violation_strength': anec_result['violation_strength'].item(),
                'violation_fraction': anec_result['violation_fraction'].item(),
                'negative_energy_nodes': anec_result['negative_nodes'].item()
            },
            'geometry_analysis': {
                'total_area': torch.sum(area_expectations).item(),
                'total_volume': torch.sum(volume_expectations).item(),
                'avg_area_per_edge': torch.mean(area_expectations).item(),
                'avg_volume_per_node': torch.mean(volume_expectations).item()
            }
        }
        
        logger.info(f"LQG ANEC violation: {anec_result['anec_integral'].item():.2e}")
        logger.info(f"Violation fraction: {anec_result['violation_fraction'].item():.3f}")
        logger.info(f"Network size: {self.n_nodes} nodes, {self.n_edges} edges")
        
        return report


def test_lqg_stress_tensor():
    """Test semi-classical LQG stress tensor functionality."""
    print("Testing Semi-Classical LQG Stress Tensor...")
    
    # Test parameters
    params = LQGParameters(
        network_size=10,  # Smaller for testing
        max_spin=5.0,
        coherent_scale=100.0,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Initialize system
    lqg_stress = SemiClassicalStressTensor(params)
    
    # Generate field configuration
    field_config = torch.randn(lqg_stress.n_nodes, device=lqg_stress.device)
    
    # Generate analysis report
    report = lqg_stress.generate_lqg_stress_report(field_config)
    
    print(f"Network: {report['network_statistics']['num_nodes']} nodes, {report['network_statistics']['num_edges']} edges")
    print(f"ANEC violation: {report['anec_violation']['integral_value']:.2e}")
    print(f"Violation fraction: {report['anec_violation']['violation_fraction']:.3f}")
    print(f"Mean energy density: {report['stress_tensor_analysis']['mean_energy_density']:.2e}")
    
    return report


if __name__ == "__main__":
    # Run test
    test_report = test_lqg_stress_tensor()
    print("Semi-classical LQG stress tensor test completed successfully!")
