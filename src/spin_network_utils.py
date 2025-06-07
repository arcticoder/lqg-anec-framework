# src/spin_network_utils.py

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
from abc import ABC, abstractmethod

class SpinNetwork:
    """
    Spin network graph representation with LQG geometric data.
    
    A spin network is a graph with:
    - Edges labeled by SU(2) representations (spins)
    - Vertices labeled by SU(2) intertwiners
    - Geometric data derived from holonomies and fluxes
    """
    
    def __init__(self, graph: Optional[nx.Graph] = None):
        """
        Initialize spin network.
        
        :param graph: NetworkX graph, creates empty if None
        """
        self.graph = graph if graph is not None else nx.Graph()
        self.edge_spins = {}      # edge → j (half-integer)
        self.vertex_intertwiners = {}  # vertex → intertwiner data
        self.edge_amplitudes = {}      # edge → complex amplitude
        self.vertex_metric_data = {}   # vertex → metric tensor components
        self.holonomies = {}          # edge → SU(2) matrix
        self.fluxes = {}             # vertex → flux vector
        
    @property
    def nodes(self):
        """Get all nodes in the network."""
        return self.graph.nodes()
        
    @property 
    def edges(self):
        """Get all edges in the network."""
        return self.graph.edges()
        
    def add_edge(self, u, v, spin: float = 0.5, **kwargs):
        """
        Add edge with spin label.
        
        :param u, v: vertex labels
        :param spin: SU(2) representation (half-integer)
        """
        self.graph.add_edge(u, v, **kwargs)
        edge = (u, v) if u < v else (v, u)
        self.edge_spins[edge] = spin
        
    def edge_length(self, edge) -> float:
        """
        Compute geometric length of an edge.
        
        In LQG, edge length comes from the eigenvalue of the
        length operator applied to the spin state.
        
        :param edge: Edge tuple (u, v)
        :return: Length in Planck units
        """
        if isinstance(edge, tuple):
            edge_key = edge if edge[0] < edge[1] else (edge[1], edge[0])
        else:
            edge_key = edge
            
        if edge_key in self.edge_spins:
            j = self.edge_spins[edge_key]
            # Length eigenvalue: √(j(j+1)) in Planck units
            return np.sqrt(j * (j + 1))
        else:
            # Default unit length
            return 1.0
    
    def edge_area(self, edge) -> float:
        """
        Compute area associated with an edge (dual face).
        
        :param edge: Edge tuple
        :return: Area in Planck units²
        """
        length = self.edge_length(edge)
        return np.pi * length  # Simple model: area ∝ length
        
    def assign_amplitudes(self, amp_dict: Dict[Any, complex]):
        """
        Assign complex amplitudes to edges (coherent state data).
        
        :param amp_dict: Dictionary mapping edge → complex amplitude
        """
        self.edge_amplitudes.update(amp_dict)
        
    def set_metric_data(self, vertex, metric_data: Dict[str, float]):
        """
        Set metric tensor components at a vertex.
        
        :param vertex: Vertex label
        :param metric_data: Dict with 'g00', 'g11', etc.
        """
        self.vertex_metric_data[vertex] = metric_data.copy()
        
    def compute_flux(self, vertex) -> np.ndarray:
        """
        Compute SU(2) flux vector at a vertex.
        
        The flux is the sum of edge contributions around the vertex,
        weighted by their spin labels and orientations.
        
        :param vertex: Vertex label
        :return: 3D flux vector (su(2) algebra)
        """
        if vertex in self.fluxes:
            return self.fluxes[vertex]
            
        flux = np.zeros(3)  # su(2) ~ R³
        
        # Sum over edges incident to vertex
        for neighbor in self.graph.neighbors(vertex):
            edge = (vertex, neighbor) if vertex < neighbor else (neighbor, vertex)
            
            if edge in self.edge_spins:
                j = self.edge_spins[edge]
                # Flux contribution: spin × orientation
                orientation = 1 if vertex < neighbor else -1
                flux += orientation * j * np.array([1, 0, 0])  # Simplified: along x-axis
                
        self.fluxes[vertex] = flux
        return flux
        
    def compute_extrinsic_curvature(self, vertex) -> float:
        """
        Compute extrinsic curvature at a vertex.
        
        This comes from holonomy around small loops containing the vertex.
        
        :param vertex: Vertex label  
        :return: Scalar curvature (simplified model)
        """
        # Find triangular loops containing vertex
        neighbors = list(self.graph.neighbors(vertex))
        if len(neighbors) < 2:
            return 0.0
            
        curvature = 0.0
        for i, n1 in enumerate(neighbors):
            for n2 in neighbors[i+1:]:
                if self.graph.has_edge(n1, n2):
                    # Found triangle: vertex - n1 - n2 - vertex
                    # Compute holonomy defect
                    angle_defect = self._triangle_angle_defect(vertex, n1, n2)
                    curvature += angle_defect
                    
        return curvature / len(neighbors) if neighbors else 0.0
        
    def _triangle_angle_defect(self, v1, v2, v3) -> float:
        """Compute angle defect in triangle (simplified)."""
        # In discrete geometry: defect = π - sum of angles
        # Simplified: assume defect proportional to spins
        edges = [(v1,v2), (v2,v3), (v3,v1)]
        total_spin = 0.0
        
        for edge in edges:
            edge_key = edge if edge[0] < edge[1] else (edge[1], edge[0])
            if edge_key in self.edge_spins:
                total_spin += self.edge_spins[edge_key]
                
        return 0.1 * total_spin  # Simplified scaling
        
    def get_edge_neighbors(self, edge) -> List:
        """
        Get edges that share a vertex with given edge.
        
        :param edge: Edge tuple (u, v)
        :return: List of neighboring edges
        """
        if isinstance(edge, tuple) and len(edge) == 2:
            u, v = edge
            neighbors = []
            
            # Edges sharing vertex u
            for w in self.graph.neighbors(u):
                if w != v:
                    neighbor_edge = (u, w) if u < w else (w, u)
                    neighbors.append(neighbor_edge)
                    
            # Edges sharing vertex v  
            for w in self.graph.neighbors(v):
                if w != u:
                    neighbor_edge = (v, w) if v < w else (w, v)
                    neighbors.append(neighbor_edge)
                    
            return neighbors
        else:
            return []
            
    def volume(self) -> float:
        """
        Compute total volume of the spin network.
        
        :return: Volume in Planck units³
        """
        total_volume = 0.0
        
        # Volume comes from vertex contributions
        for vertex in self.nodes:
            # Vertex volume ~ product of incident edge lengths
            incident_edges = [(vertex, n) if vertex < n else (n, vertex) 
                             for n in self.graph.neighbors(vertex)]
            
            vertex_volume = 1.0
            for edge in incident_edges:
                vertex_volume *= self.edge_length(edge)
                
            total_volume += vertex_volume**(1/3)  # Cube root scaling
            
        return total_volume
        
    def classical_limit_metric(self, vertex) -> np.ndarray:
        """
        Extract classical metric tensor at a vertex.
        
        :param vertex: Vertex label
        :return: 4x4 metric tensor array
        """
        if vertex in self.vertex_metric_data:
            data = self.vertex_metric_data[vertex]
            # Build 4x4 metric (simplified diagonal)
            g = np.zeros((4, 4))
            g[0,0] = data.get('g00', -1.0)
            g[1,1] = data.get('g11', 1.0) 
            g[2,2] = data.get('g22', 1.0)
            g[3,3] = data.get('g33', 1.0)
            return g
        else:
            # Default: Minkowski metric
            return np.diag([-1, 1, 1, 1])


def build_flat_graph(n_nodes: int, connectivity: str = "cubic") -> SpinNetwork:
    """
    Build a regular graph to approximate flat space.
    
    :param n_nodes: Number of vertices
    :param connectivity: Type of regular graph ("cubic", "triangular", "square")
    :return: SpinNetwork approximating flat geometry
    """
    spin_net = SpinNetwork()
    
    if connectivity == "cubic":
        # 3D cubic lattice (simplified as linear chain for small n)
        if n_nodes <= 10:
            # Linear chain
            for i in range(n_nodes - 1):
                spin_net.add_edge(i, i + 1, spin=0.5)
        else:
            # 3D cubic lattice
            side = int(np.ceil(n_nodes**(1/3)))
            for i in range(side):
                for j in range(side):
                    for k in range(side):
                        node = i * side**2 + j * side + k
                        if node >= n_nodes:
                            break
                            
                        # Add edges to neighbors
                        neighbors = [
                            (i+1, j, k), (i, j+1, k), (i, j, k+1)
                        ]
                        for ni, nj, nk in neighbors:
                            if (ni < side and nj < side and nk < side):
                                neighbor = ni * side**2 + nj * side + nk
                                if neighbor < n_nodes:
                                    spin_net.add_edge(node, neighbor, spin=0.5)
                                    
    elif connectivity == "triangular":
        # Triangular lattice (2D)
        side = int(np.ceil(np.sqrt(n_nodes)))
        for i in range(side):
            for j in range(side):
                node = i * side + j
                if node >= n_nodes:
                    break
                    
                # Triangular connections
                neighbors = [(i+1, j), (i, j+1), (i+1, j+1)]
                for ni, nj in neighbors:
                    if ni < side and nj < side:
                        neighbor = ni * side + nj
                        if neighbor < n_nodes:
                            spin_net.add_edge(node, neighbor, spin=0.5)
                            
    elif connectivity == "square":
        # Square lattice (2D)
        side = int(np.ceil(np.sqrt(n_nodes)))
        for i in range(side):
            for j in range(side):
                node = i * side + j
                if node >= n_nodes:
                    break
                    
                # Square connections
                neighbors = [(i+1, j), (i, j+1)]
                for ni, nj in neighbors:
                    if ni < side and nj < side:
                        neighbor = ni * side + nj  
                        if neighbor < n_nodes:
                            spin_net.add_edge(node, neighbor, spin=0.5)
                            
    # Set all vertices to flat metric
    flat_metric = {
        'g00': -1.0, 'g11': 1.0, 'g22': 1.0, 'g33': 1.0, 'det_g': -1.0
    }
    for vertex in spin_net.nodes:
        spin_net.set_metric_data(vertex, flat_metric)
        
    return spin_net


def random_spin_network(n_nodes: int, edge_probability: float = 0.3, 
                       spin_range: Tuple[float, float] = (0.5, 2.0)) -> SpinNetwork:
    """
    Generate a random spin network.
    
    :param n_nodes: Number of vertices
    :param edge_probability: Probability of edge between any two vertices
    :param spin_range: Range of spin values (min, max)
    :return: Random SpinNetwork
    """
    # Create random graph
    graph = nx.erdos_renyi_graph(n_nodes, edge_probability)
    spin_net = SpinNetwork(graph)
    
    # Assign random spins
    min_spin, max_spin = spin_range
    for edge in spin_net.edges:
        spin = np.random.uniform(min_spin, max_spin)
        spin = round(spin * 2) / 2  # Quantize to half-integers
        spin_net.edge_spins[edge] = spin
        
    return spin_net


def spin_network_to_networkx(spin_net: SpinNetwork) -> nx.Graph:
    """
    Convert SpinNetwork to NetworkX graph for visualization.
    
    :param spin_net: SpinNetwork object
    :return: NetworkX graph with attributes
    """
    G = spin_net.graph.copy()
    
    # Add edge attributes
    for edge in G.edges():
        edge_key = edge if edge[0] < edge[1] else (edge[1], edge[0])
        if edge_key in spin_net.edge_spins:
            G.edges[edge]['spin'] = spin_net.edge_spins[edge_key]
            G.edges[edge]['length'] = spin_net.edge_length(edge_key)
            
        if edge_key in spin_net.edge_amplitudes:
            amplitude = spin_net.edge_amplitudes[edge_key]
            G.edges[edge]['amplitude'] = abs(amplitude)
            G.edges[edge]['phase'] = np.angle(amplitude)
            
    # Add node attributes  
    for node in G.nodes():
        if node in spin_net.vertex_metric_data:
            G.nodes[node].update(spin_net.vertex_metric_data[node])
            
        flux = spin_net.compute_flux(node)
        G.nodes[node]['flux_magnitude'] = np.linalg.norm(flux)
        
    return G
