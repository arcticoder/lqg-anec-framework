# src/midisuperspace_model.py

import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod

class MidiSuperspaceModel:
    """
    Hamiltonian + semiclassical dynamics for a midisuperspace truncation.
    
    Midisuperspace models preserve more degrees of freedom than minisuperspace
    (which keeps only homogeneous modes) but fewer than the full theory.
    Typical examples include spherically symmetric or cylindrically symmetric
    spacetimes with matter fields.
    """

    def __init__(self, parameters: Dict[str, float]):
        """
        Initialize midisuperspace model.
        
        :param parameters: Model parameters including:
            - 'mu': LQG polymer parameter
            - 'gamma': Barbero-Immirzi parameter
            - 'lambda': Cosmological constant
            - 'G': Newton's constant
            - 'hbar': Reduced Planck constant
        """
        self.params = parameters
        self.mu = parameters.get('mu', 0.1)
        self.gamma = parameters.get('gamma', 0.2375)  # Standard value
        self.Lambda = parameters.get('lambda', 0.0)
        self.G = parameters.get('G', 1.0)  # Planck units
        self.hbar = parameters.get('hbar', 1.0)
        
        # Phase space variables (will be set by specific models)
        self.coordinates = {}  # Coordinate degrees of freedom
        self.momenta = {}      # Conjugate momenta
        self.constraints = []  # Constraint functions
        
    def effective_hamiltonian(self, state: Dict[str, Any]) -> float:
        """
        Compute effective Hamiltonian for given phase space state.
        
        :param state: Dict containing coordinates and momenta
        :return: Hamiltonian value H_eff
        """
        # Extract coordinates and momenta
        q = state.get('coordinates', {})
        p = state.get('momenta', {})
        
        # Kinetic term with polymer corrections
        H_kinetic = self._kinetic_energy(q, p)
        
        # Potential term (includes curvature)
        H_potential = self._potential_energy(q)
        
        # Matter contribution
        H_matter = self._matter_hamiltonian(q, p)
        
        return H_kinetic + H_potential + H_matter
        
    def _kinetic_energy(self, coordinates: Dict, momenta: Dict) -> float:
        """
        Compute kinetic energy with LQG polymer corrections.
        
        In LQG, the basic variables are SU(2) connections A and densitized
        triads E. The kinetic term involves {A,E} Poisson brackets with
        polymer corrections.
        """
        kinetic = 0.0
        
        for coord_name in coordinates:
            if coord_name in momenta:
                q = coordinates[coord_name]
                p = momenta[coord_name]
                
                # Classical kinetic term: p²/(2m_eff)
                classical_term = p**2 / (2.0 * self._effective_mass(coord_name))
                
                # Apply polymer correction: sin²(μp)/(μ²)
                if abs(self.mu * p) < 1e-10:
                    polymer_term = classical_term
                else:
                    polymer_factor = (np.sin(self.mu * p) / (self.mu * p))**2
                    polymer_term = classical_term * polymer_factor
                    
                kinetic += polymer_term
                
        return kinetic
        
    def _potential_energy(self, coordinates: Dict) -> float:
        """
        Compute potential energy from curvature and cosmological constant.
        
        V ~ √det(q) [R(q) + 2Λ] where q is the spatial metric
        """
        potential = 0.0
        
        # Extract metric components (model-dependent)
        if 'scale_factor' in coordinates:
            # FRW-like model
            a = coordinates['scale_factor']
            if a > 0:
                # Volume factor
                volume = a**3
                
                # Curvature contribution (simplified)
                curvature = -6 / a**2  # k=0 FRW curvature
                
                # Total potential
                potential = volume * (curvature + 2 * self.Lambda)
                
        elif 'radial_metric' in coordinates:
            # Spherically symmetric model
            grr = coordinates['radial_metric']
            if grr > 0:
                volume = 4 * np.pi * grr**(3/2)
                curvature = -2 / grr  # Simplified
                potential = volume * (curvature + 2 * self.Lambda)
                
        return potential
        
    def _matter_hamiltonian(self, coordinates: Dict, momenta: Dict) -> float:
        """
        Matter field contribution to Hamiltonian.
        
        For scalar field: H_matter = (1/2)[π²/√q + √q (∇φ)² + √q m²φ²]
        """
        H_matter = 0.0
        
        if 'scalar_field' in coordinates and 'scalar_momentum' in momenta:
            phi = coordinates['scalar_field']
            pi_phi = momenta['scalar_momentum']
            
            # Volume element (model-dependent)
            volume = self._volume_element(coordinates)
            
            # Kinetic term: π²/(2√q)
            kinetic = pi_phi**2 / (2 * np.sqrt(volume)) if volume > 0 else 0
            
            # Gradient term (simplified - no spatial gradients in midisuperspace)
            gradient = 0.0
            
            # Mass term: (1/2)m²φ²√q
            mass = self.params.get('scalar_mass', 0.0)
            potential = 0.5 * mass**2 * phi**2 * np.sqrt(volume)
            
            H_matter = kinetic + gradient + potential
            
        return H_matter
        
    def _effective_mass(self, coordinate_name: str) -> float:
        """Get effective mass for kinetic term."""
        if coordinate_name == 'scale_factor':
            return 1.0 / (12 * np.pi * self.G)
        elif coordinate_name == 'radial_metric':
            return 1.0 / (4 * np.pi * self.G)
        else:
            return 1.0
            
    def _volume_element(self, coordinates: Dict) -> float:
        """Compute volume element √det(q)."""
        if 'scale_factor' in coordinates:
            a = coordinates['scale_factor']
            return a**3
        elif 'radial_metric' in coordinates:
            grr = coordinates['radial_metric']
            return 4 * np.pi * grr**(3/2)
        else:
            return 1.0

    def expectation_T00(self, coherent_state) -> np.ndarray:
        """
        Compute 〈T₀₀〉 from a CoherentState on this midisuperspace.
        
        :param coherent_state: CoherentState object
        :return: Array of T₀₀ values at network vertices
        """
        # Project coherent state onto midisuperspace variables
        midispace_state = self._project_to_midispace(coherent_state)
        
        # Compute stress tensor from matter Hamiltonian
        T00_values = []
        
        for vertex in coherent_state.graph.nodes:
            # Get local field values
            local_state = self._local_state_at_vertex(vertex, midispace_state)
            
            # Compute energy density T₀₀
            if 'scalar_field' in local_state and 'scalar_momentum' in local_state:
                phi = local_state['scalar_field']
                pi_phi = local_state['scalar_momentum']
                volume = self._volume_at_vertex(vertex, coherent_state.graph)
                
                # T₀₀ = (1/2)[π²/q + q(∇φ)² + qm²φ²]
                kinetic_density = pi_phi**2 / (2 * volume) if volume > 0 else 0
                gradient_density = 0  # No gradients in midisuperspace
                mass = self.params.get('scalar_mass', 0.0)
                potential_density = 0.5 * mass**2 * phi**2 * volume
                
                T00 = kinetic_density + gradient_density + potential_density
            else:
                T00 = 0.0
                
            T00_values.append(T00)
            
        return np.array(T00_values)
        
    def _project_to_midispace(self, coherent_state) -> Dict[str, Any]:
        """
        Project full coherent state to midisuperspace variables.
        
        This involves averaging over the spin network to extract
        the symmetric modes preserved in the midisuperspace truncation.
        """
        projected_state = {
            'coordinates': {},
            'momenta': {}
        }
        
        # Extract scale factor from network volume
        total_volume = coherent_state.graph.volume()
        projected_state['coordinates']['scale_factor'] = total_volume**(1/3)
        
        # Extract scalar field from vertex average
        if hasattr(coherent_state, 'field_values'):
            avg_field = np.mean(list(coherent_state.field_values.values()))
            projected_state['coordinates']['scalar_field'] = avg_field
            
        # Momenta (would need to be computed from time derivatives)
        projected_state['momenta']['scale_momentum'] = 0.0  # Placeholder
        projected_state['momenta']['scalar_momentum'] = 0.0  # Placeholder
        
        return projected_state
        
    def _local_state_at_vertex(self, vertex, midispace_state) -> Dict[str, float]:
        """Get local field values at a specific vertex."""
        # In midisuperspace, fields are homogeneous
        return midispace_state['coordinates'].copy()
        
    def _volume_at_vertex(self, vertex, spin_network) -> float:
        """Get volume element at specific vertex."""
        # Sum volumes of incident edges/faces
        incident_edges = [(vertex, n) if vertex < n else (n, vertex) 
                         for n in spin_network.graph.neighbors(vertex)]
        
        total_length = sum(spin_network.edge_length(edge) for edge in incident_edges)
        return total_length if total_length > 0 else 1.0
        
    def hamilton_equations(self, state: Dict[str, Any], 
                          time: float) -> Dict[str, Any]:
        """
        Compute Hamilton's equations: dq/dt = ∂H/∂p, dp/dt = -∂H/∂q
        
        :param state: Current phase space state
        :param time: Current time
        :return: Time derivatives {dq/dt, dp/dt}
        """
        derivatives = {
            'coordinates': {},
            'momenta': {}
        }
        
        # Small parameter for numerical derivatives
        epsilon = 1e-8
        
        q = state.get('coordinates', {})
        p = state.get('momenta', {})
        
        # Compute ∂H/∂p (gives dq/dt)
        for coord_name in q:
            if coord_name in p:
                # Perturb momentum
                p_plus = p.copy()
                p_plus[coord_name] += epsilon
                state_plus = {'coordinates': q, 'momenta': p_plus}
                
                p_minus = p.copy()
                p_minus[coord_name] -= epsilon
                state_minus = {'coordinates': q, 'momenta': p_minus}
                
                # Numerical derivative
                H_plus = self.effective_hamiltonian(state_plus)
                H_minus = self.effective_hamiltonian(state_minus)
                dH_dp = (H_plus - H_minus) / (2 * epsilon)
                
                derivatives['coordinates'][coord_name] = dH_dp
                
        # Compute -∂H/∂q (gives dp/dt)
        for coord_name in q:
            # Perturb coordinate
            q_plus = q.copy()
            q_plus[coord_name] += epsilon
            state_plus = {'coordinates': q_plus, 'momenta': p}
            
            q_minus = q.copy()
            q_minus[coord_name] -= epsilon
            state_minus = {'coordinates': q_minus, 'momenta': p}
            
            # Numerical derivative
            H_plus = self.effective_hamiltonian(state_plus)
            H_minus = self.effective_hamiltonian(state_minus)
            dH_dq = (H_plus - H_minus) / (2 * epsilon)
            
            derivatives['momenta'][coord_name] = -dH_dq
            
        return derivatives
        
    def solve_evolution(self, initial_state: Dict[str, Any], 
                       time_span: Tuple[float, float], 
                       num_steps: int = 1000) -> Tuple[np.ndarray, List[Dict]]:
        """
        Solve Hamilton's equations numerically.
        
        :param initial_state: Initial phase space state
        :param time_span: (t_initial, t_final) 
        :param num_steps: Number of time steps
        :return: (time_array, list_of_states)
        """
        t_initial, t_final = time_span
        dt = (t_final - t_initial) / num_steps
        
        times = np.linspace(t_initial, t_final, num_steps + 1)
        states = [initial_state.copy()]
        
        current_state = initial_state.copy()
        
        # Simple Euler integration (could use RK4 for better accuracy)
        for i in range(num_steps):
            t = times[i]
            derivatives = self.hamilton_equations(current_state, t)
            
            # Update coordinates and momenta
            new_state = {'coordinates': {}, 'momenta': {}}
            
            for coord_name in current_state['coordinates']:
                current_q = current_state['coordinates'][coord_name]
                dq_dt = derivatives['coordinates'].get(coord_name, 0.0)
                new_state['coordinates'][coord_name] = current_q + dt * dq_dt
                
            for momentum_name in current_state['momenta']:
                current_p = current_state['momenta'][momentum_name]
                dp_dt = derivatives['momenta'].get(momentum_name, 0.0)
                new_state['momenta'][momentum_name] = current_p + dt * dp_dt
                
            current_state = new_state
            states.append(current_state.copy())
            
        return times, states
