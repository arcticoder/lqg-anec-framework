# src/anec_violation_analysis.py

"""
ANEC Violation Analysis Framework

Comprehensive analysis of Averaged Null Energy Condition violations
in Loop Quantum Gravity using coherent states, polymer corrections,
and effective field theory.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Callable
from scipy.integrate import quad, simpson
from scipy.optimize import minimize
import os

# Local imports
from .midisuperspace_model import MidiSuperspaceModel
from .coherent_states import CoherentState
from .spin_network_utils import build_flat_graph, SpinNetwork
from .stress_tensor_operator import LocalT00, ScalarFieldStressTensor, QuantumCorrectedStressTensor
from .polymer_quantization import polymer_quantum_inequality_bound, PolymerOperator
from .effective_action import compute_effective_terms, derive_anec_violation_from_eft

def ford_roman_sampling_function(t: float, tau: float) -> float:
    """
    Ford-Roman Gaussian sampling function.
    
    f(t) = exp(-t²/2τ²) / (√(2π) τ)
    
    :param t: Time coordinate
    :param tau: Sampling timescale
    :return: Sampling function value
    """
    return np.exp(-t**2 / (2 * tau**2)) / (np.sqrt(2 * np.pi) * tau)

def lorentzian_sampling_function(t: float, tau: float) -> float:
    """
    Alternative Lorentzian sampling function.
    
    f(t) = τ/π / (t² + τ²)
    
    :param t: Time coordinate  
    :param tau: Sampling timescale
    :return: Sampling function value
    """
    return tau / (np.pi * (t**2 + tau**2))

def compute_anec_integral(Tab_vals: List[float], 
                         geodesic: List[Tuple[float, ...]], 
                         dλ: float,
                         sampling_func: Optional[Callable] = None,
                         tau: float = 1.0) -> float:
    """
    Compute ANEC integral ∫ T_ab k^a k^b dλ along null geodesic.
    
    :param Tab_vals: List of stress tensor values T_ab at each λ
    :param geodesic: List of null tangent vectors k^a(λ)
    :param dλ: Step size along geodesic
    :param sampling_func: Optional time sampling function
    :param tau: Sampling timescale
    :return: ANEC integral value
    """
    integral = 0.0
    
    for i, (T_ab, k) in enumerate(zip(Tab_vals, geodesic)):
        # Time coordinate for sampling function
        t = (i - len(Tab_vals)/2) * dλ
        
        # Stress tensor contraction with null vector
        # For null geodesic: k^a k_a = 0, we want T_ab k^a k^b
        if isinstance(T_ab, dict):
            # If T_ab is full tensor, do proper contraction
            stress_contraction = T_ab.get('00', 0.0)  # Simplified: just T_00
        else:
            # If T_ab is scalar (e.g., T_00 component)
            stress_contraction = T_ab
        
        # Apply null vector contraction
        # For timelike k^a = (1, v_i), we get T_00 + 2T_0i v_i + T_ij v_i v_j
        # For null case, k^0 = |k|, k^i = k^0 n^i with |n| = 1
        k_magnitude_squared = sum(ki**2 for ki in k)
        stress_projected = stress_contraction * k_magnitude_squared
        
        # Apply sampling function if provided
        if sampling_func is not None:
            weight = sampling_func(t, tau)
            stress_projected *= weight
        
        integral += stress_projected * dλ
    
    return integral

def polymer_corrected_anec_bound(tau: float, mu: float, 
                               dimension: int = 4) -> float:
    """
    Compute polymer-corrected ANEC bound.
    
    The classical Ford-Roman bound gets modified by LQG polymer corrections:
    ∫ ⟨T_{00}⟩ f(t) dt ≥ -C/(τ²) × sinc(πμ)
    
    :param tau: Sampling timescale
    :param mu: Polymer parameter
    :param dimension: Spacetime dimension
    :return: Modified ANEC bound (negative value)
    """
    return polymer_quantum_inequality_bound(tau, mu, dimension)

def null_geodesic_through_network(spin_network: SpinNetwork, 
                                direction: np.ndarray = np.array([1, 1, 0, 0]),
                                num_points: int = 100) -> List[Tuple[float, ...]]:
    """
    Generate null geodesic path through spin network.
    
    :param spin_network: SpinNetwork to traverse
    :param direction: 4-vector direction (will be normalized to null)
    :param num_points: Number of points along geodesic
    :return: List of 4-vectors along geodesic
    """
    # Normalize to null vector: k^a k_a = 0
    # For Minkowski: k^0 = √(k^i k_i)
    spatial_part = direction[1:]
    spatial_magnitude = np.linalg.norm(spatial_part)
    
    if spatial_magnitude == 0:
        # Pure time direction - not null
        spatial_part = np.array([1, 0, 0])
        spatial_magnitude = 1.0
    
    # Construct null vector
    k0 = spatial_magnitude  # For null: k^0 = |k⃗|
    null_vector = np.array([k0] + list(spatial_part))
    
    # Generate geodesic path
    geodesic = []
    for i in range(num_points):
        # Parameter λ along geodesic
        lambda_param = i / num_points
        
        # For null geodesic in flat space: x^μ(λ) = x₀^μ + λ k^μ
        point = null_vector * lambda_param
        geodesic.append(tuple(point))
    
    return geodesic

def coherent_state_anec_violation(n_nodes: int = 100, 
                                alpha: float = 0.05,
                                mu: float = 0.1,
                                tau: float = 1.0,
                                field_amplitude: float = 1.0) -> Dict:
    """
    Comprehensive ANEC violation analysis using coherent states.
    
    :param n_nodes: Number of nodes in spin network
    :param alpha: Coherent state spread parameter
    :param mu: Polymer correction parameter
    :param tau: Sampling timescale
    :param field_amplitude: Scalar field amplitude
    :return: Dictionary with analysis results
    """    # Build spin network and coherent state
    graph = build_flat_graph(n_nodes, connectivity="cubic")
    coherent_state = CoherentState(graph, alpha)
    coherent_state.peak_on_flat()  # Configure the state but keep the CoherentState object
    
    # Set up scalar field configuration
    field_config = {}
    for i, vertex in enumerate(graph.nodes):
        # Gaussian field profile
        x = i / n_nodes
        field_config[vertex] = field_amplitude * np.exp(-(x - 0.5)**2 / (2 * alpha**2))
    
    # Initialize stress tensor operators
    classical_stress_op = ScalarFieldStressTensor(mass=0.0)
    classical_stress_op.set_field_configuration(field_config)
    
    polymer_stress_op = LocalT00(alpha=1.0, beta=1.0)
    
    # Midisuperspace model
    model_params = {'mu': mu, 'gamma': 0.2375, 'lambda': 0.0, 'G': 1.0}
    midi_model = MidiSuperspaceModel(model_params)
    
    # Compute stress tensors
    T_classical = classical_stress_op.apply(graph)
    T_midi = midi_model.expectation_T00(coherent_state)
    T_polymer = polymer_stress_op.apply(graph)
    
    # Define null geodesic
    n_points = len(T_classical)
    dλ = 1.0 / n_points
    geodesic = [(1.0, 1.0, 0.0, 0.0)] * n_points  # Simple null vector
    
    # Compute ANEC integrals
    anec_classical = compute_anec_integral(
        list(T_classical.values()), geodesic, dλ,
        sampling_func=ford_roman_sampling_function, tau=tau
    )
    
    anec_midi = compute_anec_integral(
        T_midi.tolist(), geodesic, dλ,
        sampling_func=ford_roman_sampling_function, tau=tau
    )
    
    # Compute bounds
    classical_bound = -3.0 / (32 * np.pi**2 * tau**2)  # Ford-Roman bound
    polymer_bound = polymer_corrected_anec_bound(tau, mu)
    
    # Analysis results
    results = {
        'anec_integral_classical': anec_classical,
        'anec_integral_midisuperspace': anec_midi,
        'anec_bound': classical_bound,
        'polymer_bound': polymer_bound,
        'classical_violation': anec_classical < classical_bound,
        'polymer_violation': anec_midi < polymer_bound,
        'violation_magnitude_classical': abs(anec_classical / classical_bound) if classical_bound != 0 else 0,
        'violation_magnitude_polymer': abs(anec_midi / polymer_bound) if polymer_bound != 0 else 0,
        'parameters': {
            'n_nodes': n_nodes, 'alpha': alpha, 'mu': mu, 'tau': tau,
            'field_amplitude': field_amplitude
        }
    }
    
    return results

def scan_anec_violation_parameters(mu_range: np.ndarray,
                                 tau_range: np.ndarray,
                                 n_nodes: int = 64,
                                 alpha: float = 0.05) -> Dict:
    """
    Systematic scan of ANEC violation in (μ, τ) parameter space.
    
    :param mu_range: Array of polymer parameter values
    :param tau_range: Array of sampling timescales
    :param n_nodes: Number of spin network nodes
    :param alpha: Coherent state spread
    :return: Scan results
    """
    violation_grid = np.zeros((len(mu_range), len(tau_range)))
    anec_grid = np.zeros((len(mu_range), len(tau_range)))
    bound_grid = np.zeros((len(mu_range), len(tau_range)))
    
    # Build base network once
    graph = build_flat_graph(n_nodes, connectivity="cubic")
    
    for i, mu in enumerate(mu_range):
        for j, tau in enumerate(tau_range):
            try:
                result = coherent_state_anec_violation(
                    n_nodes=n_nodes, alpha=alpha, mu=mu, tau=tau
                )
                
                anec_value = result['anec_integral_midisuperspace']
                bound_value = result['polymer_bound']
                
                anec_grid[i, j] = anec_value
                bound_grid[i, j] = bound_value
                
                # Violation magnitude
                if bound_value != 0:
                    violation_grid[i, j] = abs(anec_value / bound_value) if anec_value < bound_value else 0
                else:
                    violation_grid[i, j] = 0
                    
            except Exception as e:
                print(f"Error at μ={mu:.3f}, τ={tau:.3f}: {e}")
                violation_grid[i, j] = np.nan
                anec_grid[i, j] = np.nan
                bound_grid[i, j] = np.nan
    
    return {
        'violation_grid': violation_grid,
        'anec_grid': anec_grid,
        'bound_grid': bound_grid,
        'mu_range': mu_range,
        'tau_range': tau_range,
        'parameters': {'n_nodes': n_nodes, 'alpha': alpha}
    }

def effective_action_anec_analysis(graph: SpinNetwork, 
                                 coherent_state: CoherentState) -> Dict:
    """
    ANEC violation analysis using effective field theory approach.
    
    :param graph: Spin network
    :param coherent_state: Coherent state
    :return: EFT analysis results
    """
    # Model spin-foam data for EFT derivation
    spinfoam_data = {
        'amplitudes': {i: np.exp(1j * i * 0.1) for i in range(len(list(graph.nodes)))},
        'boundary_data': {'vertex_count': len(list(graph.nodes))},
        'coupling_constant': 1.0,  # Planck units
        'matter_fields': {'scalar': {'amplitude': 1.0}}
    }
    
    # Compute EFT coefficients
    eft_coefficients = compute_effective_terms(spinfoam_data)
    
    # Estimate ANEC violation from EFT
    background_curvature = 0.1  # Model background
    anec_violation_eft = derive_anec_violation_from_eft(eft_coefficients, background_curvature)
    
    return {
        'eft_coefficients': eft_coefficients,
        'anec_violation_eft': anec_violation_eft,
        'background_curvature': background_curvature
    }

def save_anec_analysis_plots(results: Dict, output_path: str, 
                           plot_format: str = 'png') -> None:
    """
    Save analysis plots to files (no interactive display).
    
    :param results: Analysis results dictionary
    :param output_path: Directory to save plots
    :param plot_format: File format (png, pdf, etc.)
    """
    os.makedirs(output_path, exist_ok=True)
    
    if 'violation_grid' in results:
        # Parameter scan heatmap
        plt.figure(figsize=(10, 8))
        mu_range = results['mu_range']
        tau_range = results['tau_range']
        
        plt.imshow(results['violation_grid'].T, 
                  extent=[mu_range.min(), mu_range.max(), 
                         tau_range.min(), tau_range.max()],
                  aspect='auto', origin='lower', cmap='plasma')
        plt.colorbar(label='Violation Magnitude |I/B|')
        plt.xlabel('Polymer parameter μ')
        plt.ylabel('Sampling timescale τ')
        plt.title('ANEC Violation Parameter Scan')
        plt.yscale('log')
        
        filename = os.path.join(output_path, f'anec_parameter_scan.{plot_format}')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved parameter scan plot: {filename}")
    
    if 'anec_integral_classical' in results:
        # Single point analysis
        values = [
            results['anec_integral_classical'],
            results['anec_integral_midisuperspace'], 
            results['anec_bound'],
            results.get('polymer_bound', results['anec_bound'])
        ]
        labels = ['Classical ANEC', 'Midisuperspace ANEC', 'Classical Bound', 'Polymer Bound']
        
        plt.figure(figsize=(10, 6))
        x_pos = np.arange(len(labels))
        colors = ['blue', 'red', 'black', 'green']
        
        bars = plt.bar(x_pos, values, color=colors, alpha=0.7)
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        plt.xlabel('Analysis Type')
        plt.ylabel('ANEC Integral Value')
        plt.title('ANEC Violation Analysis Comparison')
        plt.xticks(x_pos, labels, rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{val:.2e}', ha='center', va='bottom', fontsize=9)
        
        filename = os.path.join(output_path, f'anec_comparison.{plot_format}')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison plot: {filename}")

def main():
    """
    Main ANEC violation analysis driver.
    """
    print("LQG-ANEC Framework: Comprehensive ANEC Violation Analysis")
    print("=" * 60)
    
    # 1. Single point analysis
    print("\n1. Single Point Analysis:")
    single_result = coherent_state_anec_violation(
        n_nodes=64, alpha=0.05, mu=0.1, tau=1.0
    )
    
    print(f"ANEC Integral (Classical): {single_result['anec_integral_classical']:.3e}")
    print(f"ANEC Integral (Midisuperspace): {single_result['anec_integral_midisuperspace']:.3e}")
    print(f"ANEC Bound: {single_result['anec_bound']:.3e}")
    print(f"Classical Violation: {single_result['classical_violation']}")
    print(f"Midisuperspace Violation: {single_result['midisuperspace_violation']}")
    
    # 2. Parameter scan
    print("\n2. Parameter Space Scan:")
    mu_range = np.linspace(0.01, 0.5, 20)
    tau_range = np.logspace(-1, 1, 20)  # 0.1 to 10
    
    scan_results = scan_anec_violation_parameters(mu_range, tau_range, n_nodes=32)
    
    # Find maximum violation
    max_violation_idx = np.unravel_index(
        np.nanargmax(scan_results['violation_grid']), 
        scan_results['violation_grid'].shape
    )
    max_mu = mu_range[max_violation_idx[0]]
    max_tau = tau_range[max_violation_idx[1]]
    max_violation = scan_results['violation_grid'][max_violation_idx]
    
    print(f"Maximum violation: {max_violation:.3f} at μ={max_mu:.3f}, τ={max_tau:.3f}")
    
    # 3. Save plots
    print("\n3. Generating Analysis Plots:")
    save_anec_analysis_plots(scan_results)
    
    # 4. EFT analysis
    print("\n4. Effective Field Theory Analysis:")
    graph = build_flat_graph(32)
    coh_state = CoherentState(graph, 0.05).peak_on_flat()
    eft_results = effective_action_anec_analysis(graph, coh_state)
    
    print(f"EFT ANEC Violation: {eft_results['anec_violation_eft']:.3e}")
    print(f"Key EFT Coefficients:")
    for term, coeff in eft_results['eft_coefficients'].items():
        if abs(coeff) > 1e-10:
            print(f"  {term}: {coeff:.3e}")
    
    print(f"\nAnalysis complete. Results saved to results/ directory.")

if __name__ == "__main__":
    main()
