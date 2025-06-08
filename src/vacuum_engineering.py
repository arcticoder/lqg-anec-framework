# src/vacuum_engineering.py
"""
Vacuum Engineering Module for Negative Energy Sources

This module implements theoretical frameworks for laboratory-proven sources
of negative energy, including:

1. Casimir effect between parallel plates and metamaterial arrays
2. Dynamic Casimir effect in superconducting circuits  
3. Squeezed vacuum states in optical/microwave resonators
4. Stacked photonic-crystal "meta-Casimir" arrays
5. Optimization for maximum negative energy density

Key Features:
- Multi-layer Casimir pressure calculation with material corrections
- Dynamic Casimir effect modeling with GHz pump drives
- Squeezed vacuum energy density with active stabilization
- Metamaterial enhancement factors for exotic geometries
- Integration with existing ANEC violation analysis

Author: LQG-ANEC Framework - Vacuum Engineering Team
"""

import numpy as np
from scipy.constants import hbar, c, pi, epsilon_0, mu_0, k as kb
from scipy.optimize import minimize, differential_evolution
from scipy.integrate import quad, simpson
from scipy.special import zeta
from typing import Dict, List, Tuple, Optional, Union, Callable
import matplotlib.pyplot as plt
import warnings

# Material property database for common Casimir materials
MATERIAL_DATABASE = {
    'vacuum': {'permittivity': 1.0, 'permeability': 1.0, 'conductivity': 0.0},
    'SiO2': {'permittivity': 3.9, 'permeability': 1.0, 'conductivity': 1e-15},
    'Au': {'permittivity': -1.0 + 1j*10, 'permeability': 1.0, 'conductivity': 4.5e7},
    'Si': {'permittivity': 11.7, 'permeability': 1.0, 'conductivity': 1e-4},
    'Al': {'permittivity': -1.0 + 1j*15, 'permeability': 1.0, 'conductivity': 3.8e7},
    'metamaterial': {'permittivity': -2.5, 'permeability': -1.2, 'conductivity': 0.0}
}

class CasimirArray:
    """
    Multi-layer Casimir cavity system with metamaterial enhancements.
    
    Implements:
    - Standard Casimir pressure between parallel plates
    - Material-dependent corrections via permittivity/permeability
    - Multi-layer stacking for pressure amplification
    - Metamaterial negative-index enhancements
    """
    def __init__(self, temperature: float = 300.0):
        """
        Initialize Casimir array system.
        
        Args:
            temperature: Operating temperature in Kelvin
        """
        self.T = temperature
        self.beta = 1.0 / (kb * temperature) if temperature > 0 else np.inf
        
        # Add missing constants
        self.c = c
        self.hbar = hbar
        
        print(f"Casimir Array System initialized:")
        print(f"  Temperature: {self.T:.1f} K")
        print(f"  Thermal length: {c * hbar * self.beta:.2e} m")
    
    def casimir_pressure(self, a: float, material_perm: complex = 1.0, 
                        material_perm_mu: complex = 1.0) -> float:
        """
        Compute Casimir pressure between two plates with material corrections.
        
        Standard result: P = -(π²ℏc)/(240a⁴)
        Material corrections applied via effective refractive index.
        
        Args:
            a: Plate separation in meters
            material_perm: Relative permittivity εᵣ
            material_perm_mu: Relative permeability μᵣ
            
        Returns:
            Casimir pressure in Pa (negative for attractive force)
        """
        # Base Casimir pressure (attractive, hence negative)
        P0 = -(pi**2 * hbar * c) / (240 * a**4)
        
        # Material enhancement factor
        # For metamaterials with ε < 0, μ < 0, can get repulsive Casimir force
        n_eff = np.sqrt(material_perm * material_perm_mu)
          # Correction factor - simplified model
        if np.real(n_eff) < 0:
            # Metamaterial case - can reverse sign
            correction = -np.abs(n_eff)**2
        else:
            # Normal materials - enhancement factor
            correction = np.real(n_eff)
        
        return P0 * correction
    
    def thermal_casimir_pressure(self, a: float, material_perm: complex = 1.0) -> float:
        """
        Casimir pressure with thermal corrections at finite temperature.
        
        Uses Lifshitz formula with thermal modifications.
        """
        thermal_length = self.c * self.hbar * self.beta
        
        if a < thermal_length:
            # Zero-temperature limit
            return self.casimir_pressure(a, material_perm)
        else:
            # Thermal suppression
            thermal_factor = np.exp(-2 * pi * a / thermal_length)
            return self.casimir_pressure(a, material_perm) * thermal_factor
    
    def stack_pressure(self, layers: int, spacing_list: List[float], 
                      perm_list: List[complex], mu_list: Optional[List[complex]] = None) -> float:
        """
        Compute net pressure from stacked Casimir cavities.
        
        Args:
            layers: Number of cavity layers
            spacing_list: List of plate separations for each layer
            perm_list: List of permittivities for each layer
            mu_list: List of permeabilities for each layer (optional)
            
        Returns:
            Total negative pressure per unit area
        """
        if mu_list is None:
            mu_list = [1.0] * layers
            
        if len(spacing_list) != layers or len(perm_list) != layers:
            raise ValueError("Mismatch in layer specifications")
        
        total_pressure = 0.0
        
        for i in range(layers):
            layer_pressure = self.thermal_casimir_pressure(
                spacing_list[i], perm_list[i]
            )
            total_pressure += layer_pressure
            
        return total_pressure
    
    def optimize_stack(self, n_layers: int, a_min: float, a_max: float, 
                      materials: List[str], target_pressure: float,
                      method: str = 'grid') -> Dict:
        """
        Optimize layer configuration to achieve target negative pressure.
        
        Args:
            n_layers: Number of layers to optimize
            a_min, a_max: Minimum and maximum plate separations
            materials: List of material names from database
            target_pressure: Target negative pressure (Pa)
            method: Optimization method ('grid' or 'evolution')
            
        Returns:
            Dictionary with optimal configuration and achieved pressure
        """
        best_result = {'spacing': None, 'materials': None, 'pressure': 0.0, 'error': np.inf}
        
        if method == 'grid':
            # Grid search over configurations
            spacing_grid = np.linspace(a_min, a_max, 20)
            
            for material in materials:
                mat_props = MATERIAL_DATABASE[material]
                perm = mat_props['permittivity']
                mu = mat_props['permeability']
                
                for spacing in spacing_grid:
                    spacing_list = [spacing] * n_layers
                    perm_list = [perm] * n_layers
                    mu_list = [mu] * n_layers
                    
                    pressure = self.stack_pressure(n_layers, spacing_list, perm_list, mu_list)
                    error = abs(pressure - target_pressure)
                    
                    if error < best_result['error']:
                        best_result = {
                            'spacing': spacing,
                            'materials': [material] * n_layers,
                            'pressure': pressure,
                            'error': error,
                            'spacing_list': spacing_list,
                            'enhancement': pressure / self.casimir_pressure(spacing)
                        }
        
        elif method == 'evolution':
            # Differential evolution optimization
            def objective(params):
                # params = [spacing_1, ..., spacing_n, material_indices]
                spacings = params[:n_layers]
                mat_indices = np.clip(np.round(params[n_layers:]).astype(int), 
                                    0, len(materials)-1)
                
                perm_list = [MATERIAL_DATABASE[materials[i]]['permittivity'] 
                           for i in mat_indices]
                mu_list = [MATERIAL_DATABASE[materials[i]]['permeability'] 
                          for i in mat_indices]
                
                pressure = self.stack_pressure(n_layers, spacings, perm_list, mu_list)
                return abs(pressure - target_pressure)
            
            bounds = [(a_min, a_max)] * n_layers + [(0, len(materials)-1)] * n_layers
            result = differential_evolution(objective, bounds, seed=42, maxiter=100)
            
            if result.success:
                spacings = result.x[:n_layers]
                mat_indices = np.clip(np.round(result.x[n_layers:]).astype(int), 
                                    0, len(materials)-1)
                
                perm_list = [MATERIAL_DATABASE[materials[i]]['permittivity'] 
                           for i in mat_indices]
                mu_list = [MATERIAL_DATABASE[materials[i]]['permeability'] 
                          for i in mat_indices]
                
                pressure = self.stack_pressure(n_layers, spacings, perm_list, mu_list)
                
                best_result = {
                    'spacing_list': spacings,
                    'materials': [materials[i] for i in mat_indices],
                    'pressure': pressure,
                    'error': result.fun,
                    'enhancement': pressure / self.casimir_pressure(np.mean(spacings))
                }
        
        return best_result

class DynamicCasimirEffect:
    """
    Dynamic Casimir effect in superconducting circuits with GHz drives.
    
    Models photon creation from oscillating boundary conditions
    and associated negative energy densities.
    """
    def __init__(self, circuit_frequency: float = 10e9, drive_amplitude: float = 0.1):
        """
        Initialize dynamic Casimir system.
        
        Args:
            circuit_frequency: Base circuit frequency in Hz
            drive_amplitude: Dimensionless drive amplitude
        """
        self.f0 = circuit_frequency
        self.omega0 = 2 * pi * circuit_frequency
        self.drive_amp = drive_amplitude
        
        print(f"Dynamic Casimir Effect System:")
        print(f"  Circuit frequency: {self.f0:.2e} Hz")
        print(f"  Drive amplitude: {self.drive_amp:.3f}")
    
    def photon_creation_rate(self, drive_frequency: float, quality_factor: float = 1000) -> float:
        """
        Compute photon creation rate from oscillating boundary.
        
        Rate ∝ (drive amplitude)² × (quality factor) × frequency scaling
        """
        omega_drive = 2 * pi * drive_frequency
        
        # Resonance enhancement when drive ≈ 2×circuit frequency
        resonance_factor = quality_factor / (1 + ((omega_drive - 2*self.omega0) / (self.omega0/quality_factor))**2)
        
        # Base rate (simplified model)
        rate = (self.drive_amp**2) * self.omega0 * resonance_factor / hbar
        
        return rate
    
    def negative_energy_density(self, drive_frequency: float, volume: float, 
                              quality_factor: float = 1000) -> float:
        """
        Estimate negative energy density from dynamic Casimir effect.
        
        Negative energy appears during photon creation process.
        """
        rate = self.photon_creation_rate(drive_frequency, quality_factor)
        omega_drive = 2 * pi * drive_frequency
        
        # Energy per created photon pair
        photon_energy = hbar * omega_drive / 2
        
        # Negative energy density (transient during creation)
        energy_density = -(rate * photon_energy) / volume
        
        return energy_density

class SqueezedVacuumResonator:
    """
    Squeezed vacuum states in optical/microwave resonators with active stabilization.
    
    Models continuous squeezed-vacuum channels for sustained negative energy.
    """
    
    def __init__(self, resonator_frequency: float = 1e12, squeezing_parameter: float = 1.0):
        """
        Initialize squeezed vacuum resonator.
        
        Args:
            resonator_frequency: Resonator frequency in Hz
            squeezing_parameter: Dimensionless squeezing strength
        """
        self.omega_res = 2 * pi * resonator_frequency
        self.xi = squeezing_parameter
        
        print(f"Squeezed Vacuum Resonator:")
        print(f"  Frequency: {resonator_frequency:.2e} Hz")
        print(f"  Squeezing parameter: {self.xi:.2f}")
    
    def squeezed_energy_density(self, volume: float) -> float:
        """
        Compute energy density of squeezed vacuum state.
        
        For squeezing parameter ξ, energy density can be negative
        for certain quadratures.
        """
        # Zero-point energy modification
        vacuum_energy = 0.5 * hbar * self.omega_res / volume
        
        # Squeezing modification - can yield negative contribution
        squeeze_factor = np.cosh(2 * self.xi) - np.sinh(2 * self.xi)
        
        return vacuum_energy * squeeze_factor
    
    def stabilization_power(self, feedback_bandwidth: float = 1e6) -> float:
        """
        Estimate power required for active stabilization of squeezed state.
        """
        # Power scales with squeezing strength and feedback bandwidth
        power = hbar * self.omega_res * (self.xi**2) * feedback_bandwidth
        return power

class MetamaterialCasimir:
    """
    Advanced metamaterial Casimir arrays with negative refractive index.
    
    Exploits metamaterial properties to enhance or reverse Casimir forces.
    """
    
    def __init__(self, unit_cell_size: float = 100e-9):
        """
        Initialize metamaterial Casimir system.
        
        Args:
            unit_cell_size: Metamaterial unit cell size in meters
        """
        self.a_cell = unit_cell_size
        
    def metamaterial_enhancement(self, epsilon: complex, mu: complex, 
                                frequency: float) -> complex:
        """
        Compute metamaterial enhancement factor for Casimir force.
        
        For double-negative metamaterials (ε < 0, μ < 0), can get
        repulsive Casimir forces.
        """
        omega = 2 * pi * frequency
        
        # Effective refractive index
        n_eff = np.sqrt(epsilon * mu)
        
        # Enhancement includes dispersion effects
        if np.real(n_eff) < 0:
            # Negative index - can reverse force
            enhancement = -np.abs(n_eff)**2
        else:
            # Positive index - force enhancement
            enhancement = np.abs(n_eff)**2
            
        return enhancement
    
    def design_optimal_metamaterial(self, target_enhancement: float) -> Dict:
        """
        Design metamaterial parameters to achieve target Casimir enhancement.
        """
        def objective(params):
            epsilon_r, epsilon_i, mu_r, mu_i = params
            epsilon = epsilon_r + 1j * epsilon_i
            mu = mu_r + 1j * mu_i
            
            # Frequency range for optimization
            frequencies = np.logspace(12, 15, 50)  # THz range
            enhancements = [self.metamaterial_enhancement(epsilon, mu, f) for f in frequencies]
            
            # Target: achieve desired enhancement across frequency range
            avg_enhancement = np.mean(np.real(enhancements))
            return abs(avg_enhancement - target_enhancement)
        
        # Optimization bounds for metamaterial parameters
        bounds = [(-5, 5), (0, 10), (-5, 5), (0, 10)]  # ε_r, ε_i, μ_r, μ_i
        
        result = differential_evolution(objective, bounds, seed=42)
        
        if result.success:
            epsilon_r, epsilon_i, mu_r, mu_i = result.x
            optimal_params = {
                'epsilon': epsilon_r + 1j * epsilon_i,
                'mu': mu_r + 1j * mu_i,
                'enhancement': -result.fun + target_enhancement,
                'feasible': np.abs(epsilon_r) < 10 and np.abs(mu_r) < 10
            }
        else:
            optimal_params = {'epsilon': 1.0, 'mu': 1.0, 'enhancement': 1.0, 'feasible': False}
            
        return optimal_params

def vacuum_energy_to_anec_flux(energy_density: float, volume: float, 
                              tau: float, smearing_kernel: Callable) -> float:
    """
    Convert vacuum negative energy density to ANEC violation flux.
    
    Integrates energy density with quantum inequality smearing kernel
    to compute effective ANEC violation.
    
    Args:
        energy_density: Negative energy density (J/m³)
        volume: Spatial volume over which energy exists (m³)
        tau: Temporal smearing scale (s)
        smearing_kernel: Function for temporal smearing f(t, tau)
        
    Returns:
        ANEC violation flux (W) integrated over null geodesic
    """
    total_energy = energy_density * volume
    
    # Temporal integration with smearing kernel
    def integrand(t):
        return smearing_kernel(t, tau) * total_energy / tau
    
    # Integrate over characteristic time scale
    flux, _ = quad(integrand, -3*tau, 3*tau)
    
    return flux

def comprehensive_vacuum_analysis(target_flux: float = 1e-25) -> Dict:
    """
    Comprehensive analysis of all vacuum engineering approaches.
    
    Compares Casimir arrays, dynamic Casimir, and squeezed vacuum
    for achieving target negative energy flux.
    """
    results = {}
    
    # 1. Casimir Array Analysis
    casimir = CasimirArray(temperature=4.0)  # Cryogenic operation
    
    # Test realistic parameters
    n_layers = 10
    a_range = (10e-9, 1e-6)  # 10 nm to 1 μm spacing
    materials = ['Au', 'SiO2', 'metamaterial']
    target_pressure = -1e6  # 1 MPa negative pressure
    
    casimir_opt = casimir.optimize_stack(n_layers, a_range[0], a_range[1], 
                                       materials, target_pressure, method='evolution')
    
    # Estimate energy density and volume
    typical_area = (1e-3)**2  # 1 mm² area
    typical_thickness = 1e-6  # 1 μm total thickness
    casimir_volume = typical_area * typical_thickness
    casimir_energy_density = casimir_opt['pressure'] * typical_thickness / casimir_volume
    
    results['casimir'] = {
        'energy_density': casimir_energy_density,
        'volume': casimir_volume,
        'configuration': casimir_opt,
        'feasible': casimir_opt['error'] < 0.1 * abs(target_pressure)
    }
    
    # 2. Dynamic Casimir Effect
    dynamic = DynamicCasimirEffect(circuit_frequency=10e9, drive_amplitude=0.2)
    
    drive_freq = 20e9  # Optimal 2×circuit frequency
    circuit_volume = (1e-3)**3  # 1 mm³ circuit volume
    quality_factor = 10000  # High-Q superconducting circuit
    
    dynamic_energy_density = dynamic.negative_energy_density(drive_freq, circuit_volume, quality_factor)
    
    results['dynamic_casimir'] = {
        'energy_density': dynamic_energy_density,
        'volume': circuit_volume,
        'drive_frequency': drive_freq,
        'feasible': abs(dynamic_energy_density) > 1e-15  # J/m³
    }
    
    # 3. Squeezed Vacuum Resonator
    squeezed = SqueezedVacuumResonator(resonator_frequency=1e14, squeezing_parameter=2.0)
    
    resonator_volume = pi * (50e-6)**2 * 1e-3  # Optical fiber-like geometry
    squeezed_energy_density = squeezed.squeezed_energy_density(resonator_volume)
    stabilization_power = squeezed.stabilization_power()
    
    results['squeezed_vacuum'] = {
        'energy_density': squeezed_energy_density,
        'volume': resonator_volume,
        'stabilization_power': stabilization_power,
        'feasible': stabilization_power < 1e-3  # < 1 mW
    }
    
    # 4. Convert to ANEC fluxes
    tau = 1e-6  # Microsecond timescale
    
    def gaussian_kernel(t, tau_scale):
        return np.exp(-t**2 / (2*tau_scale**2)) / np.sqrt(2*pi*tau_scale**2)
    
    for method in results:
        energy_density = results[method]['energy_density']
        volume = results[method]['volume']
        
        if energy_density < 0:  # Only negative energy contributes
            flux = vacuum_energy_to_anec_flux(energy_density, volume, tau, gaussian_kernel)
            results[method]['anec_flux'] = flux
            results[method]['target_ratio'] = abs(flux / target_flux)
        else:
            results[method]['anec_flux'] = 0.0
            results[method]['target_ratio'] = 0.0
    
    return results

# Simple interface functions as requested by user
def casimir_pressure(a, material_perm):
    """
    Compute idealized Casimir pressure between two plates:
      P = - (pi^2 ħ c) / (240 a^4)
    then apply a simple permittivity correction factor.
    
    Args:
        a: Plate separation in meters
        material_perm: Material permittivity correction factor
        
    Returns:
        Casimir pressure in Pa (negative for attractive)
    """
    P0 = -(pi**2 * hbar * c) / (240 * a**4)
    return P0 * material_perm

def stack_pressure(layers, spacing_list, perm_list):
    """
    Given N layers, each with spacing a_i and permittivity ε_i,
    return the net negative pressure per unit area.
    
    Args:
        layers: Number of layers (for consistency)
        spacing_list: List of plate separations
        perm_list: List of permittivity values
        
    Returns:
        Total negative pressure (Pa)
    """
    P_layers = [casimir_pressure(a, ε) for a, ε in zip(spacing_list, perm_list)]
    # approximate additivity
    return sum(P_layers)

def optimize_stack(n_layers, a_min, a_max, ε_vals, target_pressure):
    """
    Simple grid‐search over layer spacings and materials to reach
    target negative pressure (proxy for energy density).
    
    Args:
        n_layers: Number of layers
        a_min, a_max: Min/max spacing values
        ε_vals: List of permittivity values to try
        target_pressure: Target negative pressure
        
    Returns:
        Tuple of (best_spacing, best_permittivity, achieved_pressure)
    """
    best = None
    grid = np.linspace(a_min, a_max, 20)
    for perm in ε_vals:
        for spacing in grid:
            P = stack_pressure(n_layers, [spacing]*n_layers, [perm]*n_layers)
            if best is None or abs(P - target_pressure) < abs(best[2] - target_pressure):
                best = (spacing, perm, P)
    return best

if __name__ == "__main__":
    # Example usage and testing
    print("Vacuum Engineering Analysis")
    print("=" * 50)
    
    # Run comprehensive analysis
    analysis = comprehensive_vacuum_analysis()
    
    print("\nResults Summary:")
    print("-" * 30)
    
    for method, data in analysis.items():
        print(f"\n{method.replace('_', ' ').title()}:")
        print(f"  Energy density: {data['energy_density']:.2e} J/m³")
        print(f"  Volume: {data['volume']:.2e} m³")
        print(f"  ANEC flux: {data['anec_flux']:.2e} W")
        print(f"  Target ratio: {data['target_ratio']:.2e}")
        print(f"  Feasible: {data['feasible']}")
    
    # Find best approach
    best_method = max(analysis.keys(), key=lambda k: analysis[k]['target_ratio'])
    print(f"\nBest approach: {best_method.replace('_', ' ').title()}")
    print(f"Target ratio: {analysis[best_method]['target_ratio']:.2e}")
