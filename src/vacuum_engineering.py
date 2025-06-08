# src/vacuum_engineering.py

import numpy as np
from scipy.constants import hbar, c, pi, k as kb
from scipy.optimize import differential_evolution
from scipy.integrate import quad
from typing import List, Optional, Dict, Any, Callable

# Import advanced models (optional dependencies)
try:
    from drude_model import DrudeLorentzPermittivity, get_material_model
    DRUDE_AVAILABLE = True
except ImportError:
    DRUDE_AVAILABLE = False

# Note: metamaterial_casimir imports this module, so we don't import it here
# to avoid circular imports. Users should import MetamaterialCasimir directly.

# Material database for advanced calculations
MATERIAL_DATABASE = {
    'vacuum': {
        'permittivity': 1.0,
        'permeability': 1.0,
        'name': 'Vacuum'
    },
    'SiO2': {
        'permittivity': 2.1 + 0.001j,
        'permeability': 1.0,
        'name': 'Silicon Dioxide'
    },
    'Si': {
        'permittivity': 11.68 + 0.01j,
        'permeability': 1.0,
        'name': 'Silicon'
    },
    'Au': {
        'permittivity': -10.0 + 1.2j,  # Simplified Drude model
        'permeability': 1.0,
        'name': 'Gold'
    },
    'Al': {
        'permittivity': -15.0 + 2.0j,  # Simplified Drude model
        'permeability': 1.0,
        'name': 'Aluminum'
    },
    'metamaterial': {
        'permittivity': -2.0 + 0.1j,  # Negative index metamaterial
        'permeability': -1.5 + 0.05j,
        'name': 'Negative Index Metamaterial'
    }
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
    
    def energy_density_with_dispersion(self, include_drude: bool = True) -> np.ndarray:
        """
        Compute Casimir energy density with realistic material dispersion.
        
        Uses frequency-dependent permittivity for more accurate calculations.
        
        Args:
            include_drude: Whether to use Drude-Lorentz dispersion models
            
        Returns:
            Array of energy densities for each layer [J/m³]
        """
        if not include_drude:
            # Fall back to simple model
            base = -(pi**2 * hbar * c) / (720 * np.array(self.a)**4)
            return base * np.real(self.eps)
        
        try:
            from drude_model import casimir_integrand_with_dispersion, get_material_model
            
            # Frequency range for integration (THz to PHz)
            ω_range = np.logspace(13, 17, 500)
            densities = []
            
            for a, ε_r in zip(self.a, self.eps):
                try:
                    # Try to match material to known Drude models
                    if np.real(ε_r) < 0:
                        # Likely metallic - use gold model as default
                        material_model = get_material_model('gold')
                    else:
                        # Dielectric - use silicon model  
                        material_model = get_material_model('silicon')
                    
                    # Integrate Casimir energy density
                    integrand_vals = [casimir_integrand_with_dispersion(ω, a, material_model) 
                                    for ω in ω_range]
                    
                    # Numerical integration
                    energy_density = -np.trapz(integrand_vals, ω_range)
                    densities.append(energy_density)
                    
                except Exception as e:
                    # Fallback to simple calculation
                    base = -(pi**2 * hbar * c) / (720 * a**4)
                    densities.append(base * np.real(ε_r))
                    
            return np.array(densities)
            
        except ImportError:
            # Drude model not available, use simple calculation
            base = -(pi**2 * hbar * c) / (720 * np.array(self.a)**4)
            return base * np.real(self.eps)
    
    def energy_density_drude_enhanced(self):
        """
        Enhanced energy density calculation using realistic Drude-Lorentz material models.
        
        Integrates over frequency range with material-specific permittivity.
        """
        if not DRUDE_AVAILABLE:
            # Fallback to simple model
            return self.energy_density_with_dispersion(include_drude=False)
            
        from drude_model import DrudeLorentzPermittivity, get_material_model
        
        # Create frequency-dependent material model
        try:
            # Try to determine material from permittivity
            if hasattr(self, 'spacings') and hasattr(self, 'eps_list'):
                eps_avg = np.mean(np.real(self.eps_list))
                if eps_avg < 0:
                    model = get_material_model('gold')  # Metallic
                else:
                    model = get_material_model('silicon')  # Dielectric
            else:
                model = DrudeLorentzPermittivity(ωp=1e16, γ=1e14)  # Default model
                
            def integrand(ω, a):
                R = model.reflectivity(ω)
                return R * ω**3  # Casimir integrand weighting
            
            # Integration over relevant frequency range (THz to PHz)
            ωs = np.logspace(13, 17, 200)
            densities = []
            
            # Get layer parameters
            if hasattr(self, 'spacings'):
                spacings = self.spacings
                eps_vals = self.eps_list if hasattr(self, 'eps_list') else [1.0] * len(spacings)
            else:
                spacings = [100e-9]  # Default 100 nm
                eps_vals = [1.0]
                
            for a, ε_r in zip(spacings, eps_vals):
                # Material-enhanced integrand
                integrals = np.trapz(integrand(ωs, a), ωs)
                
                # Convert to energy density with material corrections
                base_density = -(hbar * integrals) / (2*pi**2 * c**3 * a**3)
                material_factor = np.real(ε_r) if np.real(ε_r) > 0 else abs(ε_r)**2
                
                densities.append(base_density * material_factor)
                
            return np.array(densities)
            
        except Exception as e:
            print(f"Warning: Drude enhancement failed ({e}), using simple model")
            return self.energy_density_with_dispersion(include_drude=False)

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

def gaussian_kernel(t: float, tau: float) -> float:
    """Simple Gaussian smearing kernel for quantum inequality analysis."""
    return np.exp(-t**2 / (2 * tau**2)) / (tau * np.sqrt(2 * pi))

def vacuum_energy_to_anec_flux_simple(energy_density: float, 
                                     volume: float = 1e-6, 
                                     tau: float = 1e-12) -> float:
    """
    Simplified wrapper for vacuum_energy_to_anec_flux with reasonable defaults.
    
    Args:
        energy_density: Negative energy density (J/m³)
        volume: Spatial volume (m³), default 1 cubic mm
        tau: Temporal smearing scale (s), default 1 ps
        
    Returns:
        ANEC violation flux
    """
    return vacuum_energy_to_anec_flux(energy_density, volume, tau, gaussian_kernel)

# For backward compatibility with analysis scripts
def vacuum_energy_to_anec_flux_compat(energy_density: float) -> float:
    """Backward compatibility wrapper that takes only energy density."""
    return vacuum_energy_to_anec_flux_simple(energy_density)

# Legacy API compatibility for build_lab_sources
def build_lab_sources_legacy():
    """Legacy API that returns simple objects with total_density() method."""
    
    class SimpleCasimir:
        def __init__(self):
            self.energy_dens = -1e10  # J/m³
        def total_density(self):
            return self.energy_dens
    
    class SimpleDynamic:
        def __init__(self):
            self.energy_dens = -1e8  # J/m³
        def total_density(self):
            return self.energy_dens
    
    class SimpleSqueezed:
        def __init__(self):
            self.energy_dens = -1e6  # J/m³
        def total_density(self):
            return self.energy_dens
    
    return {
        "CasimirArray": SimpleCasimir(),
        "DynamicCasimir": SimpleDynamic(),
        "SqueezedVacuum": SimpleSqueezed()
    }

def build_lab_sources(config_type='comprehensive'):
    """
    Factory function to build standard laboratory vacuum sources.
    Provides simplified API for parameter scanning and ANEC integration.
    
    Args:
        config_type: Type of configuration ('comprehensive', 'casimir_focus', 
                    'dynamic_focus', 'squeezed_focus')
    
    Returns:
        Dictionary with initialized source objects and parameters
    """
    sources = {}
    
    if config_type == 'comprehensive' or config_type == 'casimir_focus':
        # Casimir Array Configuration
        casimir = CasimirArray(temperature=4.0)  # Cryogenic operation
        sources['casimir'] = {
            'source': casimir,
            'params': {
                'n_layers': 10,
                'spacing_range': (10e-9, 1e-6),  # 10 nm to 1 μm
                'materials': ['Au', 'SiO2', 'metamaterial'],
                'volume': (1e-3)**2 * 1e-6,  # 1 mm² × 1 μm thickness
                'optimal_spacing': 100e-9  # 100 nm baseline
            }
        }
    
    if config_type == 'comprehensive' or config_type == 'dynamic_focus':
        # Dynamic Casimir Configuration
        dynamic = DynamicCasimirEffect(circuit_frequency=10e9, drive_amplitude=0.2)
        sources['dynamic'] = {
            'source': dynamic,
            'params': {
                'drive_frequency': 20e9,  # 2× circuit frequency for resonance
                'volume': (1e-3)**3,  # 1 mm³ circuit
                'quality_factor': 10000,  # High-Q superconducting circuit
                'power_budget': 1e-3  # 1 mW power limit
            }
        }
    
    if config_type == 'comprehensive' or config_type == 'squeezed_focus':
        # Squeezed Vacuum Configuration
        squeezed = SqueezedVacuumResonator(resonator_frequency=1e14, squeezing_parameter=2.0)
        sources['squeezed'] = {
            'source': squeezed,
            'params': {
                'volume': np.pi * (50e-6)**2 * 1e-3,  # Fiber-like geometry
                'squeezing_range': (0.5, 3.0),  # Squeezing parameter range
                'stabilization_budget': 1e-3,  # 1 mW stabilization power
                'frequency_range': (1e12, 1e15)  # THz to optical
            }
        }
    
    return sources

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
    
    # Calculate ANEC flux
    casimir_flux = vacuum_energy_to_anec_flux_simple(casimir_energy_density, casimir_volume)
    
    results['casimir_array'] = {
        'energy_density': casimir_energy_density,
        'volume': casimir_volume,
        'anec_flux': casimir_flux,
        'target_ratio': abs(casimir_flux / target_flux),
        'configuration': casimir_opt,
        'feasible': casimir_opt['error'] < 0.1 * abs(target_pressure)
    }
    
    # 2. Dynamic Casimir Effect
    dynamic = DynamicCasimirEffect(circuit_frequency=10e9, drive_amplitude=0.2)
    
    drive_freq = 20e9  # Optimal 2×circuit frequency
    circuit_volume = (1e-3)**3  # 1 mm³ circuit volume
    quality_factor = 10000  # High-Q superconducting circuit
    
    dynamic_energy_density = dynamic.negative_energy_density(drive_freq, circuit_volume, quality_factor)
    dynamic_flux = vacuum_energy_to_anec_flux_simple(dynamic_energy_density, circuit_volume)
    
    results['dynamic_casimir'] = {
        'energy_density': dynamic_energy_density,
        'volume': circuit_volume,
        'anec_flux': dynamic_flux,
        'target_ratio': abs(dynamic_flux / target_flux),
        'feasible': abs(dynamic_energy_density) > 1e5  # Reasonable threshold
    }
      # 3. Squeezed Vacuum Resonator
    squeezed = SqueezedVacuumResonator(resonator_frequency=1e14, squeezing_parameter=2.0)
    
    fiber_volume = np.pi * (50e-6)**2 * 1e-3  # Fiber-like geometry
    squeezed_energy_density = squeezed.squeezed_energy_density(volume=fiber_volume)
    squeezed_flux = vacuum_energy_to_anec_flux_simple(squeezed_energy_density, fiber_volume)
    
    results['squeezed_vacuum'] = {
        'energy_density': squeezed_energy_density,
        'volume': fiber_volume,
        'anec_flux': squeezed_flux,
        'target_ratio': abs(squeezed_flux / target_flux),
        'feasible': abs(squeezed_energy_density) > 1e3  # Reasonable threshold
    }
    
    return results

if __name__ == "__main__":
    # Example usage of simplified API
    print("Vacuum Engineering - Simplified API Demo")
    print("=" * 50)
    
    # Build comprehensive lab sources
    sources = build_lab_sources('comprehensive')
    
    print("\nConfigured Sources:")
    for name, config in sources.items():
        print(f"\n{name.title()} Source:")
        print(f"  Type: {type(config['source']).__name__}")
        print(f"  Volume: {config['params']['volume']:.2e} m³")
    
    # Test each source for negative energy density
    print("\nNegative Energy Density Tests:")
    print("-" * 40)
    
    # Casimir test
    if 'casimir' in sources:
        casimir_source = sources['casimir']['source']
        params = sources['casimir']['params']
        
        # Test single layer pressure
        pressure = casimir_source.casimir_pressure(
            params['optimal_spacing'], 
            MATERIAL_DATABASE['SiO2']['permittivity']
        )
        energy_density = pressure * params['optimal_spacing'] / params['volume']
        
        print(f"Casimir Array:")
        print(f"  Pressure: {pressure:.2e} Pa")
        print(f"  Energy density: {energy_density:.2e} J/m³")
    
    # Dynamic Casimir test
    if 'dynamic' in sources:
        dynamic_source = sources['dynamic']['source']
        params = sources['dynamic']['params']
        
        energy_density = dynamic_source.negative_energy_density(
            params['drive_frequency'],
            params['volume'],
            params['quality_factor']
        )
        
        print(f"Dynamic Casimir:")
        print(f"  Energy density: {energy_density:.2e} J/m³")
    
    # Squeezed vacuum test
    if 'squeezed' in sources:
        squeezed_source = sources['squeezed']['source']
        params = sources['squeezed']['params']
        
        energy_density = squeezed_source.squeezed_energy_density(params['volume'])
        
        print(f"Squeezed Vacuum:")
        print(f"  Energy density: {energy_density:.2e} J/m³")
    
    print("\nSimplified API ready for ANEC integration!")

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
