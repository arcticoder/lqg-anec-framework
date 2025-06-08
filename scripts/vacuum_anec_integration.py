#!/usr/bin/env python3
"""
Vacuum Engineering ANEC Integration Module

Integrates laboratory vacuum engineering sources with the existing ANEC violation
analysis framework. Provides tools to:

1. Convert vacuum energy densities to ANEC violation predictions
2. Apply quantum inequality smearing kernels to lab sources
3. Interface with existing LQG coherent state analysis
4. Optimize vacuum parameters for maximum ANEC violation
5. Compare theoretical predictions with experimental constraints

Author: LQG-ANEC Framework - Integration Team
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.optimize import minimize, differential_evolution
from typing import Dict, List, Tuple, Optional, Callable, Union
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from vacuum_engineering import (
        CasimirArray, DynamicCasimirEffect, SqueezedVacuumResonator,
        MetamaterialCasimir, vacuum_energy_to_anec_flux
    )
    from anec_violation_analysis import coherent_state_anec_violation
    from custom_kernels import CustomKernelLibrary, create_standard_library
    from polymer_quantization import polymer_quantum_inequality_bound
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Some functionality may be limited.")

class VacuumANECIntegrator:
    """
    Integrates vacuum engineering sources with ANEC violation analysis.
    
    Provides unified framework for comparing laboratory vacuum sources
    with theoretical LQG predictions and quantum inequality constraints.
    """
    
    def __init__(self, hbar: float = 1.055e-34, c: float = 299792458):
        """
        Initialize vacuum-ANEC integrator.
        
        Args:
            hbar: Reduced Planck constant
            c: Speed of light
        """
        self.hbar = hbar
        self.c = c
        
        # Initialize vacuum engineering systems
        self.casimir = CasimirArray(temperature=4.0)
        self.dynamic = DynamicCasimirEffect()
        self.squeezed = SqueezedVacuumResonator()
        self.metamaterial = MetamaterialCasimir()
        
        # Load kernel library
        try:
            self.kernel_lib = create_standard_library()
        except:
            self.kernel_lib = None
            print("Warning: Could not load kernel library")
        
        print("Vacuum-ANEC Integrator initialized")
    
    def compute_vacuum_anec_violation(self, vacuum_type: str, 
                                    parameters: Dict, 
                                    temporal_profile: Dict) -> Dict:
        """
        Compute ANEC violation from specified vacuum source.
        
        Args:
            vacuum_type: Type of vacuum source ('casimir', 'dynamic', 'squeezed')
            parameters: Source-specific parameters
            temporal_profile: Temporal smearing parameters {'tau': scale, 'kernel': type}
            
        Returns:
            Dictionary with ANEC violation analysis results
        """
        results = {'vacuum_type': vacuum_type, 'parameters': parameters}
        
        # Extract energy density and volume based on vacuum type
        if vacuum_type == 'casimir':
            if 'layers' in parameters and 'spacing_list' in parameters:
                # Multi-layer configuration
                energy_density = self._compute_casimir_energy_density(
                    parameters['layers'], parameters['spacing_list'],
                    parameters.get('materials', ['Au'] * parameters['layers'])
                )
                volume = parameters.get('area', 1e-6) * sum(parameters['spacing_list'])
            else:
                # Single layer
                spacing = parameters.get('spacing', 100e-9)
                material = parameters.get('material', 'Au')
                pressure = self.casimir.casimir_pressure(spacing, material)
                area = parameters.get('area', 1e-6)
                volume = area * spacing
                energy_density = pressure * spacing / volume
                
        elif vacuum_type == 'dynamic':
            circuit_freq = parameters.get('circuit_frequency', 10e9)
            drive_freq = parameters.get('drive_frequency', 20e9)
            drive_amp = parameters.get('drive_amplitude', 0.1)
            quality_factor = parameters.get('quality_factor', 1000)
            volume = parameters.get('volume', 1e-9)
            
            self.dynamic.f0 = circuit_freq
            self.dynamic.drive_amp = drive_amp
            energy_density = self.dynamic.negative_energy_density(drive_freq, volume, quality_factor)
            
        elif vacuum_type == 'squeezed':
            resonator_freq = parameters.get('frequency', 1e14)
            squeezing_param = parameters.get('squeezing', 1.0)
            volume = parameters.get('volume', 1e-12)
            
            self.squeezed.omega_res = 2 * np.pi * resonator_freq
            self.squeezed.xi = squeezing_param
            energy_density = self.squeezed.squeezed_energy_density(volume)
            
        else:
            raise ValueError(f"Unknown vacuum type: {vacuum_type}")
        
        results['energy_density'] = energy_density
        results['volume'] = volume
        results['total_energy'] = energy_density * volume
        
        # Apply temporal smearing for ANEC violation
        tau = temporal_profile.get('tau', 1e-6)
        kernel_type = temporal_profile.get('kernel', 'gaussian')
        
        # Define smearing kernel
        if kernel_type == 'gaussian':
            def kernel(t, tau_scale):
                return np.exp(-t**2 / (2*tau_scale**2)) / np.sqrt(2*np.pi*tau_scale**2)
        elif kernel_type == 'lorentzian':
            def kernel(t, tau_scale):
                return (tau_scale/np.pi) / (t**2 + tau_scale**2)
        elif kernel_type == 'exponential':
            def kernel(t, tau_scale):
                return np.exp(-abs(t)/tau_scale) / (2*tau_scale)
        else:
            # Default to gaussian
            def kernel(t, tau_scale):
                return np.exp(-t**2 / (2*tau_scale**2)) / np.sqrt(2*np.pi*tau_scale**2)
        
        # Compute ANEC violation flux
        anec_flux = vacuum_energy_to_anec_flux(energy_density, volume, tau, kernel)
        results['anec_flux'] = anec_flux
        results['temporal_scale'] = tau
        
        # Compare with quantum inequality bounds
        try:
            qi_bound = polymer_quantum_inequality_bound(tau, mu=1e-35)
            results['qi_bound'] = qi_bound
            results['qi_violation_ratio'] = abs(anec_flux) / qi_bound if qi_bound > 0 else np.inf
        except:
            results['qi_bound'] = None
            results['qi_violation_ratio'] = None
        
        return results
    
    def _compute_casimir_energy_density(self, layers: int, spacing_list: List[float], 
                                      materials: List[str]) -> float:
        """Helper function to compute multi-layer Casimir energy density."""
        from vacuum_engineering import MATERIAL_DATABASE
        
        perm_list = [MATERIAL_DATABASE[mat]['permittivity'] for mat in materials]
        mu_list = [MATERIAL_DATABASE[mat]['permeability'] for mat in materials]
        
        pressure = self.casimir.stack_pressure(layers, spacing_list, perm_list, mu_list)
        
        # Convert pressure to energy density
        total_thickness = sum(spacing_list)
        return pressure / total_thickness  # Simplified conversion
    
    def optimize_vacuum_for_anec(self, vacuum_type: str, 
                               target_flux: float = 1e-25,
                               constraints: Optional[Dict] = None) -> Dict:
        """
        Optimize vacuum source parameters to maximize ANEC violation.
        
        Args:
            vacuum_type: Type of vacuum source to optimize
            target_flux: Target ANEC violation flux (W)
            constraints: Dictionary of parameter constraints
            
        Returns:
            Optimization results with best parameters and achieved flux
        """
        constraints = constraints or {}
        
        def objective(params):
            """Objective function: minimize |achieved_flux - target_flux|"""
            try:
                # Convert parameter array to dictionary based on vacuum type
                param_dict = self._params_array_to_dict(vacuum_type, params, constraints)
                
                # Compute ANEC violation
                temporal_profile = {'tau': 1e-6, 'kernel': 'gaussian'}
                result = self.compute_vacuum_anec_violation(vacuum_type, param_dict, temporal_profile)
                
                achieved_flux = abs(result['anec_flux'])
                error = abs(achieved_flux - target_flux)
                
                # Add penalty for unfeasible configurations
                if not self._check_feasibility(vacuum_type, param_dict):
                    error *= 1000
                
                return error
                
            except Exception as e:
                return 1e10  # Large penalty for invalid configurations
        
        # Define parameter bounds based on vacuum type
        bounds = self._get_parameter_bounds(vacuum_type, constraints)
        
        # Run optimization
        result = differential_evolution(objective, bounds, seed=42, maxiter=200)
        
        if result.success:
            # Convert optimal parameters back to dictionary
            optimal_params = self._params_array_to_dict(vacuum_type, result.x, constraints)
            
            # Compute final result with optimal parameters
            temporal_profile = {'tau': 1e-6, 'kernel': 'gaussian'}
            final_result = self.compute_vacuum_anec_violation(vacuum_type, optimal_params, temporal_profile)
            
            optimization_result = {
                'success': True,
                'optimal_parameters': optimal_params,
                'achieved_flux': final_result['anec_flux'],
                'target_flux': target_flux,
                'error': result.fun,
                'full_analysis': final_result
            }
        else:
            optimization_result = {
                'success': False,
                'error': 'Optimization failed',
                'message': result.message
            }
        
        return optimization_result
    
    def _params_array_to_dict(self, vacuum_type: str, params: np.ndarray, 
                            constraints: Dict) -> Dict:
        """Convert parameter array to dictionary for specific vacuum type."""
        param_dict = {}
        
        if vacuum_type == 'casimir':
            # Parameters: [spacing, area, n_layers, material_index]
            param_dict['spacing'] = params[0]
            param_dict['area'] = params[1] 
            param_dict['layers'] = int(np.round(params[2]))
            
            # Material selection
            materials = constraints.get('materials', ['Au', 'SiO2', 'metamaterial'])
            mat_idx = int(np.clip(np.round(params[3]), 0, len(materials)-1))
            param_dict['material'] = materials[mat_idx]
            
            # Create layer configuration
            param_dict['spacing_list'] = [param_dict['spacing']] * param_dict['layers']
            param_dict['materials'] = [param_dict['material']] * param_dict['layers']
            
        elif vacuum_type == 'dynamic':
            # Parameters: [circuit_freq, drive_freq, drive_amp, quality_factor, volume]
            param_dict['circuit_frequency'] = params[0]
            param_dict['drive_frequency'] = params[1]
            param_dict['drive_amplitude'] = params[2]
            param_dict['quality_factor'] = params[3]
            param_dict['volume'] = params[4]
            
        elif vacuum_type == 'squeezed':
            # Parameters: [frequency, squeezing, volume]
            param_dict['frequency'] = params[0]
            param_dict['squeezing'] = params[1]
            param_dict['volume'] = params[2]
        
        return param_dict
    
    def _get_parameter_bounds(self, vacuum_type: str, constraints: Dict) -> List[Tuple]:
        """Get parameter bounds for optimization based on vacuum type."""
        if vacuum_type == 'casimir':
            spacing_range = constraints.get('spacing_range', (10e-9, 1e-6))
            area_range = constraints.get('area_range', (1e-8, 1e-4))
            layer_range = constraints.get('layer_range', (1, 50))
            material_range = (0, len(constraints.get('materials', ['Au', 'SiO2', 'metamaterial']))-1)
            
            return [spacing_range, area_range, layer_range, material_range]
            
        elif vacuum_type == 'dynamic':
            freq_range = constraints.get('circuit_freq_range', (1e9, 1e12))
            drive_freq_range = constraints.get('drive_freq_range', (2e9, 2e12))
            amp_range = constraints.get('drive_amp_range', (0.01, 0.5))
            q_range = constraints.get('quality_range', (100, 100000))
            vol_range = constraints.get('volume_range', (1e-12, 1e-6))
            
            return [freq_range, drive_freq_range, amp_range, q_range, vol_range]
            
        elif vacuum_type == 'squeezed':
            freq_range = constraints.get('freq_range', (1e12, 1e15))
            squeeze_range = constraints.get('squeeze_range', (0.1, 5.0))
            vol_range = constraints.get('volume_range', (1e-15, 1e-9))
            
            return [freq_range, squeeze_range, vol_range]
        
        return []
    
    def _check_feasibility(self, vacuum_type: str, params: Dict) -> bool:
        """Check if parameter configuration is physically feasible."""
        if vacuum_type == 'casimir':
            spacing = params.get('spacing', 0)
            layers = params.get('layers', 0)
            return spacing > 1e-9 and spacing < 1e-3 and layers > 0 and layers < 100
            
        elif vacuum_type == 'dynamic':
            circuit_freq = params.get('circuit_frequency', 0)
            drive_freq = params.get('drive_frequency', 0)
            drive_amp = params.get('drive_amplitude', 0)
            
            # Drive frequency should be near 2×circuit frequency for resonance
            freq_ratio = drive_freq / circuit_freq if circuit_freq > 0 else 0
            return (1.5 < freq_ratio < 3.0 and 
                   0.01 < drive_amp < 0.5 and
                   circuit_freq < 1e12)
            
        elif vacuum_type == 'squeezed':
            squeezing = params.get('squeezing', 0)
            frequency = params.get('frequency', 0)
            return 0.1 < squeezing < 5.0 and frequency > 1e12
        
        return True
    
    def compare_with_lqg_predictions(self, vacuum_results: Dict) -> Dict:
        """
        Compare vacuum engineering results with LQG coherent state predictions.
        
        Args:
            vacuum_results: Results from vacuum ANEC violation analysis
            
        Returns:
            Comparison analysis including ratios and feasibility assessment
        """
        comparison = {'vacuum_results': vacuum_results}
        
        # Attempt to compute LQG coherent state ANEC violation for comparison
        try:
            # Use parameters similar to vacuum system
            lqg_params = {
                'polymer_scale': 1e-35,  # Planck scale
                'coherent_alpha': 1.0,
                'temporal_scale': vacuum_results.get('temporal_scale', 1e-6)
            }
            
            # This would require implementation in anec_violation_analysis module
            # For now, use simplified estimate
            lqg_flux_estimate = self._estimate_lqg_flux(lqg_params)
            
            comparison['lqg_flux_estimate'] = lqg_flux_estimate
            comparison['vacuum_to_lqg_ratio'] = (
                abs(vacuum_results['anec_flux']) / abs(lqg_flux_estimate) 
                if lqg_flux_estimate != 0 else np.inf
            )
            
        except Exception as e:
            comparison['lqg_comparison_error'] = str(e)
            comparison['lqg_flux_estimate'] = None
            comparison['vacuum_to_lqg_ratio'] = None
        
        # Add feasibility assessment
        vacuum_flux = abs(vacuum_results['anec_flux'])
        target_flux = 1e-25  # W
        
        comparison['target_achievement'] = vacuum_flux / target_flux
        comparison['orders_of_magnitude_gap'] = np.log10(target_flux / vacuum_flux) if vacuum_flux > 0 else np.inf
        
        # Assessment categories
        if comparison['target_achievement'] > 0.1:
            comparison['feasibility'] = 'high'
        elif comparison['target_achievement'] > 0.01:
            comparison['feasibility'] = 'moderate'
        else:
            comparison['feasibility'] = 'low'
        
        return comparison
    
    def _estimate_lqg_flux(self, params: Dict) -> float:
        """Simplified estimate of LQG coherent state ANEC violation flux."""
        # This is a placeholder - would need full LQG implementation
        polymer_scale = params['polymer_scale']
        alpha = params['coherent_alpha']
        tau = params['temporal_scale']
        
        # Dimensional analysis estimate
        flux_estimate = (self.hbar * self.c / polymer_scale**2) * alpha**2 * tau
        
        return -flux_estimate  # Negative for ANEC violation

def run_comprehensive_vacuum_anec_analysis():
    """
    Run comprehensive analysis comparing all vacuum sources with ANEC targets.
    """
    print("Comprehensive Vacuum-ANEC Analysis")
    print("=" * 50)
    
    integrator = VacuumANECIntegrator()
    
    # Define test configurations for each vacuum type
    test_configs = {
        'casimir': {
            'parameters': {
                'spacing': 100e-9,  # 100 nm
                'area': 1e-6,       # 1 mm²
                'layers': 10,
                'material': 'Au'
            },
            'temporal_profile': {'tau': 1e-6, 'kernel': 'gaussian'}
        },
        'dynamic': {
            'parameters': {
                'circuit_frequency': 10e9,   # 10 GHz
                'drive_frequency': 20e9,     # 20 GHz  
                'drive_amplitude': 0.2,
                'quality_factor': 10000,
                'volume': 1e-9              # 1 mm³
            },
            'temporal_profile': {'tau': 1e-6, 'kernel': 'gaussian'}
        },
        'squeezed': {
            'parameters': {
                'frequency': 1e14,          # 100 THz (infrared)
                'squeezing': 2.0,
                'volume': 1e-12             # 1 μm³
            },
            'temporal_profile': {'tau': 1e-6, 'kernel': 'gaussian'}
        }
    }
    
    results = {}
    
    # Analyze each vacuum type
    for vacuum_type, config in test_configs.items():
        print(f"\nAnalyzing {vacuum_type} vacuum source...")
        
        # Compute ANEC violation
        vacuum_result = integrator.compute_vacuum_anec_violation(
            vacuum_type, config['parameters'], config['temporal_profile']
        )
        
        # Compare with LQG predictions
        comparison = integrator.compare_with_lqg_predictions(vacuum_result)
        
        # Optimize for target flux
        print(f"  Optimizing {vacuum_type} configuration...")
        optimization = integrator.optimize_vacuum_for_anec(vacuum_type, target_flux=1e-25)
        
        results[vacuum_type] = {
            'baseline_analysis': vacuum_result,
            'lqg_comparison': comparison,
            'optimization': optimization
        }
        
        # Print summary
        print(f"  Baseline flux: {vacuum_result['anec_flux']:.2e} W")
        print(f"  Target achievement: {comparison['target_achievement']:.2e}")
        print(f"  Feasibility: {comparison['feasibility']}")
        
        if optimization['success']:
            print(f"  Optimized flux: {optimization['achieved_flux']:.2e} W")
            print(f"  Optimization improvement: {abs(optimization['achieved_flux']/vacuum_result['anec_flux']):.2f}×")
    
    # Overall comparison
    print(f"\nOverall Comparison:")
    print("-" * 30)
    
    best_baseline = max(results.keys(), 
                       key=lambda k: abs(results[k]['baseline_analysis']['anec_flux']))
    best_optimized = max(results.keys(),
                        key=lambda k: abs(results[k]['optimization']['achieved_flux']) 
                        if results[k]['optimization']['success'] else 0)
    
    print(f"Best baseline approach: {best_baseline}")
    print(f"Best optimized approach: {best_optimized}")
    
    best_flux = abs(results[best_optimized]['optimization']['achieved_flux'])
    target_flux = 1e-25
    gap = target_flux / best_flux if best_flux > 0 else np.inf
    
    print(f"Best achieved flux: {best_flux:.2e} W")
    print(f"Gap to target: {gap:.2e}× improvement needed")
    
    return results

if __name__ == "__main__":
    try:
        results = run_comprehensive_vacuum_anec_analysis()
        print("\n✓ Comprehensive analysis completed!")
        
    except Exception as e:
        print(f"\n✗ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
