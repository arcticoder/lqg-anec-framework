"""
Comprehensive Integration Test Script

Integrates and tests all four new modules for advanced QI circumvention:
1. Custom Kernels Library (QI bound analysis)
2. Ghost-Condensate EFT (ANEC violation)
3. Semi-Classical LQG Stress Tensor (discrete geometry)
4. Backreaction & Geometry Stability (Einstein equations)

Runs systematic parameter sweeps, cross-validates results between approaches,
and generates comprehensive analysis reports.
"""

import torch
import numpy as np
import json
import argparse
import sys
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
from dataclasses import dataclass

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from custom_kernels import CustomKernelLibrary, generate_optimal_qi_kernel
    from ghost_condensate_eft import GhostCondensateEFT, GhostEFTParameters, scan_ghost_eft_parameters
    from semi_classical_stress import SemiClassicalStressTensor, LQGParameters, SpinNetworkType
except ImportError as e:
    print(f"Warning: Could not import some modules: {e}")
    print("Running in partial mode...")

# Import backreaction analyzer
try:
    from test_backreaction import GeometryStabilityAnalyzer, BackreactionParameters
except ImportError as e:
    print(f"Warning: Could not import backreaction module: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('comprehensive_integration.log')
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class IntegrationParameters:
    """Parameters for comprehensive integration analysis."""
    # Analysis scope
    run_kernel_sweep: bool = True
    run_ghost_eft: bool = True
    run_lqg_analysis: bool = True
    run_backreaction: bool = True
    
    # Parameter ranges
    kernel_count: int = 20
    ghost_eft_samples: int = 15
    lqg_network_sizes: List[int] = None
    backreaction_grid_sizes: List[int] = None
    
    # Comparison settings
    cross_validate: bool = True
    generate_plots: bool = True
    detailed_analysis: bool = True
    
    # Performance settings
    device: str = "cuda"
    max_memory_gb: float = 8.0
    parallel_jobs: int = 4
    
    # Output settings
    output_dir: str = "results_comprehensive"
    save_intermediate: bool = True
    compression_level: int = 6
    
    def __post_init__(self):
        if self.lqg_network_sizes is None:
            self.lqg_network_sizes = [8, 12, 16]
        if self.backreaction_grid_sizes is None:
            self.backreaction_grid_sizes = [24, 32, 48]


class ComprehensiveIntegrationAnalyzer:
    """
    Comprehensive analyzer integrating all four new modules.
    
    Performs systematic parameter sweeps, cross-validation between approaches,
    and generates unified analysis reports for QI circumvention research.
    """
    
    def __init__(self, params: IntegrationParameters):
        """Initialize comprehensive integration analyzer."""
        self.params = params
        self.device = torch.device(params.device if torch.cuda.is_available() else "cpu")
        
        # Results storage
        self.results = {
            'kernel_analysis': {},
            'ghost_eft_analysis': {},
            'lqg_analysis': {},
            'backreaction_analysis': {},
            'cross_validation': {},
            'unified_findings': {}
        }
        
        # Performance tracking
        self.timing_data = {}
        self.memory_usage = {}
        
        # Create output directory
        os.makedirs(params.output_dir, exist_ok=True)
        
        logger.info(f"Comprehensive integration analyzer initialized on {self.device}")
        logger.info(f"Analysis scope: Kernels={params.run_kernel_sweep}, Ghost={params.run_ghost_eft}, "
                   f"LQG={params.run_lqg_analysis}, Backreaction={params.run_backreaction}")
    
    def run_kernel_sweep_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive custom kernel analysis for QI bound circumvention.
        """
        if not self.params.run_kernel_sweep:
            logger.info("Skipping kernel sweep analysis")
            return {}
        
        logger.info("Starting custom kernel sweep analysis...")
        start_time = time.time()
        
        try:
            # Initialize kernel library
            kernel_lib = CustomKernelLibrary()
            
            # Test different kernel families
            kernel_families = [
                ('gaussian', {'sigma': np.linspace(0.1, 2.0, 5)}),
                ('lorentzian', {'gamma': np.linspace(0.05, 1.0, 5)}),
                ('exponential', {'lambda_param': np.linspace(0.5, 3.0, 5)}),
                ('polynomial_basis', {'R': [1.0, 2.0], 'n': [2, 4, 6]}),
                ('fourier_basis', {'omega_max': [5.0, 10.0], 'N_modes': [3, 5]}),
                ('wavelet_basis', {'scale': [0.5, 1.0, 2.0], 'translation': [0.0]})
            ]
            
            kernel_results = []
            
            for family_name, param_dict in kernel_families:
                if not hasattr(kernel_lib, family_name):
                    logger.warning(f"Kernel family {family_name} not available")
                    continue
                
                logger.info(f"Testing {family_name} kernels...")
                
                # Generate parameter combinations
                if family_name == 'polynomial_basis':
                    for R in param_dict['R']:
                        for n in param_dict['n']:
                            kernel_result = self._test_single_kernel(kernel_lib, family_name, {'R': R, 'n': n})
                            kernel_results.append(kernel_result)
                
                elif family_name in ['fourier_basis', 'wavelet_basis']:
                    # Handle multi-parameter cases
                    param_keys = list(param_dict.keys())
                    if len(param_keys) == 2:
                        key1, key2 = param_keys
                        for val1 in param_dict[key1]:
                            for val2 in param_dict[key2]:
                                params = {key1: val1, key2: val2}
                                kernel_result = self._test_single_kernel(kernel_lib, family_name, params)
                                kernel_results.append(kernel_result)
                
                else:
                    # Single parameter families
                    param_name = list(param_dict.keys())[0]
                    for param_value in param_dict[param_name]:
                        params = {param_name: param_value}
                        kernel_result = self._test_single_kernel(kernel_lib, family_name, params)
                        kernel_results.append(kernel_result)
            
            # Find best performing kernels
            kernel_results.sort(key=lambda x: x.get('qi_violation_magnitude', 0), reverse=True)
            
            analysis_time = time.time() - start_time
            self.timing_data['kernel_analysis'] = analysis_time
            
            kernel_analysis = {
                'total_kernels_tested': len(kernel_results),
                'best_kernels': kernel_results[:10],  # Top 10
                'kernel_families_tested': [family[0] for family in kernel_families],
                'analysis_time': analysis_time,
                'best_qi_violation': kernel_results[0].get('qi_violation_magnitude', 0) if kernel_results else 0,
                'violation_statistics': self._compute_kernel_statistics(kernel_results)
            }
            
            logger.info(f"Kernel sweep completed: {len(kernel_results)} kernels tested in {analysis_time:.2f}s")
            logger.info(f"Best QI violation: {kernel_analysis['best_qi_violation']:.2e}")
            
            return kernel_analysis
            
        except Exception as e:
            logger.error(f"Kernel sweep analysis failed: {e}")
            return {'error': str(e)}
    
    def _test_single_kernel(self, kernel_lib: Any, family_name: str, params: Dict) -> Dict[str, Any]:
        """Test a single kernel configuration."""
        try:
            # Time range for testing
            tau_range = 10.0
            tau_points = 1000
            tau = np.linspace(-tau_range/2, tau_range/2, tau_points)
            
            # Generate kernel
            if family_name == 'gaussian':
                kernel_func = lambda t: kernel_lib.gaussian(t, params['sigma'])
            elif family_name == 'lorentzian':
                kernel_func = lambda t: kernel_lib.lorentzian(t, params['gamma'])
            elif family_name == 'exponential':
                kernel_func = lambda t: kernel_lib.exponential(t, params['lambda_param'])
            elif family_name == 'polynomial_basis':
                kernel_func = lambda t: kernel_lib.polynomial_basis(t, params['R'], params['n'])
            elif family_name == 'fourier_basis':
                kernel_func = lambda t: kernel_lib.fourier_basis(t, params['omega_max'], params['N_modes'])
            elif family_name == 'wavelet_basis':
                kernel_func = lambda t: kernel_lib.wavelet_basis(t, params['scale'], params['translation'])
            else:
                return {'error': f'Unknown kernel family: {family_name}'}
            
            # Evaluate kernel
            f_values = kernel_func(tau)
            
            # Test QI bound violation (simplified)
            # Use characteristic time scale
            tau0 = tau_range / 10
            qi_bound = -3.0 / (32 * np.pi**2 * tau0**4)  # Simplified Ford-Roman bound
            
            # Compute effective violation metric
            negative_values = f_values[f_values < 0]
            if len(negative_values) > 0:
                violation_magnitude = -np.sum(negative_values) * (tau[1] - tau[0])
            else:
                violation_magnitude = 0.0
            
            # Kernel properties
            kernel_norm = np.trapz(np.abs(f_values), tau)
            kernel_support = np.sum(np.abs(f_values) > 1e-10) * (tau[1] - tau[0])
            
            return {
                'kernel_family': family_name,
                'parameters': params,
                'qi_violation_magnitude': violation_magnitude,
                'kernel_norm': kernel_norm,
                'kernel_support': kernel_support,
                'min_value': np.min(f_values),
                'max_value': np.max(f_values),
                'qi_bound_reference': qi_bound
            }
            
        except Exception as e:
            return {
                'kernel_family': family_name,
                'parameters': params,
                'error': str(e)
            }
    
    def _compute_kernel_statistics(self, kernel_results: List[Dict]) -> Dict[str, Any]:
        """Compute statistical summary of kernel performance."""
        if not kernel_results:
            return {}
        
        violations = [r.get('qi_violation_magnitude', 0) for r in kernel_results if 'error' not in r]
        norms = [r.get('kernel_norm', 0) for r in kernel_results if 'error' not in r]
        supports = [r.get('kernel_support', 0) for r in kernel_results if 'error' not in r]
        
        return {
            'violation_mean': np.mean(violations),
            'violation_std': np.std(violations),
            'violation_max': np.max(violations) if violations else 0,
            'violation_min': np.min(violations) if violations else 0,
            'norm_mean': np.mean(norms),
            'support_mean': np.mean(supports),
            'success_rate': len(violations) / len(kernel_results),
            'num_successful': len(violations)
        }
    
    def run_ghost_eft_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive ghost-condensate EFT analysis.
        """
        if not self.params.run_ghost_eft:
            logger.info("Skipping ghost EFT analysis")
            return {}
        
        logger.info("Starting ghost-condensate EFT analysis...")
        start_time = time.time()
        
        try:
            # Parameter ranges for scanning
            parameter_ranges = {
                'lambda_ghost': (0.01, 0.5),
                'cutoff_scale': (2.0, 20.0),
                'phi_0': (0.5, 2.0),
                'higher_deriv_coeff': (0.001, 0.1)
            }
            
            # Run parameter scan
            scan_results = scan_ghost_eft_parameters(
                parameter_ranges=parameter_ranges,
                num_samples=self.params.ghost_eft_samples,
                device=self.params.device
            )
            
            # Analyze best configurations
            best_configs = sorted(scan_results, 
                                key=lambda x: abs(x.get('min_anec_violation', 0)), 
                                reverse=True)[:5]
            
            # Detailed analysis of top configuration
            if best_configs:
                top_config = best_configs[0]['parameter_config']
                
                # Create EFT with best parameters
                best_params = GhostEFTParameters(
                    lambda_ghost=top_config.get('lambda_ghost', 0.1),
                    cutoff_scale=top_config.get('cutoff_scale', 10.0),
                    phi_0=top_config.get('phi_0', 1.0),
                    higher_deriv_coeff=top_config.get('higher_deriv_coeff', 0.01),
                    grid_size=48,  # Smaller for integration test
                    device=self.params.device
                )
                
                ghost_eft = GhostCondensateEFT(best_params)
                detailed_report = ghost_eft.generate_anec_violation_report()
            else:
                detailed_report = {}
            
            analysis_time = time.time() - start_time
            self.timing_data['ghost_eft_analysis'] = analysis_time
            
            ghost_analysis = {
                'parameter_scan_results': len(scan_results),
                'best_configurations': best_configs,
                'detailed_analysis': detailed_report,
                'analysis_time': analysis_time,
                'max_anec_violation': best_configs[0].get('min_anec_violation', 0) if best_configs else 0,
                'scan_statistics': self._compute_ghost_statistics(scan_results)
            }
            
            logger.info(f"Ghost EFT analysis completed in {analysis_time:.2f}s")
            logger.info(f"Best ANEC violation: {ghost_analysis['max_anec_violation']:.2e}")
            
            return ghost_analysis
            
        except Exception as e:
            logger.error(f"Ghost EFT analysis failed: {e}")
            return {'error': str(e)}
    
    def _compute_ghost_statistics(self, scan_results: List[Dict]) -> Dict[str, Any]:
        """Compute statistical summary of ghost EFT scan."""
        if not scan_results:
            return {}
        
        violations = [abs(r.get('min_anec_violation', 0)) for r in scan_results]
        negative_fractions = [r.get('negative_energy_fraction', 0) for r in scan_results]
        
        return {
            'anec_violation_mean': np.mean(violations),
            'anec_violation_std': np.std(violations),
            'anec_violation_max': np.max(violations) if violations else 0,
            'negative_energy_mean': np.mean(negative_fractions),
            'configurations_tested': len(scan_results)
        }
    
    def run_lqg_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive LQG stress tensor analysis.
        """
        if not self.params.run_lqg_analysis:
            logger.info("Skipping LQG analysis")
            return {}
        
        logger.info("Starting LQG stress tensor analysis...")
        start_time = time.time()
        
        try:
            lqg_results = []
            
            # Test different network configurations
            network_types = [SpinNetworkType.CUBICAL, SpinNetworkType.TETRAHEDRAL]
            
            for network_type in network_types:
                for network_size in self.params.lqg_network_sizes:
                    logger.info(f"Testing {network_type.value} network, size {network_size}")
                    
                    lqg_params = LQGParameters(
                        network_type=network_type,
                        network_size=network_size,
                        max_spin=6.0,
                        coherent_scale=200.0,
                        polymer_scale=1e-25,  # Planck-scale polymer effects
                        device=self.params.device
                    )
                    
                    # Initialize LQG stress tensor
                    lqg_stress = SemiClassicalStressTensor(lqg_params)
                    
                    # Generate field configuration
                    field_config = torch.randn(lqg_stress.n_nodes, device=lqg_stress.device)
                    
                    # Generate analysis report
                    lqg_report = lqg_stress.generate_lqg_stress_report(field_config)
                    
                    # Add configuration info
                    lqg_report['configuration'] = {
                        'network_type': network_type.value,
                        'network_size': network_size,
                        'actual_nodes': lqg_stress.n_nodes,
                        'actual_edges': lqg_stress.n_edges
                    }
                    
                    lqg_results.append(lqg_report)
            
            # Find best ANEC violations
            best_lqg = sorted(lqg_results, 
                            key=lambda x: abs(x['anec_violation']['integral_value']), 
                            reverse=True)[:3]
            
            analysis_time = time.time() - start_time
            self.timing_data['lqg_analysis'] = analysis_time
            
            lqg_analysis = {
                'configurations_tested': len(lqg_results),
                'best_configurations': best_lqg,
                'all_results': lqg_results,
                'analysis_time': analysis_time,
                'max_anec_violation': best_lqg[0]['anec_violation']['integral_value'] if best_lqg else 0,
                'network_statistics': self._compute_lqg_statistics(lqg_results)
            }
            
            logger.info(f"LQG analysis completed in {analysis_time:.2f}s")
            logger.info(f"Best LQG ANEC violation: {lqg_analysis['max_anec_violation']:.2e}")
            
            return lqg_analysis
            
        except Exception as e:
            logger.error(f"LQG analysis failed: {e}")
            return {'error': str(e)}
    
    def _compute_lqg_statistics(self, lqg_results: List[Dict]) -> Dict[str, Any]:
        """Compute statistical summary of LQG analysis."""
        if not lqg_results:
            return {}
        
        anec_violations = [abs(r['anec_violation']['integral_value']) for r in lqg_results]
        violation_fractions = [r['anec_violation']['violation_fraction'] for r in lqg_results]
        node_counts = [r['configuration']['actual_nodes'] for r in lqg_results]
        
        return {
            'anec_violation_mean': np.mean(anec_violations),
            'anec_violation_std': np.std(anec_violations),
            'anec_violation_max': np.max(anec_violations),
            'violation_fraction_mean': np.mean(violation_fractions),
            'avg_network_size': np.mean(node_counts)
        }
    
    def run_backreaction_analysis(self) -> Dict[str, Any]:
        """
        Run comprehensive backreaction and geometry stability analysis.
        """
        if not self.params.run_backreaction:
            logger.info("Skipping backreaction analysis")
            return {}
        
        logger.info("Starting backreaction analysis...")
        start_time = time.time()
        
        try:
            backreaction_results = []
            
            # Test different grid resolutions
            for grid_size in self.params.backreaction_grid_sizes:
                logger.info(f"Testing backreaction with grid size {grid_size}")
                
                backreaction_params = BackreactionParameters(
                    spatial_grid_size=grid_size,
                    integration_steps=200,  # Reduced for integration test
                    temporal_range=0.5,
                    spatial_range=5.0,
                    perturbation_amplitude=1e-6,
                    stability_threshold=1e-4,
                    device=self.params.device,
                    plot_results=False  # Skip plots for integration
                )
                
                # Initialize analyzer
                analyzer = GeometryStabilityAnalyzer(backreaction_params)
                
                # Run analysis
                report = analyzer.generate_comprehensive_report()
                
                # Add configuration info
                report['configuration'] = {
                    'grid_size': grid_size,
                    'integration_steps': backreaction_params.integration_steps,
                    'temporal_range': backreaction_params.temporal_range
                }
                
                backreaction_results.append(report)
            
            # Analyze stability across configurations
            stable_configs = [r for r in backreaction_results if r['critical_findings']['geometry_stable']]
            unstable_configs = [r for r in backreaction_results if not r['critical_findings']['geometry_stable']]
            
            analysis_time = time.time() - start_time
            self.timing_data['backreaction_analysis'] = analysis_time
            
            backreaction_analysis = {
                'configurations_tested': len(backreaction_results),
                'stable_configurations': len(stable_configs),
                'unstable_configurations': len(unstable_configs),
                'all_results': backreaction_results,
                'analysis_time': analysis_time,
                'stability_statistics': self._compute_backreaction_statistics(backreaction_results)
            }
            
            logger.info(f"Backreaction analysis completed in {analysis_time:.2f}s")
            logger.info(f"Stable/Unstable configurations: {len(stable_configs)}/{len(unstable_configs)}")
            
            return backreaction_analysis
            
        except Exception as e:
            logger.error(f"Backreaction analysis failed: {e}")
            return {'error': str(e)}
    
    def _compute_backreaction_statistics(self, backreaction_results: List[Dict]) -> Dict[str, Any]:
        """Compute statistical summary of backreaction analysis."""
        if not backreaction_results:
            return {}
        
        growth_rates = []
        neg_energy_fractions = []
        
        for result in backreaction_results:
            if 'stability_analysis' in result:
                growth_rates.extend(result['stability_analysis']['growth_rates'])
            if 'negative_energy_bounds' in result:
                neg_energy_fractions.append(result['negative_energy_bounds']['negative_energy_fraction'])
        
        return {
            'max_growth_rate': np.max(growth_rates) if growth_rates else 0,
            'mean_growth_rate': np.mean(growth_rates) if growth_rates else 0,
            'std_growth_rate': np.std(growth_rates) if growth_rates else 0,
            'mean_negative_energy_fraction': np.mean(neg_energy_fractions) if neg_energy_fractions else 0,
            'stability_fraction': len([r for r in backreaction_results if r['critical_findings']['geometry_stable']]) / len(backreaction_results)
        }
    
    def run_cross_validation(self) -> Dict[str, Any]:
        """
        Cross-validate results between different approaches.
        """
        if not self.params.cross_validate:
            logger.info("Skipping cross-validation")
            return {}
        
        logger.info("Running cross-validation between approaches...")
        
        cross_validation = {
            'anec_violation_comparison': {},
            'energy_scale_consistency': {},
            'stability_correlation': {},
            'method_agreement': {}
        }
        
        # Compare ANEC violations between Ghost EFT and LQG
        if 'ghost_eft_analysis' in self.results and 'lqg_analysis' in self.results:
            ghost_violations = []
            lqg_violations = []
            
            # Extract violations
            if 'best_configurations' in self.results['ghost_eft_analysis']:
                for config in self.results['ghost_eft_analysis']['best_configurations']:
                    ghost_violations.append(abs(config.get('min_anec_violation', 0)))
            
            if 'best_configurations' in self.results['lqg_analysis']:
                for config in self.results['lqg_analysis']['best_configurations']:
                    lqg_violations.append(abs(config['anec_violation']['integral_value']))
            
            if ghost_violations and lqg_violations:
                cross_validation['anec_violation_comparison'] = {
                    'ghost_eft_mean': np.mean(ghost_violations),
                    'lqg_mean': np.mean(lqg_violations),
                    'ratio': np.mean(ghost_violations) / np.mean(lqg_violations),
                    'correlation_possible': len(ghost_violations) > 1 and len(lqg_violations) > 1
                }
        
        # Energy scale consistency checks
        energy_scales = {}
        if 'kernel_analysis' in self.results and 'best_kernels' in self.results['kernel_analysis']:
            kernel_scales = [k.get('qi_violation_magnitude', 0) for k in self.results['kernel_analysis']['best_kernels']]
            energy_scales['kernel_qi'] = np.mean(kernel_scales)
        
        if 'ghost_eft_analysis' in self.results and 'max_anec_violation' in self.results['ghost_eft_analysis']:
            energy_scales['ghost_anec'] = self.results['ghost_eft_analysis']['max_anec_violation']
        
        if 'lqg_analysis' in self.results and 'max_anec_violation' in self.results['lqg_analysis']:
            energy_scales['lqg_anec'] = self.results['lqg_analysis']['max_anec_violation']
        
        cross_validation['energy_scale_consistency'] = energy_scales
        
        # Stability correlation
        if 'backreaction_analysis' in self.results:
            stability_data = self.results['backreaction_analysis'].get('stability_statistics', {})
            stability_fraction = stability_data.get('stability_fraction', 0)
            
            cross_validation['stability_correlation'] = {
                'geometry_stability_fraction': stability_fraction,
                'negative_energy_tolerance': stability_data.get('mean_negative_energy_fraction', 0),
                'consistent_with_violations': stability_fraction > 0.5  # Heuristic
            }
        
        logger.info("Cross-validation completed")
        return cross_validation
    
    def generate_unified_findings(self) -> Dict[str, Any]:
        """
        Generate unified findings across all approaches.
        """
        logger.info("Generating unified findings...")
        
        unified_findings = {
            'qi_circumvention_evidence': {},
            'anec_violation_mechanisms': {},
            'stability_constraints': {},
            'optimal_configurations': {},
            'theoretical_implications': {}
        }
        
        # QI circumvention evidence
        qi_evidence = {}
        if 'kernel_analysis' in self.results:
            best_violation = self.results['kernel_analysis'].get('best_qi_violation', 0)
            qi_evidence['custom_kernel_violations'] = best_violation
            qi_evidence['kernel_success_rate'] = self.results['kernel_analysis'].get('violation_statistics', {}).get('success_rate', 0)
        
        unified_findings['qi_circumvention_evidence'] = qi_evidence
        
        # ANEC violation mechanisms
        anec_mechanisms = {}
        if 'ghost_eft_analysis' in self.results:
            anec_mechanisms['ghost_condensate'] = {
                'max_violation': self.results['ghost_eft_analysis'].get('max_anec_violation', 0),
                'mechanism': 'negative kinetic energy from ghost field'
            }
        
        if 'lqg_analysis' in self.results:
            anec_mechanisms['lqg_discreteness'] = {
                'max_violation': self.results['lqg_analysis'].get('max_anec_violation', 0),
                'mechanism': 'polymer-enhanced stress tensor on discrete geometry'
            }
        
        unified_findings['anec_violation_mechanisms'] = anec_mechanisms
        
        # Stability constraints
        if 'backreaction_analysis' in self.results:
            stability_stats = self.results['backreaction_analysis'].get('stability_statistics', {})
            unified_findings['stability_constraints'] = {
                'geometry_stability_threshold': stability_stats.get('max_growth_rate', 0),
                'negative_energy_tolerance': stability_stats.get('mean_negative_energy_fraction', 0),
                'stable_configuration_fraction': stability_stats.get('stability_fraction', 0)
            }
        
        # Optimal configurations
        optimal_configs = {}
        
        # Best kernel
        if 'kernel_analysis' in self.results and 'best_kernels' in self.results['kernel_analysis']:
            best_kernel = self.results['kernel_analysis']['best_kernels'][0]
            optimal_configs['best_qi_kernel'] = {
                'family': best_kernel.get('kernel_family', 'unknown'),
                'parameters': best_kernel.get('parameters', {}),
                'violation': best_kernel.get('qi_violation_magnitude', 0)
            }
        
        # Best ghost EFT
        if ('ghost_eft_analysis' in self.results and 
            'best_configurations' in self.results['ghost_eft_analysis']):
            best_ghost = self.results['ghost_eft_analysis']['best_configurations'][0]
            optimal_configs['best_ghost_eft'] = {
                'parameters': best_ghost.get('parameter_config', {}),
                'anec_violation': best_ghost.get('min_anec_violation', 0)
            }
        
        # Best LQG
        if ('lqg_analysis' in self.results and 
            'best_configurations' in self.results['lqg_analysis']):
            best_lqg = self.results['lqg_analysis']['best_configurations'][0]
            optimal_configs['best_lqg_network'] = {
                'configuration': best_lqg.get('configuration', {}),
                'anec_violation': best_lqg['anec_violation']['integral_value']
            }
        
        unified_findings['optimal_configurations'] = optimal_configs
        
        # Theoretical implications
        implications = []
        
        # Check for consistent violations
        if (qi_evidence.get('custom_kernel_violations', 0) > 1e-10 and
            anec_mechanisms.get('ghost_condensate', {}).get('max_violation', 0) < -1e-10):
            implications.append("Custom kernels and ghost EFT show consistent QI/ANEC violations")
        
        if anec_mechanisms.get('lqg_discreteness', {}).get('max_violation', 0) < -1e-10:
            implications.append("LQG discreteness enables ANEC violation via polymer enhancement")
        
        if (self.results.get('backreaction_analysis', {}).get('stability_statistics', {}).get('stability_fraction', 0) > 0.5):
            implications.append("Negative energy sources can maintain geometric stability")
        
        unified_findings['theoretical_implications'] = implications
        
        logger.info(f"Unified findings generated with {len(implications)} key implications")
        return unified_findings
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        Run complete comprehensive analysis integrating all modules.
        """
        logger.info("Starting comprehensive integration analysis...")
        total_start_time = time.time()
        
        # Run individual analyses
        self.results['kernel_analysis'] = self.run_kernel_sweep_analysis()
        self.results['ghost_eft_analysis'] = self.run_ghost_eft_analysis()
        self.results['lqg_analysis'] = self.run_lqg_analysis()
        self.results['backreaction_analysis'] = self.run_backreaction_analysis()
        
        # Cross-validation
        self.results['cross_validation'] = self.run_cross_validation()
        
        # Unified findings
        self.results['unified_findings'] = self.generate_unified_findings()
        
        # Performance summary
        total_time = time.time() - total_start_time
        self.timing_data['total_analysis'] = total_time
        
        # Add metadata
        self.results['analysis_metadata'] = {
            'total_computation_time': total_time,
            'timing_breakdown': self.timing_data,
            'device_used': str(self.device),
            'parameters': {
                'kernel_count': self.params.kernel_count,
                'ghost_eft_samples': self.params.ghost_eft_samples,
                'lqg_network_sizes': self.params.lqg_network_sizes,
                'backreaction_grid_sizes': self.params.backreaction_grid_sizes
            }
        }
        
        # Add memory usage if available
        if torch.cuda.is_available():
            self.results['analysis_metadata']['gpu_memory_peak'] = torch.cuda.max_memory_allocated() / 1e9
            self.results['analysis_metadata']['gpu_memory_current'] = torch.cuda.memory_allocated() / 1e9
        
        logger.info(f"Comprehensive analysis completed in {total_time:.2f} seconds")
        
        return self.results
    
    def save_results(self, output_dir: Optional[str] = None):
        """Save comprehensive analysis results."""
        if output_dir is None:
            output_dir = self.params.output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        results_file = os.path.join(output_dir, "comprehensive_integration_results.json")
        with open(results_file, 'w') as f:
            # Convert tensors for JSON serialization
            json_results = self._convert_tensors_to_json(self.results)
            json.dump(json_results, f, indent=2)
        
        # Save summary report
        summary_file = os.path.join(output_dir, "integration_summary.md")
        self._save_markdown_summary(summary_file)
        
        logger.info(f"Results saved to {output_dir}/")
    
    def _convert_tensors_to_json(self, obj):
        """Convert tensors to JSON-serializable format."""
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: self._convert_tensors_to_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_tensors_to_json(item) for item in obj]
        else:
            return obj
    
    def _save_markdown_summary(self, summary_file: str):
        """Save markdown summary of results."""
        with open(summary_file, 'w') as f:
            f.write("# Comprehensive QI Circumvention Integration Analysis\n\n")
            
            # Analysis overview
            f.write("## Analysis Overview\n\n")
            f.write(f"- **Total computation time**: {self.results['analysis_metadata']['total_computation_time']:.2f} seconds\n")
            f.write(f"- **Device used**: {self.results['analysis_metadata']['device_used']}\n")
            
            # Key findings
            f.write("\n## Key Findings\n\n")
            
            # Kernel analysis
            if 'kernel_analysis' in self.results and 'best_qi_violation' in self.results['kernel_analysis']:
                f.write(f"- **Best QI violation (kernels)**: {self.results['kernel_analysis']['best_qi_violation']:.2e}\n")
            
            # Ghost EFT
            if 'ghost_eft_analysis' in self.results and 'max_anec_violation' in self.results['ghost_eft_analysis']:
                f.write(f"- **Best ANEC violation (Ghost EFT)**: {self.results['ghost_eft_analysis']['max_anec_violation']:.2e}\n")
            
            # LQG
            if 'lqg_analysis' in self.results and 'max_anec_violation' in self.results['lqg_analysis']:
                f.write(f"- **Best ANEC violation (LQG)**: {self.results['lqg_analysis']['max_anec_violation']:.2e}\n")
            
            # Stability
            if 'backreaction_analysis' in self.results:
                stability_fraction = self.results['backreaction_analysis'].get('stability_statistics', {}).get('stability_fraction', 0)
                f.write(f"- **Geometry stability fraction**: {stability_fraction:.3f}\n")
            
            # Theoretical implications
            f.write("\n## Theoretical Implications\n\n")
            if 'unified_findings' in self.results and 'theoretical_implications' in self.results['unified_findings']:
                for implication in self.results['unified_findings']['theoretical_implications']:
                    f.write(f"- {implication}\n")
            
            # Optimal configurations
            f.write("\n## Optimal Configurations\n\n")
            if 'unified_findings' in self.results and 'optimal_configurations' in self.results['unified_findings']:
                optimal = self.results['unified_findings']['optimal_configurations']
                
                if 'best_qi_kernel' in optimal:
                    kernel = optimal['best_qi_kernel']
                    f.write(f"- **Best QI kernel**: {kernel['family']} with violation {kernel['violation']:.2e}\n")
                
                if 'best_ghost_eft' in optimal:
                    ghost = optimal['best_ghost_eft']
                    f.write(f"- **Best Ghost EFT**: ANEC violation {ghost['anec_violation']:.2e}\n")
                
                if 'best_lqg_network' in optimal:
                    lqg = optimal['best_lqg_network']
                    f.write(f"- **Best LQG network**: ANEC violation {lqg['anec_violation']:.2e}\n")


def main():
    """Main CLI interface for comprehensive integration analysis."""
    parser = argparse.ArgumentParser(description="Comprehensive QI Circumvention Integration Analysis")
    
    parser.add_argument("--kernel-count", type=int, default=20, help="Number of kernels to test")
    parser.add_argument("--ghost-samples", type=int, default=15, help="Ghost EFT parameter samples")
    parser.add_argument("--lqg-sizes", nargs='+', type=int, default=[8, 12], help="LQG network sizes")
    parser.add_argument("--backreaction-grids", nargs='+', type=int, default=[24, 32], help="Backreaction grid sizes")
    
    parser.add_argument("--skip-kernels", action="store_true", help="Skip kernel analysis")
    parser.add_argument("--skip-ghost", action="store_true", help="Skip ghost EFT analysis")
    parser.add_argument("--skip-lqg", action="store_true", help="Skip LQG analysis")
    parser.add_argument("--skip-backreaction", action="store_true", help="Skip backreaction analysis")
    
    parser.add_argument("--device", type=str, default="cuda", help="Computation device")
    parser.add_argument("--output-dir", type=str, default="results_comprehensive", help="Output directory")
    parser.add_argument("--no-cross-validate", action="store_true", help="Skip cross-validation")
    
    args = parser.parse_args()
    
    # Create integration parameters
    params = IntegrationParameters(
        run_kernel_sweep=not args.skip_kernels,
        run_ghost_eft=not args.skip_ghost,
        run_lqg_analysis=not args.skip_lqg,
        run_backreaction=not args.skip_backreaction,
        kernel_count=args.kernel_count,
        ghost_eft_samples=args.ghost_samples,
        lqg_network_sizes=args.lqg_sizes,
        backreaction_grid_sizes=args.backreaction_grids,
        cross_validate=not args.no_cross_validate,
        device=args.device,
        output_dir=args.output_dir
    )
    
    logger.info("Starting comprehensive QI circumvention integration analysis...")
    logger.info(f"Analysis scope: Kernels={params.run_kernel_sweep}, Ghost={params.run_ghost_eft}, "
               f"LQG={params.run_lqg_analysis}, Backreaction={params.run_backreaction}")
    
    try:
        # Initialize analyzer
        analyzer = ComprehensiveIntegrationAnalyzer(params)
        
        # Run comprehensive analysis
        results = analyzer.run_comprehensive_analysis()
        
        # Save results
        analyzer.save_results()
        
        # Print summary
        print("\n" + "="*80)
        print("COMPREHENSIVE QI CIRCUMVENTION INTEGRATION ANALYSIS SUMMARY")
        print("="*80)
        
        metadata = results.get('analysis_metadata', {})
        print(f"Total computation time: {metadata.get('total_computation_time', 0):.2f} seconds")
        print(f"Device used: {metadata.get('device_used', 'unknown')}")
        
        if torch.cuda.is_available() and 'gpu_memory_peak' in metadata:
            print(f"Peak GPU memory: {metadata['gpu_memory_peak']:.2f} GB")
        
        # Key results
        print("\nKey Results:")
        
        if 'kernel_analysis' in results:
            kernel_violation = results['kernel_analysis'].get('best_qi_violation', 0)
            print(f"  Best QI kernel violation: {kernel_violation:.2e}")
        
        if 'ghost_eft_analysis' in results:
            ghost_violation = results['ghost_eft_analysis'].get('max_anec_violation', 0)
            print(f"  Best Ghost EFT ANEC violation: {ghost_violation:.2e}")
        
        if 'lqg_analysis' in results:
            lqg_violation = results['lqg_analysis'].get('max_anec_violation', 0)
            print(f"  Best LQG ANEC violation: {lqg_violation:.2e}")
        
        if 'backreaction_analysis' in results:
            stability_stats = results['backreaction_analysis'].get('stability_statistics', {})
            stability_fraction = stability_stats.get('stability_fraction', 0)
            print(f"  Geometry stability fraction: {stability_fraction:.3f}")
        
        # Theoretical implications
        if 'unified_findings' in results and 'theoretical_implications' in results['unified_findings']:
            implications = results['unified_findings']['theoretical_implications']
            if implications:
                print(f"\nTheoretical Implications ({len(implications)}):")
                for i, implication in enumerate(implications, 1):
                    print(f"  {i}. {implication}")
        
        print(f"\nDetailed results saved to: {args.output_dir}/")
        print("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Comprehensive analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
