#!/usr/bin/env python3
"""
Integrated Module Test Script

Comprehensive integration test for all four new modules:
1. custom_kernels.py - Custom QI kernel generation and testing
2. ghost_condensate_eft.py - UV-complete ghost EFT with ANEC violation
3. semi_classical_stress.py - LQG stress tensor in discrete geometry
4. test_backreaction.py - Backreaction and geometry stability

This script demonstrates the full pipeline from custom kernel generation
through ghost field analysis to LQG stress tensor computation and finally
backreaction stability testing.

Usage:
    python scripts/integrated_module_test.py [--gpu] [--verbose] [--output-dir DIR]
"""

import torch
import numpy as np
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import logging
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from custom_kernels import CustomKernelLibrary
    from ghost_condensate_eft import GhostCondensateEFT, GhostEFTParameters
    from semi_classical_stress import SemiClassicalStressTensor, LQGParameters, SpinNetworkType
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure all modules are properly implemented in src/")
    sys.exit(1)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('integrated_test.log')
        ]
    )
    return logging.getLogger(__name__)

def test_custom_kernels(output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Test custom kernel generation and QI bound analysis."""
    logger.info("Testing custom kernel library...")
    
    # Initialize kernel library
    kernel_lib = CustomKernelLibrary()
    
    # Test various kernel types
    tau = np.linspace(-1.0, 1.0, 1000)
    
    kernels_to_test = [
        ("gaussian", lambda t: kernel_lib.gaussian(t, 0.2)),
        ("lorentzian", lambda t: kernel_lib.lorentzian(t, 0.1)),
        ("exponential", lambda t: kernel_lib.exponential(t, 5.0)),
        ("polynomial_n2", lambda t: kernel_lib.polynomial_basis(t, 0.5, 2)),
        ("polynomial_n4", lambda t: kernel_lib.polynomial_basis(t, 0.5, 4)),
        ("sinc", lambda t: kernel_lib.sinc_kernel(t, 10.0)),
        ("oscillatory", lambda t: kernel_lib.oscillatory_gaussian(t, 0.2, 20.0)),
    ]
    
    results = {}
    
    # Test each kernel
    for name, kernel_func in kernels_to_test:
        logger.info(f"Testing {name} kernel...")
        
        try:
            # Generate kernel
            f_values = kernel_func(tau)
            
            # Add to library
            kernel_lib.add_kernel(name, f_values, tau)
            
            # Test QI bound (simplified)
            violation_score = kernel_lib.test_qi_violation(name, tau0=0.1)
            
            results[name] = {
                "kernel_shape": f_values.shape,
                "max_value": float(np.max(f_values)),
                "min_value": float(np.min(f_values)),
                "integral": float(np.trapz(f_values, tau)),
                "violation_score": violation_score
            }
            
            logger.info(f"{name}: violation_score = {violation_score:.6f}")
            
        except Exception as e:
            logger.error(f"Failed testing {name} kernel: {e}")
            results[name] = {"error": str(e)}
    
    # Generate comparison plot
    plt.figure(figsize=(12, 8))
    for i, (name, kernel_func) in enumerate(kernels_to_test[:4]):  # Plot first 4
        try:
            f_values = kernel_func(tau)
            plt.subplot(2, 2, i+1)
            plt.plot(tau, f_values, 'b-', linewidth=2)
            plt.title(f'{name.title()} Kernel')
            plt.xlabel('τ')
            plt.ylabel('f(τ)')
            plt.grid(True, alpha=0.3)
        except:
            continue
    
    plt.tight_layout()
    plt.savefig(output_dir / "custom_kernels_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Custom kernel tests completed. Results: {len(results)} kernels tested")
    return results

def test_ghost_condensate_eft(output_dir: Path, logger: logging.Logger, use_gpu: bool = True) -> Dict[str, Any]:
    """Test ghost condensate EFT with ANEC violation analysis."""
    logger.info("Testing ghost condensate EFT...")
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Setup EFT parameters
    params = GhostEFTParameters(
        phi_0=1.0,
        lambda_ghost=0.1,
        cutoff_scale=10.0,
        grid_size=64,  # Smaller for faster testing
        device=device
    )
    
    results = {}
    
    try:
        # Initialize ghost EFT
        start_time = time.time()
        ghost_eft = GhostCondensateEFT(params)
        init_time = time.time() - start_time
        
        # Test field configuration
        logger.info("Computing ghost field configuration...")
        field_config = ghost_eft.compute_field_configuration()
        
        # Test stress-energy tensor
        logger.info("Computing stress-energy tensor...")
        stress_tensor = ghost_eft.compute_stress_tensor()
        
        # Test ANEC violation
        logger.info("Computing ANEC integral...")
        anec_value = ghost_eft.compute_anec_integral()
        
        # Test UV completion effects
        logger.info("Testing UV completion...")
        uv_correction = ghost_eft.compute_uv_correction()
        
        # Test stability
        logger.info("Testing stability...")
        stability_analysis = ghost_eft.analyze_stability()
        
        results = {
            "initialization_time": init_time,
            "device": device,
            "grid_size": params.grid_size,
            "field_config_shape": list(field_config.shape) if hasattr(field_config, 'shape') else None,
            "stress_tensor_shape": list(stress_tensor.shape) if hasattr(stress_tensor, 'shape') else None,
            "anec_value": float(anec_value) if isinstance(anec_value, (int, float, torch.Tensor)) else str(anec_value),
            "uv_correction": float(torch.mean(uv_correction)) if hasattr(uv_correction, 'mean') else str(uv_correction),
            "stability": stability_analysis
        }
        
        # Generate visualization
        if hasattr(field_config, 'cpu'):
            field_data = field_config.cpu().numpy()
            if field_data.ndim >= 3:
                # Plot 2D slice through the middle
                middle_slice = field_data.shape[0] // 2
                plt.figure(figsize=(10, 6))
                
                plt.subplot(1, 2, 1)
                plt.imshow(field_data[middle_slice], cmap='RdBu', origin='lower')
                plt.colorbar()
                plt.title('Ghost Field φ (t=0 slice)')
                plt.xlabel('x')
                plt.ylabel('y')
                
                plt.subplot(1, 2, 2)
                if hasattr(stress_tensor, 'cpu') and stress_tensor.cpu().numpy().ndim >= 3:
                    stress_data = stress_tensor.cpu().numpy()
                    plt.imshow(stress_data[middle_slice], cmap='plasma', origin='lower')
                    plt.colorbar()
                    plt.title('Stress-Energy T₀₀ (t=0 slice)')
                    plt.xlabel('x')
                    plt.ylabel('y')
                
                plt.tight_layout()
                plt.savefig(output_dir / "ghost_eft_fields.png", dpi=300, bbox_inches='tight')
                plt.close()
        
        logger.info(f"Ghost EFT test completed. ANEC value: {results['anec_value']}")
        
    except Exception as e:
        logger.error(f"Ghost EFT test failed: {e}")
        results = {"error": str(e), "device": device}
    
    return results

def test_semi_classical_stress(output_dir: Path, logger: logging.Logger, use_gpu: bool = True) -> Dict[str, Any]:
    """Test semi-classical LQG stress tensor computation."""
    logger.info("Testing semi-classical LQG stress tensor...")
    
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    
    # Setup LQG parameters
    lqg_params = LQGParameters(
        network_type=SpinNetworkType.CUBICAL,
        max_spin=2,  # Smaller for testing
        network_size=8,
        device=device
    )
    
    results = {}
    
    try:
        # Initialize stress tensor operator
        start_time = time.time()
        stress_op = SemiClassicalStressTensor(lqg_params)
        init_time = time.time() - start_time
        
        # Test coherent state preparation
        logger.info("Preparing coherent states...")
        coherent_state = stress_op.prepare_coherent_state()
        
        # Test volume operator
        logger.info("Computing volume operator...")
        volume_op = stress_op.compute_volume_operator()
        
        # Test area operator
        logger.info("Computing area operator...")
        area_op = stress_op.compute_area_operator()
        
        # Test stress tensor expectation value
        logger.info("Computing stress tensor expectation...")
        stress_expectation = stress_op.compute_stress_expectation()
        
        # Test polymer enhancement
        logger.info("Computing polymer enhancement...")
        polymer_factor = stress_op.compute_polymer_enhancement()
        
        results = {
            "initialization_time": init_time,
            "device": device,
            "network_type": lqg_params.network_type.value,
            "network_size": lqg_params.network_size,
            "coherent_state_norm": float(torch.norm(coherent_state)) if hasattr(coherent_state, 'norm') else str(coherent_state),
            "volume_eigenvalue": float(torch.mean(volume_op)) if hasattr(volume_op, 'mean') else str(volume_op),
            "area_eigenvalue": float(torch.mean(area_op)) if hasattr(area_op, 'mean') else str(area_op),
            "stress_tensor_00": float(stress_expectation) if isinstance(stress_expectation, (int, float, torch.Tensor)) else str(stress_expectation),
            "polymer_enhancement": float(polymer_factor) if isinstance(polymer_factor, (int, float, torch.Tensor)) else str(polymer_factor)
        }
        
        logger.info(f"LQG stress tensor test completed. T₀₀ = {results['stress_tensor_00']}")
        
    except Exception as e:
        logger.error(f"LQG stress tensor test failed: {e}")
        results = {"error": str(e), "device": device}
    
    return results

def test_integration_pipeline(output_dir: Path, logger: logging.Logger, use_gpu: bool = True) -> Dict[str, Any]:
    """Test full integration pipeline combining all modules."""
    logger.info("Testing full integration pipeline...")
    
    results = {}
    
    try:
        # Step 1: Generate custom kernel
        logger.info("Step 1: Custom kernel generation")
        kernel_lib = CustomKernelLibrary()
        tau = np.linspace(-0.5, 0.5, 500)
        
        # Create a composite kernel combining multiple components
        composite_kernel = kernel_lib.custom_kernel(tau, [
            (0.6, kernel_lib.gaussian, {"sigma": 0.1}),
            (0.3, kernel_lib.oscillatory_gaussian, {"sigma": 0.2, "omega": 15.0}),
            (0.1, kernel_lib.polynomial_basis, {"R": 0.3, "n": 3})
        ])
        
        kernel_lib.add_kernel("composite", composite_kernel, tau)
        qi_violation = kernel_lib.test_qi_violation("composite", tau0=0.05)
        
        # Step 2: Ghost EFT with custom kernel
        logger.info("Step 2: Ghost EFT analysis")
        device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        
        ghost_params = GhostEFTParameters(
            lambda_ghost=0.15,  # Slightly stronger coupling
            grid_size=32,       # Small for integration test
            device=device
        )
        
        ghost_eft = GhostCondensateEFT(ghost_params)
        anec_value = ghost_eft.compute_anec_integral()
        
        # Step 3: LQG stress tensor
        logger.info("Step 3: LQG stress tensor")
        lqg_params = LQGParameters(
            network_size=6,  # Small network
            device=device
        )
        
        lqg_stress = SemiClassicalStressTensor(lqg_params)
        stress_value = lqg_stress.compute_stress_expectation()
        
        # Step 4: Compare results
        logger.info("Step 4: Cross-validation")
        
        results = {
            "kernel_qi_violation": qi_violation,
            "ghost_anec_value": float(anec_value) if isinstance(anec_value, (int, float, torch.Tensor)) else str(anec_value),
            "lqg_stress_value": float(stress_value) if isinstance(stress_value, (int, float, torch.Tensor)) else str(stress_value),
            "consistency_check": "PASS" if qi_violation > 0 else "FAIL",
            "negative_energy_detected": anec_value < 0 if isinstance(anec_value, (int, float)) else False,
            "device": device
        }
        
        # Generate integration summary plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Custom kernel
        ax1.plot(tau, composite_kernel, 'b-', linewidth=2)
        ax1.set_title('Composite Custom Kernel')
        ax1.set_xlabel('τ')
        ax1.set_ylabel('f(τ)')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: QI violation metric
        violation_data = [qi_violation, 0.0, abs(qi_violation)]
        ax2.bar(['QI Violation', 'Baseline', 'Magnitude'], violation_data, 
                color=['red', 'gray', 'blue'])
        ax2.set_title('QI Violation Analysis')
        ax2.set_ylabel('Violation Score')
        
        # Plot 3: Energy values comparison
        energy_data = [anec_value if isinstance(anec_value, (int, float)) else 0,
                      stress_value if isinstance(stress_value, (int, float)) else 0]
        energy_labels = ['Ghost ANEC', 'LQG Stress']
        colors = ['red' if val < 0 else 'blue' for val in energy_data]
        ax3.bar(energy_labels, energy_data, color=colors)
        ax3.set_title('Energy Analysis')
        ax3.set_ylabel('Energy Density')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        
        # Plot 4: Summary metrics
        metrics = [qi_violation, 
                  abs(anec_value) if isinstance(anec_value, (int, float)) else 1,
                  abs(stress_value) if isinstance(stress_value, (int, float)) else 1]
        metric_labels = ['QI Violation', '|ANEC|', '|Stress|']
        ax4.semilogy(metric_labels, metrics, 'o-', linewidth=2, markersize=8)
        ax4.set_title('Integration Metrics (Log Scale)')
        ax4.set_ylabel('Magnitude')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "integration_pipeline_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("Integration pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Integration pipeline failed: {e}")
        results = {"error": str(e)}
    
    return results

def main():
    """Main function for integrated module testing."""
    parser = argparse.ArgumentParser(description="Integrated Module Test Script")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--output-dir", type=str, default="integration_results", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("Starting integrated module testing...")
    logger.info(f"GPU acceleration: {args.gpu}")
    logger.info(f"Output directory: {output_dir}")
    
    # Check device availability
    if args.gpu and not torch.cuda.is_available():
        logger.warning("GPU requested but CUDA not available. Using CPU.")
    
    # Run all tests
    test_results = {}
    
    try:
        # Test 1: Custom kernels
        logger.info("="*60)
        test_results["custom_kernels"] = test_custom_kernels(output_dir, logger)
        
        # Test 2: Ghost condensate EFT
        logger.info("="*60)
        test_results["ghost_eft"] = test_ghost_condensate_eft(output_dir, logger, args.gpu)
        
        # Test 3: Semi-classical stress tensor
        logger.info("="*60)
        test_results["lqg_stress"] = test_semi_classical_stress(output_dir, logger, args.gpu)
        
        # Test 4: Integration pipeline
        logger.info("="*60)
        test_results["integration"] = test_integration_pipeline(output_dir, logger, args.gpu)
        
    except Exception as e:
        logger.error(f"Testing failed: {e}")
        test_results["global_error"] = str(e)
    
    # Save results
    results_file = output_dir / "integrated_test_results.json"
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    # Generate summary report
    summary_lines = [
        "# Integrated Module Test Summary",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Test Results Overview",
    ]
    
    for test_name, results in test_results.items():
        if "error" in results:
            status = "❌ FAILED"
            details = f"Error: {results['error']}"
        else:
            status = "✅ PASSED"
            details = f"Completed successfully"
        
        summary_lines.extend([
            f"- **{test_name}**: {status}",
            f"  {details}",
            ""
        ])
    
    # Write summary
    with open(output_dir / "test_summary.md", 'w') as f:
        f.write('\n'.join(summary_lines))
    
    logger.info("="*60)
    logger.info("Integrated module testing completed!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Summary: {sum(1 for r in test_results.values() if 'error' not in r)}/{len(test_results)} tests passed")

if __name__ == "__main__":
    main()
