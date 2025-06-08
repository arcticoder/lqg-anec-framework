#!/usr/bin/env python3
"""
Custom Kernel Sweep Script

Systematic sweep through custom kernel parameter space to identify
optimal configurations for quantum inequality bound violation.

Features:
- Automated kernel parameter optimization
- Batch testing with GPU acceleration  
- Best performer identification and logging
- Comprehensive visualization and analysis
- CLI-driven automation with file output

Usage:
    python scripts/sweep_custom_kernels.py [--num-kernels N] [--gpu] [--output-dir DIR]
"""

import numpy as np
import sys
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
import logging
import matplotlib.pyplot as plt
from itertools import product
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

try:
    from custom_kernels import CustomKernelLibrary
except ImportError as e:
    print(f"Error importing custom_kernels: {e}")
    print("Please ensure custom_kernels.py is properly implemented in src/")
    sys.exit(1)

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('kernel_sweep.log')
        ]
    )
    return logging.getLogger(__name__)

def generate_parameter_grid() -> List[Dict[str, Any]]:
    """Generate systematic parameter grid for kernel testing."""
    
    # Parameter ranges for different kernel types
    parameter_grids = {
        'gaussian': {
            'sigma': np.logspace(-2, 0, 20)  # 0.01 to 1.0
        },
        'lorentzian': {
            'gamma': np.logspace(-2, 0, 20)  # 0.01 to 1.0
        },
        'exponential': {
            'lambda_param': np.logspace(0, 2, 20)  # 1.0 to 100
        },
        'polynomial_basis': {
            'R': np.linspace(0.1, 1.0, 10),
            'n': [1, 2, 3, 4, 5, 6]
        },
        'sinc_kernel': {
            'cutoff': np.logspace(0, 2, 20)  # 1.0 to 100
        },
        'oscillatory_gaussian': {
            'sigma': np.logspace(-2, 0, 15),  # 0.01 to 1.0
            'omega': np.logspace(0, 2, 15)    # 1.0 to 100
        }
    }
    
    all_configs = []
    
    # Generate all combinations for each kernel type
    for kernel_type, param_dict in parameter_grids.items():
        param_names = list(param_dict.keys())
        param_values = list(param_dict.values())
        
        for combination in product(*param_values):
            config = {
                'kernel_type': kernel_type,
                'params': dict(zip(param_names, combination))
            }
            all_configs.append(config)
    
    return all_configs

def test_single_kernel(config: Dict[str, Any], tau_grid: np.ndarray, tau0_values: List[float]) -> Dict[str, Any]:
    """Test a single kernel configuration."""
    
    kernel_lib = CustomKernelLibrary()
    kernel_type = config['kernel_type']
    params = config['params']
    
    try:
        # Generate kernel
        if kernel_type == 'gaussian':
            f_values = kernel_lib.gaussian(tau_grid, **params)
        elif kernel_type == 'lorentzian':
            f_values = kernel_lib.lorentzian(tau_grid, **params)
        elif kernel_type == 'exponential':
            f_values = kernel_lib.exponential(tau_grid, **params)
        elif kernel_type == 'polynomial_basis':
            f_values = kernel_lib.polynomial_basis(tau_grid, **params)
        elif kernel_type == 'sinc_kernel':
            f_values = kernel_lib.sinc_kernel(tau_grid, **params)
        elif kernel_type == 'oscillatory_gaussian':
            f_values = kernel_lib.oscillatory_gaussian(tau_grid, **params)
        else:
            return {'error': f'Unknown kernel type: {kernel_type}'}
        
        # Basic validation
        if np.any(np.isnan(f_values)) or np.any(np.isinf(f_values)):
            return {'error': 'Invalid kernel values (NaN or Inf)'}
        
        if np.all(f_values == 0):
            return {'error': 'Zero kernel'}
        
        # Add to library and test QI violations
        kernel_name = f"{kernel_type}_{hash(str(params)) % 10000}"
        kernel_lib.add_kernel(kernel_name, f_values, tau_grid)
        
        # Test multiple tau0 values
        violation_scores = []
        for tau0 in tau0_values:
            try:
                score = kernel_lib.test_qi_violation(kernel_name, tau0=tau0)
                violation_scores.append(score)
            except:
                violation_scores.append(0.0)
        
        # Compute kernel statistics
        integral = np.trapz(f_values, tau_grid)
        max_val = np.max(f_values)
        min_val = np.min(f_values)
        std_val = np.std(f_values)
        
        # Compute kernel "quality" metrics
        compactness = np.sum(f_values > 0.1 * max_val) / len(f_values)
        smoothness = 1.0 / (1.0 + np.mean(np.abs(np.diff(f_values))))
        
        result = {
            'kernel_type': kernel_type,
            'params': params,
            'violation_scores': violation_scores,
            'max_violation': max(violation_scores),
            'mean_violation': np.mean(violation_scores),
            'integral': integral,
            'max_value': max_val,
            'min_value': min_val,
            'std_value': std_val,
            'compactness': compactness,
            'smoothness': smoothness,
            'kernel_quality': max(violation_scores) * smoothness * compactness,
            'success': True
        }
        
        return result
        
    except Exception as e:
        return {
            'kernel_type': kernel_type,
            'params': params,
            'error': str(e),
            'success': False
        }

def run_kernel_sweep(num_kernels: int, num_workers: int, output_dir: Path, logger: logging.Logger) -> Dict[str, Any]:
    """Run systematic kernel parameter sweep."""
    
    logger.info(f"Starting kernel sweep with {num_kernels} configurations...")
    
    # Setup tau grid and tau0 values for testing
    tau_grid = np.linspace(-1.0, 1.0, 1000)
    tau0_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    
    # Generate parameter configurations
    all_configs = generate_parameter_grid()
    
    # Limit to requested number
    if num_kernels < len(all_configs):
        # Sample evenly across parameter space
        indices = np.linspace(0, len(all_configs)-1, num_kernels, dtype=int)
        configs_to_test = [all_configs[i] for i in indices]
    else:
        configs_to_test = all_configs
    
    logger.info(f"Testing {len(configs_to_test)} kernel configurations")
    logger.info(f"Using {num_workers} worker processes")
    
    # Run parallel testing
    results = []
    successful_tests = 0
    failed_tests = 0
    
    start_time = time.time()
    
    if num_workers > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            # Submit all tasks
            future_to_config = {
                executor.submit(test_single_kernel, config, tau_grid, tau0_values): config 
                for config in configs_to_test
            }
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_config)):
                try:
                    result = future.result()
                    results.append(result)
                    
                    if result.get('success', False):
                        successful_tests += 1
                    else:
                        failed_tests += 1
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Completed {i+1}/{len(configs_to_test)} tests")
                        
                except Exception as e:
                    logger.error(f"Task failed: {e}")
                    failed_tests += 1
    else:
        # Sequential execution
        for i, config in enumerate(configs_to_test):
            result = test_single_kernel(config, tau_grid, tau0_values)
            results.append(result)
            
            if result.get('success', False):
                successful_tests += 1
            else:
                failed_tests += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"Completed {i+1}/{len(configs_to_test)} tests")
    
    total_time = time.time() - start_time
    
    logger.info(f"Sweep completed in {total_time:.2f} seconds")
    logger.info(f"Successful tests: {successful_tests}")
    logger.info(f"Failed tests: {failed_tests}")
    
    # Analyze results
    successful_results = [r for r in results if r.get('success', False)]
    
    if not successful_results:
        logger.error("No successful kernel tests!")
        return {'error': 'No successful tests', 'total_time': total_time}
    
    # Sort by various metrics
    by_max_violation = sorted(successful_results, key=lambda x: x['max_violation'], reverse=True)
    by_mean_violation = sorted(successful_results, key=lambda x: x['mean_violation'], reverse=True)
    by_quality = sorted(successful_results, key=lambda x: x['kernel_quality'], reverse=True)
    
    # Extract top performers
    top_10_max = by_max_violation[:10]
    top_10_mean = by_mean_violation[:10]
    top_10_quality = by_quality[:10]
    
    logger.info("Top 3 kernels by max violation:")
    for i, result in enumerate(top_10_max[:3]):
        logger.info(f"  {i+1}. {result['kernel_type']} - Max violation: {result['max_violation']:.6f}")
    
    # Generate summary statistics
    max_violations = [r['max_violation'] for r in successful_results]
    mean_violations = [r['mean_violation'] for r in successful_results]
    qualities = [r['kernel_quality'] for r in successful_results]
    
    summary_stats = {
        'total_tests': len(configs_to_test),
        'successful_tests': successful_tests,
        'failed_tests': failed_tests,
        'total_time': total_time,
        'tests_per_second': len(configs_to_test) / total_time,
        'max_violation_stats': {
            'max': max(max_violations),
            'min': min(max_violations),
            'mean': np.mean(max_violations),
            'std': np.std(max_violations),
            'median': np.median(max_violations)
        },
        'mean_violation_stats': {
            'max': max(mean_violations),
            'min': min(mean_violations),
            'mean': np.mean(mean_violations),
            'std': np.std(mean_violations),
            'median': np.median(mean_violations)
        },
        'quality_stats': {
            'max': max(qualities),
            'min': min(qualities),
            'mean': np.mean(qualities),
            'std': np.std(qualities),
            'median': np.median(qualities)
        }
    }
    
    # Save detailed results
    detailed_results = {
        'summary_stats': summary_stats,
        'top_performers': {
            'by_max_violation': top_10_max,
            'by_mean_violation': top_10_mean,
            'by_quality': top_10_quality
        },
        'all_results': successful_results
    }
    
    return detailed_results

def generate_visualizations(results: Dict[str, Any], output_dir: Path, logger: logging.Logger):
    """Generate comprehensive visualization of sweep results."""
    
    logger.info("Generating visualizations...")
    
    if 'error' in results:
        logger.error("Cannot generate visualizations - sweep failed")
        return
    
    successful_results = results['all_results']
    
    # Extract data for plotting
    kernel_types = [r['kernel_type'] for r in successful_results]
    max_violations = [r['max_violation'] for r in successful_results]
    mean_violations = [r['mean_violation'] for r in successful_results]
    qualities = [r['kernel_quality'] for r in successful_results]
    
    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: Max violation distribution
    axes[0, 0].hist(max_violations, bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Max QI Violation Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Max Violation Scores')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].axvline(np.mean(max_violations), color='blue', linestyle='--', 
                       label=f'Mean: {np.mean(max_violations):.4f}')
    axes[0, 0].legend()
    
    # Plot 2: Mean violation vs Max violation
    axes[0, 1].scatter(mean_violations, max_violations, alpha=0.6, s=20)
    axes[0, 1].set_xlabel('Mean QI Violation Score')
    axes[0, 1].set_ylabel('Max QI Violation Score')
    axes[0, 1].set_title('Mean vs Max Violation Correlation')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Kernel quality distribution
    axes[0, 2].hist(qualities, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0, 2].set_xlabel('Kernel Quality Score')
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].set_title('Distribution of Kernel Quality')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].axvline(np.mean(qualities), color='blue', linestyle='--',
                       label=f'Mean: {np.mean(qualities):.4f}')
    axes[0, 2].legend()
    
    # Plot 4: Performance by kernel type
    unique_types = list(set(kernel_types))
    type_max_violations = [np.mean([r['max_violation'] for r in successful_results if r['kernel_type'] == kt]) 
                          for kt in unique_types]
    
    axes[1, 0].bar(unique_types, type_max_violations, alpha=0.7, color='purple')
    axes[1, 0].set_xlabel('Kernel Type')
    axes[1, 0].set_ylabel('Mean Max Violation')
    axes[1, 0].set_title('Performance by Kernel Type')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Top 10 performers
    top_10 = sorted(successful_results, key=lambda x: x['max_violation'], reverse=True)[:10]
    top_labels = [f"{r['kernel_type'][:8]}..." for r in top_10]
    top_scores = [r['max_violation'] for r in top_10]
    
    axes[1, 1].barh(range(len(top_labels)), top_scores, alpha=0.7, color='orange')
    axes[1, 1].set_yticks(range(len(top_labels)))
    axes[1, 1].set_yticklabels(top_labels)
    axes[1, 1].set_xlabel('Max QI Violation Score')
    axes[1, 1].set_title('Top 10 Performers')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Quality vs Max violation
    axes[1, 2].scatter(qualities, max_violations, alpha=0.6, s=20, c='red')
    axes[1, 2].set_xlabel('Kernel Quality Score')
    axes[1, 2].set_ylabel('Max QI Violation Score')
    axes[1, 2].set_title('Quality vs Performance')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "kernel_sweep_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Generate top performer kernels visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    tau_grid = np.linspace(-1.0, 1.0, 1000)
    kernel_lib = CustomKernelLibrary()
    
    for i, result in enumerate(top_10[:6]):  # Show top 6
        kernel_type = result['kernel_type']
        params = result['params']
        
        try:
            # Regenerate kernel
            if kernel_type == 'gaussian':
                f_values = kernel_lib.gaussian(tau_grid, **params)
            elif kernel_type == 'lorentzian':
                f_values = kernel_lib.lorentzian(tau_grid, **params)
            elif kernel_type == 'exponential':
                f_values = kernel_lib.exponential(tau_grid, **params)
            elif kernel_type == 'polynomial_basis':
                f_values = kernel_lib.polynomial_basis(tau_grid, **params)
            elif kernel_type == 'sinc_kernel':
                f_values = kernel_lib.sinc_kernel(tau_grid, **params)
            elif kernel_type == 'oscillatory_gaussian':
                f_values = kernel_lib.oscillatory_gaussian(tau_grid, **params)
            else:
                continue
            
            axes[i].plot(tau_grid, f_values, 'b-', linewidth=2)
            axes[i].set_title(f'#{i+1}: {kernel_type}\nScore: {result["max_violation"]:.4f}')
            axes[i].set_xlabel('τ')
            axes[i].set_ylabel('f(τ)')
            axes[i].grid(True, alpha=0.3)
            
        except Exception as e:
            axes[i].text(0.5, 0.5, f'Error: {str(e)[:30]}...', 
                        transform=axes[i].transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / "top_performer_kernels.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info("Visualizations completed and saved")

def main():
    """Main function for kernel sweep."""
    parser = argparse.ArgumentParser(description="Custom Kernel Sweep Script")
    parser.add_argument("--num-kernels", type=int, default=1000,
                       help="Number of kernel configurations to test")
    parser.add_argument("--num-workers", type=int, default=4,
                       help="Number of worker processes")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument("--output-dir", type=str, default="kernel_sweep_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    logger.info("Starting custom kernel parameter sweep...")
    logger.info(f"Number of kernels: {args.num_kernels}")
    logger.info(f"Number of workers: {args.num_workers}")
    logger.info(f"Output directory: {output_dir}")
    
    # Run sweep
    try:
        results = run_kernel_sweep(args.num_kernels, args.num_workers, output_dir, logger)
        
        # Save results
        results_file = output_dir / "kernel_sweep_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Generate visualizations
        generate_visualizations(results, output_dir, logger)
        
        # Generate summary report
        if 'error' not in results:
            summary_lines = [
                "# Custom Kernel Sweep Summary",
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "## Overall Statistics",
                f"- Total configurations tested: {results['summary_stats']['total_tests']}",
                f"- Successful tests: {results['summary_stats']['successful_tests']}",
                f"- Failed tests: {results['summary_stats']['failed_tests']}",
                f"- Total execution time: {results['summary_stats']['total_time']:.2f} seconds",
                f"- Tests per second: {results['summary_stats']['tests_per_second']:.2f}",
                "",
                "## Top Performers",
                ""
            ]
            
            for i, result in enumerate(results['top_performers']['by_max_violation'][:5]):
                summary_lines.extend([
                    f"### {i+1}. {result['kernel_type'].title()}",
                    f"- Max violation score: {result['max_violation']:.6f}",
                    f"- Mean violation score: {result['mean_violation']:.6f}",
                    f"- Kernel quality: {result['kernel_quality']:.6f}",
                    f"- Parameters: {result['params']}",
                    ""
                ])
            
            # Write summary
            with open(output_dir / "sweep_summary.md", 'w') as f:
                f.write('\n'.join(summary_lines))
            
            logger.info("Kernel sweep completed successfully!")
            logger.info(f"Best performer: {results['top_performers']['by_max_violation'][0]['kernel_type']} "
                       f"(score: {results['top_performers']['by_max_violation'][0]['max_violation']:.6f})")
        else:
            logger.error("Kernel sweep failed!")
            
    except Exception as e:
        logger.error(f"Sweep execution failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
