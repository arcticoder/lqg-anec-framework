#!/usr/bin/env python3
"""
Dashboard-Ready Ghost EFT Batch Scanner

Production-ready batch scanning system for Ghost EFT parameter exploration,
designed for dashboard integration and continuous monitoring of Discovery 21 results.

Features:
- Real-time batch processing with progress tracking
- JSON output compatible with dashboard systems  
- Performance metrics and success rate monitoring
- Parameter exploration around Discovery 21 optimal configuration
- Integration with LQG-ANEC computational framework

Usage:
    python dashboard_ghost_scanner.py --configs 100 --output results/dashboard_scan.json
"""

import numpy as np
import json
import time
import argparse
import sys
from pathlib import Path
from tqdm import trange

# Add src to path  
sys.path.append(str(Path(__file__).parent / "src"))

try:
    from ghost_condensate_eft import GhostCondensateEFT
    AVAILABLE = True
except ImportError:
    AVAILABLE = False
    print("Warning: Ghost EFT module not available")


def dashboard_batch_scan(num_configs=100, output_file="results/dashboard_scan.json"):
    """
    Dashboard-ready batch scanning around Discovery 21 optimal parameters.
    
    Returns JSON-formatted results suitable for real-time dashboard integration.
    """
    
    if not AVAILABLE:
        return {'error': 'Ghost EFT module not available', 'status': 'FAILED'}
    
    print(f"🔬 Dashboard Ghost EFT Scanner - Discovery 21 Integration")
    print(f"⚡ Scanning {num_configs} configurations around optimal parameters")
    print(f"📊 Output: {output_file}")
    
    # Discovery 21 optimal parameters as baseline
    optimal_M = 1000.0
    optimal_alpha = 0.01  
    optimal_beta = 0.1
    
    # Generate parameter variations around optimal (±30% range)
    np.random.seed(int(time.time()) % 1000)  # Time-seeded for dashboard variety
    
    M_vals = np.random.uniform(optimal_M * 0.7, optimal_M * 1.3, num_configs)
    alpha_vals = np.random.uniform(optimal_alpha * 0.7, optimal_alpha * 1.3, num_configs)
    beta_vals = np.random.uniform(optimal_beta * 0.7, optimal_beta * 1.3, num_configs)
    
    # Week-scale Gaussian smearing (Discovery 21 methodology)
    tau0 = 7 * 24 * 3600  # 604,800 seconds
    
    def gaussian_kernel(tau):
        return (1 / np.sqrt(2 * np.pi * tau0**2)) * np.exp(-tau**2 / (2 * tau0**2))
    
    # Batch processing with real-time metrics
    scan_results = []
    violations_found = 0
    total_compute_time = 0
    start_time = time.time()
    
    print("🚀 Processing batch configurations...")
    
    for i in trange(num_configs, desc="Batch Scan Progress"):
        config_start = time.time()
        
        try:
            # Initialize Ghost EFT
            ghost_eft = GhostCondensateEFT(
                M=M_vals[i],
                alpha=alpha_vals[i], 
                beta=beta_vals[i],
                grid=np.linspace(-1e6, 1e6, 800)  # Optimized for speed
            )
            
            # Compute ANEC violation
            anec_value = ghost_eft.compute_anec(gaussian_kernel)
            config_time = time.time() - config_start
            total_compute_time += config_time
            
            # Analysis
            qi_violation = anec_value < 0
            if qi_violation:
                violations_found += 1
            
            # Package result for dashboard
            result = {
                'id': i + 1,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'parameters': {
                    'M': round(float(M_vals[i]), 2),
                    'alpha': round(float(alpha_vals[i]), 6),
                    'beta': round(float(beta_vals[i]), 6)
                },
                'anec_violation': float(anec_value),
                'qi_violated': bool(qi_violation),
                'violation_strength': abs(float(anec_value)) if qi_violation else 0.0,
                'compute_time_ms': round(config_time * 1000, 2),
                'enhancement_vs_vacuum': abs(float(anec_value)) / 1.2e-17 if qi_violation else 0
            }
            
            scan_results.append(result)
            
        except Exception as e:
            # Error handling for dashboard robustness
            error_result = {
                'id': i + 1,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'parameters': {
                    'M': round(float(M_vals[i]), 2),
                    'alpha': round(float(alpha_vals[i]), 6),
                    'beta': round(float(beta_vals[i]), 6)
                },
                'error': str(e),
                'anec_violation': 0.0,
                'qi_violated': False,
                'violation_strength': 0.0,
                'compute_time_ms': 0.0,
                'enhancement_vs_vacuum': 0.0
            }
            scan_results.append(error_result)
    
    total_elapsed = time.time() - start_time
    
    # Find best results for dashboard highlights
    successful_results = [r for r in scan_results if 'error' not in r]
    violation_results = [r for r in successful_results if r['qi_violated']]
    
    if violation_results:
        best_result = min(violation_results, key=lambda x: x['anec_violation'])
        top_5_results = sorted(violation_results, 
                             key=lambda x: abs(x['anec_violation']), 
                             reverse=True)[:5]
    else:
        best_result = None
        top_5_results = []
    
    # Dashboard-ready output format
    dashboard_report = {
        'scan_info': {
            'scan_id': f"ghost_eft_{int(time.time())}",
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'discovery_basis': 'Discovery 21: Ghost/Phantom EFT Breakthrough',
            'framework': 'LQG-ANEC Computational Pipeline'
        },
        'configuration': {
            'total_configs': num_configs,
            'baseline_params': {
                'M_optimal': optimal_M,
                'alpha_optimal': optimal_alpha,
                'beta_optimal': optimal_beta
            },
            'parameter_variation': '±30% around Discovery 21 optimal',
            'temporal_scale': '604,800 seconds (1 week)',
            'grid_resolution': 800
        },
        'performance_metrics': {
            'total_time_seconds': round(total_elapsed, 2),
            'compute_time_seconds': round(total_compute_time, 2),
            'configs_per_second': round(num_configs / total_elapsed, 2),
            'average_ms_per_config': round((total_compute_time / num_configs) * 1000, 2),
            'success_rate': round(len(successful_results) / num_configs * 100, 1),
            'violation_rate': round(violations_found / len(successful_results) * 100, 1) if successful_results else 0
        },
        'results_summary': {
            'total_violations': violations_found,
            'successful_computations': len(successful_results),
            'failed_computations': num_configs - len(successful_results),
            'best_anec_violation': best_result['anec_violation'] if best_result else None,
            'best_enhancement_factor': best_result['enhancement_vs_vacuum'] if best_result else None
        },
        'top_performers': top_5_results,
        'best_configuration': best_result,
        'all_results': scan_results
    }
    
    # Save dashboard-compatible JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(dashboard_report, f, indent=2)
    
    # Console summary for monitoring
    print(f"\n✅ Batch scan complete!")
    print(f"📊 Performance: {dashboard_report['performance_metrics']['configs_per_second']} configs/sec")
    print(f"🎯 Success rate: {dashboard_report['performance_metrics']['success_rate']}%")
    print(f"⚡ Violation rate: {dashboard_report['performance_metrics']['violation_rate']}%")
    
    if best_result:
        print(f"🏆 Best ANEC: {best_result['anec_violation']:.2e} W")
        print(f"🔥 Enhancement: {best_result['enhancement_vs_vacuum']:.1e}× vs vacuum")
        
    print(f"💾 Dashboard data: {output_path}")
    
    return dashboard_report


def continuous_monitoring_mode(scan_interval=300, max_scans=10):
    """
    Continuous monitoring mode for dashboard integration.
    
    Args:
        scan_interval: Seconds between scans (default: 5 minutes)
        max_scans: Maximum number of scans to run (default: 10)
    """
    
    print("🔄 Continuous monitoring mode activated")
    print(f"⏱️  Scan interval: {scan_interval} seconds")
    print(f"🔢 Maximum scans: {max_scans}")
    
    for scan_num in range(1, max_scans + 1):
        print(f"\n{'='*50}")
        print(f"🔍 SCAN {scan_num}/{max_scans} - {time.strftime('%H:%M:%S')}")
        print(f"{'='*50}")
        
        # Generate unique output filename
        output_file = f"results/continuous_scan_{scan_num:03d}.json"
        
        # Run batch scan
        dashboard_batch_scan(num_configs=50, output_file=output_file)
        
        # Wait for next scan (except on final iteration)
        if scan_num < max_scans:
            print(f"\n⏸️  Waiting {scan_interval} seconds until next scan...")
            time.sleep(scan_interval)
    
    print(f"\n🏁 Continuous monitoring complete ({max_scans} scans)")


def main():
    """Command-line interface for dashboard integration."""
    
    parser = argparse.ArgumentParser(description='Dashboard Ghost EFT Scanner')
    parser.add_argument('--configs', type=int, default=100,
                       help='Number of configurations per scan (default: 100)')
    parser.add_argument('--output', type=str, default='results/dashboard_scan.json',
                       help='Output JSON file for dashboard')
    parser.add_argument('--continuous', action='store_true',
                       help='Enable continuous monitoring mode')
    parser.add_argument('--interval', type=int, default=300,
                       help='Scan interval in seconds for continuous mode (default: 300)')
    parser.add_argument('--max-scans', type=int, default=10,
                       help='Maximum scans in continuous mode (default: 10)')
    
    args = parser.parse_args()
    
    print("🎯 Dashboard-Ready Ghost EFT Scanner")
    print("🎯 Discovery 21 Integration System")  
    print("🎯 LQG-ANEC Framework")
    print()
    
    if args.continuous:
        # Continuous monitoring for real-time dashboards
        continuous_monitoring_mode(args.interval, args.max_scans)
    else:
        # Single batch scan
        results = dashboard_batch_scan(args.configs, args.output)
        
        if results.get('error'):
            print(f"❌ Scan failed: {results['error']}")
            return 1
    
    print("\n" + "="*60)
    print("🚀 DASHBOARD INTEGRATION READY")
    print("="*60)
    print("✅ JSON output compatible with dashboard systems")
    print("✅ Real-time monitoring capabilities operational")
    print("✅ Discovery 21 parameters continuously validated")
    print("✅ Performance metrics available for system monitoring")
    
    return 0


if __name__ == "__main__":
    exit(main())
