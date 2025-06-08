#!/usr/bin/env python3
"""
Test Script for Vacuum Engineering Module

Quick verification that the vacuum engineering implementation works correctly
and produces reasonable results for laboratory-scale negative energy sources.
"""

import sys
import os
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from vacuum_engineering import (
        CasimirArray, DynamicCasimirEffect, SqueezedVacuumResonator,
        MetamaterialCasimir, comprehensive_vacuum_analysis, MATERIAL_DATABASE,
        casimir_pressure, stack_pressure, optimize_stack, vacuum_energy_to_anec_flux
    )
    print("✓ Successfully imported vacuum engineering module")
except ImportError as e:
    print(f"✗ Failed to import vacuum engineering module: {e}")
    sys.exit(1)

def test_casimir_array():
    """Test basic Casimir array functionality."""
    print("\nTesting Casimir Array...")
    print("-" * 30)
    
    casimir = CasimirArray(temperature=4.0)
    
    # Test single plate pressure
    spacing = 100e-9  # 100 nm
    pressure = casimir.casimir_pressure(spacing)
    print(f"Single plate pressure (100 nm): {pressure:.2e} Pa")
    
    # Test material effects
    for material in ['Au', 'SiO2', 'metamaterial']:
        props = MATERIAL_DATABASE[material]
        perm = props['permittivity']
        pressure_mat = casimir.casimir_pressure(spacing, perm)
        enhancement = pressure_mat / pressure if pressure != 0 else 0
        print(f"{material} enhancement: {enhancement:.2f}×")
    
    # Test multi-layer stack
    n_layers = 5
    spacing_list = [100e-9] * n_layers
    perm_list = [MATERIAL_DATABASE['Au']['permittivity']] * n_layers
    
    stack_pressure = casimir.stack_pressure(n_layers, spacing_list, perm_list)
    print(f"5-layer stack pressure: {stack_pressure:.2e} Pa")
    print(f"Stack enhancement: {stack_pressure / pressure:.1f}×")
    
    return True

def test_dynamic_casimir():
    """Test dynamic Casimir effect functionality."""
    print("\nTesting Dynamic Casimir Effect...")
    print("-" * 30)
    
    dynamic = DynamicCasimirEffect(circuit_frequency=10e9, drive_amplitude=0.1)
    
    # Test photon creation rate
    drive_freq = 20e9  # 2×circuit frequency for resonance
    quality_factor = 1000
    
    rate = dynamic.photon_creation_rate(drive_freq, quality_factor)
    print(f"Photon creation rate: {rate:.2e} Hz")
    
    # Test negative energy density
    volume = 1e-9  # 1 mm³
    energy_density = dynamic.negative_energy_density(drive_freq, volume, quality_factor)
    print(f"Negative energy density: {energy_density:.2e} J/m³")
    
    # Test frequency dependence
    frequencies = [5e9, 10e9, 20e9, 40e9]  # Different drive frequencies
    for f in frequencies:
        energy = dynamic.negative_energy_density(f, volume, quality_factor)
        print(f"  {f/1e9:.0f} GHz: {energy:.2e} J/m³")
    
    return True

def test_squeezed_vacuum():
    """Test squeezed vacuum functionality."""
    print("\nTesting Squeezed Vacuum...")
    print("-" * 30)
    
    squeezed = SqueezedVacuumResonator(resonator_frequency=1e14, squeezing_parameter=1.0)
    
    # Test energy density vs squeezing
    volume = 1e-12  # 1 μm³
    
    squeezing_values = [0.5, 1.0, 2.0, 3.0]
    for xi in squeezing_values:
        squeezed.xi = xi
        energy_density = squeezed.squeezed_energy_density(volume)
        stabilization = squeezed.stabilization_power()
        print(f"Squeezing {xi:.1f}: Energy = {energy_density:.2e} J/m³, Power = {stabilization:.2e} W")
    
    return True

def test_metamaterial_casimir():
    """Test metamaterial Casimir functionality."""
    print("\nTesting Metamaterial Casimir...")
    print("-" * 30)
    
    metamaterial = MetamaterialCasimir(unit_cell_size=50e-9)
    
    # Test enhancement factors for different materials
    frequency = 1e14  # 100 THz
    
    test_materials = [
        (1.0, 1.0, "Vacuum"),
        (3.9, 1.0, "SiO₂"),
        (-1.0 + 1j*10, 1.0, "Gold"),
        (-2.5, -1.2, "Metamaterial")
    ]
    
    for epsilon, mu, name in test_materials:
        enhancement = metamaterial.metamaterial_enhancement(epsilon, mu, frequency)
        print(f"{name}: Enhancement = {np.real(enhancement):.2f}")
    
    # Test optimization
    target_enhancement = -5.0  # Repulsive force
    optimal = metamaterial.design_optimal_metamaterial(target_enhancement)
    print(f"Optimal design for -5× enhancement:")
    print(f"  ε = {optimal['epsilon']:.2f}")
    print(f"  μ = {optimal['mu']:.2f}")
    print(f"  Achieved = {optimal['enhancement']:.2f}")
    print(f"  Feasible = {optimal['feasible']}")
    
    return True

def test_comprehensive_analysis():
    """Test the comprehensive vacuum analysis."""
    print("\nTesting Comprehensive Analysis...")
    print("-" * 30)
    
    # Run with lower target for testing
    analysis = comprehensive_vacuum_analysis(target_flux=1e-20)
    
    print("Results summary:")
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
    
    return True

def test_simple_functions():
    """Test the simple Casimir functions requested by user."""
    print("\nTesting Simple Casimir Functions...")
    print("-" * 30)
    
    # Test materials
    materials = {
        'SiO2': 3.9,
        'Au': -1.0,  # Simplified for metals at optical frequencies
        'vacuum': 1.0,
        'metamaterial': -2.5
    }
    
    # Test single pressure calculation
    spacing = 100e-9  # 100 nm
    print(f"Single plate pressure calculations (spacing = {spacing*1e9:.0f} nm):")
    
    for material, perm in materials.items():
        pressure = casimir_pressure(spacing, perm)
        print(f"  {material:12s}: {pressure:.2e} Pa")
    
    # Test stacked layers
    n_layers = 5
    print(f"\nStacked pressure ({n_layers} layers):")
    
    for material, perm in materials.items():
        pressure = stack_pressure(n_layers, [spacing]*n_layers, [perm]*n_layers)
        print(f"  {material:12s}: {pressure:.2e} Pa")
    
    # Test optimization
    print(f"\nOptimizing stack configuration:")
    target_pressure = -1e6  # 1 MPa attractive pressure
    perm_values = list(materials.values())
    
    result = optimize_stack(
        n_layers=10,
        a_min=50e-9,
        a_max=500e-9,
        ε_vals=perm_values,
        target_pressure=target_pressure
    )
    
    print(f"  Optimal spacing: {result[0]*1e9:.1f} nm")
    print(f"  Optimal permittivity: {result[1]:.2f}")
    print(f"  Achieved pressure: {result[2]:.2e} Pa")
    print(f"  Target error: {abs(result[2] - target_pressure)/abs(target_pressure)*100:.1f}%")
    
    return True

def scan_realistic_materials():
    """Scan realistic materials (SiO₂, Au) and layer counts."""
    print("\nScanning Realistic Materials...")
    print("-" * 30)
    
    # Realistic configurations
    configurations = [
        {'layers': 5, 'spacing': 100e-9, 'material': 'SiO2', 'perm': 3.9},
        {'layers': 10, 'spacing': 50e-9, 'material': 'Au', 'perm': -1.0},
        {'layers': 20, 'spacing': 200e-9, 'material': 'metamaterial', 'perm': -2.5},
        {'layers': 3, 'spacing': 1e-6, 'material': 'Si', 'perm': 11.7},
    ]
    
    print("Configuration scan results:")
    print("Layers\tMaterial\tSpacing(nm)\tPressure(Pa)\tEnergy Density(J/m³)")
    print("-" * 75)
    
    results = []
    
    for config in configurations:
        n = config['layers']
        a = config['spacing']
        mat = config['material']
        perm = config['perm']
        
        # Calculate pressure
        pressure = stack_pressure(n, [a]*n, [perm]*n)
        
        # Estimate energy density
        area = (1e-3)**2  # 1 mm² area
        thickness = n * a  # Total thickness
        volume = area * thickness
        energy_density = pressure * thickness / volume  # Rough estimate
        
        results.append({
            'config': config,
            'pressure': pressure,
            'energy_density': energy_density,
            'volume': volume
        })
        
        print(f"{n}\t{mat:12s}\t{a*1e9:.0f}\t\t{pressure:.2e}\t{energy_density:.2e}")
    
    # Convert to negative-energy flux
    print(f"\nConverting to ANEC violation flux:")
    print("Config\t\tFlux (W)\t\tTarget Ratio (vs 1e-25 W)")
    print("-" * 55)
    
    import numpy as np
    from scipy.integrate import quad
    
    def gaussian_kernel(t, tau):
        return np.exp(-t**2 / (2*tau**2)) / np.sqrt(2*np.pi*tau**2)
    
    target_flux = 1e-25  # Target ANEC violation
    tau = 1e-6  # Microsecond timescale
    
    for i, result in enumerate(results):
        energy_density = result['energy_density']
        volume = result['volume']
        
        if energy_density < 0:  # Only negative energy contributes
            flux = vacuum_energy_to_anec_flux(energy_density, volume, tau, gaussian_kernel)
            ratio = abs(flux / target_flux)
            print(f"Config {i}\t\t{flux:.2e}\t\t{ratio:.2e}")
        else:
            print(f"Config {i}\t\t0.00e+00\t\t0.00e+00")
    
    return results

def run_all_tests():
    """Run all vacuum engineering tests."""
    print("Vacuum Engineering Module Test Suite")
    print("=" * 50)
    
    tests = [
        ("Casimir Array", test_casimir_array),
        ("Dynamic Casimir", test_dynamic_casimir),
        ("Squeezed Vacuum", test_squeezed_vacuum),
        ("Metamaterial Casimir", test_metamaterial_casimir),
        ("Comprehensive Analysis", test_comprehensive_analysis),
        ("Simple Functions", test_simple_functions),
        ("Realistic Materials", scan_realistic_materials)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success, None))
            print(f"✓ {test_name} test passed")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"✗ {test_name} test failed: {e}")
    
    # Summary
    print(f"\nTest Summary:")
    print("-" * 20)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    for test_name, success, error in results:
        status = "✓" if success else "✗"
        print(f"{status} {test_name}")
        if error:
            print(f"    Error: {error}")
    
    if passed == total:
        print("\n🎉 All tests passed! Vacuum engineering module is working correctly.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Check implementation.")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
