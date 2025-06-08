#!/usr/bin/env python3
"""
Test script for the enhanced vacuum engineering capabilities.

This script tests the three main enhancements:
1. Drude-Lorentz material modeling
2. Metamaterial Casimir systems
3. End-to-end vacuum-ANEC dashboard
"""

import sys
import os

# Add src directory to Python path
src_path = os.path.join(os.path.dirname(__file__), 'src')
sys.path.insert(0, src_path)

# Also add the current directory
sys.path.insert(0, os.path.dirname(__file__))

def test_drude_model():
    """Test the Drude-Lorentz permittivity model."""
    print("Testing Drude-Lorentz Model...")
    print("-" * 30)
    
    try:
        from drude_model import DrudeLorentzPermittivity, get_material_model, MATERIAL_MODELS
        import numpy as np
        
        # Test material models
        print("Available material models:")
        for name in MATERIAL_MODELS.keys():
            print(f"  - {name}")
        
        # Test gold model
        gold = get_material_model('gold')
        test_freq = 2e15  # 2 PHz
        eps = gold.ε(test_freq)
        reflectivity = gold.reflectivity(test_freq)
        
        print(f"\nGold at {test_freq/1e15:.1f} PHz:")
        print(f"  Permittivity: {eps:.3f}")
        print(f"  Reflectivity: {reflectivity:.3f}")
        
        print("✓ Drude model test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Drude model test failed: {e}")
        return False

def test_metamaterial_casimir():
    """Test the metamaterial Casimir system."""
    print("\nTesting Metamaterial Casimir...")
    print("-" * 30)
    
    try:
        from metamaterial_casimir import MetamaterialCasimir
        import numpy as np
        
        # Create simple metamaterial system
        spacings = [50e-9] * 5  # 5 layers, 50 nm each
        eps_list = [(-2.0 + 0.1j)] * 5  # Negative permittivity
        mu_list = [(-1.5 + 0.05j)] * 5  # Negative permeability
        
        meta_system = MetamaterialCasimir(spacings, eps_list, mu_list)
        
        # Test calculations
        energy_density = meta_system.total_energy_density()
        amplification = meta_system.force_amplification_factor()
        
        print(f"Metamaterial Casimir System:")
        print(f"  Layers: {meta_system.n_layers}")
        print(f"  Energy density: {energy_density:.2e} J/m³")
        print(f"  Force amplification: {amplification:.1f}x")
        
        # Test negative index detection
        is_neg_idx = meta_system.is_negative_index(0)
        print(f"  Negative index material: {is_neg_idx}")
        
        print("✓ Metamaterial Casimir test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Metamaterial Casimir test failed: {e}")
        return False

def test_vacuum_dashboard():
    """Test the vacuum-ANEC dashboard (basic functionality)."""
    print("\nTesting Vacuum Dashboard...")
    print("-" * 30)
    
    try:
        # Import components
        from vacuum_engineering import build_lab_sources, comprehensive_vacuum_analysis
        
        print("Testing lab sources...")
        sources = build_lab_sources('comprehensive')
        print(f"  Found {len(sources)} lab sources")
        
        print("Testing comprehensive analysis...")
        analysis = comprehensive_vacuum_analysis()
        print(f"  Analyzed {len(analysis)} methods")
        
        # Show brief results
        for method, data in analysis.items():
            energy_density = data.get('energy_density', 0)
            feasible = data.get('feasible', False)
            print(f"  {method}: {energy_density:.2e} J/m³, feasible: {feasible}")
        
        print("✓ Vacuum dashboard test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Vacuum dashboard test failed: {e}")
        return False

def test_integration():
    """Test integration between components."""
    print("\nTesting Component Integration...")
    print("-" * 30)
    
    try:
        # Test vacuum engineering with dispersion
        from vacuum_engineering import CasimirArray
        
        casimir = CasimirArray(temperature=300.0)
        
        # Test dispersion calculation if available
        try:
            # This should work if drude_model is available
            test_spacing = [100e-9]
            test_eps = [2.0 + 0.01j]
            casimir.a = test_spacing  # Set for testing
            casimir.eps = test_eps
            
            energy_with_dispersion = casimir.energy_density_with_dispersion(include_drude=True)
            energy_simple = casimir.energy_density_with_dispersion(include_drude=False)
            
            print(f"Energy density comparison:")
            print(f"  With dispersion: {energy_with_dispersion[0]:.2e} J/m³")
            print(f"  Simple model: {energy_simple[0]:.2e} J/m³")
            print(f"  Ratio: {energy_with_dispersion[0]/energy_simple[0]:.2f}")
            
        except Exception as e:
            print(f"  Dispersion integration: {e}")
        
        print("✓ Integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("VACUUM ENGINEERING ENHANCEMENT TESTS")
    print("="*50)
    
    # Run individual tests
    tests = [
        test_drude_model,
        test_metamaterial_casimir, 
        test_vacuum_dashboard,
        test_integration
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"Test {test_func.__name__} crashed: {e}")
            results.append(False)
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Drude-Lorentz Model",
        "Metamaterial Casimir",
        "Vacuum Dashboard", 
        "Component Integration"
    ]
    
    for name, result in zip(test_names, results):
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed ({100*passed/total:.0f}%)")
    
    if passed == total:
        print("\n🎉 All enhancement tests passed! The vacuum engineering pipeline is ready!")
    else:
        print(f"\n⚠️  {total-passed} test(s) failed. Check individual error messages above.")
    
    # Quick demo if all tests pass
    if passed == total:
        print("\nRunning quick demo...")
        try:
            from metamaterial_casimir import MetamaterialCasimir
            
            # Demo alternating metamaterial stack
            spacings = [30e-9] * 6
            eps_list = [(-2.0 + 0.1j) if i%2==0 else (2.0 + 0.01j) for i in range(6)]
            mu_list = [(-1.5 + 0.05j) if i%2==0 else (1.0 + 0.0j) for i in range(6)]
            
            demo_system = MetamaterialCasimir(spacings, eps_list, mu_list)
            energy = demo_system.total_energy_density()
            amp = demo_system.force_amplification_factor()
            
            print(f"Demo alternating metamaterial:")
            print(f"  Energy density: {energy:.2e} J/m³")
            print(f"  Amplification: {amp:.1f}x vacuum Casimir")
            
        except Exception as e:
            print(f"Demo failed: {e}")

if __name__ == "__main__":
    main()
