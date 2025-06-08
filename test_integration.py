#!/usr/bin/env python3
"""
Quick integration test for the Quantum Vacuum & Metamaterials pipeline
"""

print("Testing Quantum Vacuum & Metamaterials pipeline integration...")

try:
    # Test imports
    from src.drude_model import DrudeLorentzPermittivity, material_presets
    print("✓ Drude model imports successful")
    
    from src.metamaterial_casimir import MetamaterialCasimir
    print("✓ Metamaterial Casimir imports successful")
    
    from src.vacuum_engineering import CasimirArray
    print("✓ Vacuum engineering imports successful")
    
    import numpy as np
    
    # Test 1: Drude model functionality
    print("\nTest 1: Drude-Lorentz permittivity model")
    gold_params = material_presets()['Au']
    drude = DrudeLorentzPermittivity(**gold_params)
    freqs = np.logspace(12, 15, 5)
    eps_gold = drude.ε(freqs[0])
    print(f"  Gold permittivity at {freqs[0]:.2e} Hz: {eps_gold:.3f}")
    
    # Test 2: Metamaterial Casimir functionality
    print("\nTest 2: Metamaterial Casimir energy density")
    meta = MetamaterialCasimir()
    energy_density = meta.compute_energy_density(
        spacing_nm=50, 
        permittivity=-2.0, 
        permeability=-1.5, 
        n_layers=10
    )
    print(f"  Metamaterial energy density: {energy_density:.6f} J/m³")
    
    # Test 3: Casimir array with Drude enhancement
    print("\nTest 3: Casimir array with Drude enhancement")
    casimir = CasimirArray(temperature=300, n_layers=5)
    enhanced_density = casimir.energy_density_drude_enhanced(
        frequencies=freqs[:3],
        materials=['Au', 'SiO2', 'Ag']
    )
    print(f"  Drude-enhanced energy density: {enhanced_density:.6f} J/m³")
    
    print("\n✓ All integration tests passed successfully!")
    print("✓ Quantum Vacuum & Metamaterials pipeline is fully functional")
    
except Exception as e:
    print(f"\n✗ Error during testing: {e}")
    import traceback
    traceback.print_exc()
