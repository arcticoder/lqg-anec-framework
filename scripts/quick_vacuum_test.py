#!/usr/bin/env python3
"""
Quick test of vacuum configuration analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from vacuum_engineering import CasimirArray, SqueezedVacuumResonator, MATERIAL_DATABASE

def quick_test():
    print("Quick Vacuum Configuration Test")
    print("=" * 40)
    
    # Test Casimir array
    casimir = CasimirArray(temperature=4.0)
    spacing = 100e-9  # 100 nm
    material = 'SiO2'
    layers = 10
    
    pressure = casimir.casimir_pressure(
        spacing, 
        MATERIAL_DATABASE[material]['permittivity']
    )
    
    print(f"Casimir Array Test:")
    print(f"  Material: {material}")
    print(f"  Spacing: {spacing:.1e} m")
    print(f"  Layers: {layers}")
    print(f"  Single pressure: {pressure:.2e} Pa")
    print(f"  Total pressure: {pressure * layers:.2e} Pa")
    
    # Test squeezed vacuum
    squeezed = SqueezedVacuumResonator(
        resonator_frequency=1e14,
        squeezing_parameter=2.0
    )
    
    volume = 1e-12  # Picoliter
    energy_density = squeezed.squeezed_energy_density(volume)
    power = squeezed.stabilization_power()
    
    print(f"\nSqueezed Vacuum Test:")
    print(f"  Frequency: {1e14:.1e} Hz")
    print(f"  Squeezing: 2.0")
    print(f"  Volume: {volume:.1e} m³")
    print(f"  Energy density: {energy_density:.2e} J/m³")
    print(f"  Stabilization power: {power:.2e} W")
    
    print(f"\nTest completed successfully!")

if __name__ == "__main__":
    quick_test()
