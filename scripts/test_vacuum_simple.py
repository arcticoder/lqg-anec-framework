#!/usr/bin/env python3
"""
Quick test of vacuum configuration analysis
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from vacuum_engineering import CasimirArray, DynamicCasimir, SqueezedVacuumResonator

def test_vacuum_analysis():
    print("Testing vacuum engineering analysis...")
    
    # Test Casimir array
    casimir = CasimirArray(temperature=300.0)
    spacings = [10e-9] * 50
    permittivities = [1.0 + 0.001j] * 50
    casimir_pressure = casimir.stack_pressure(50, spacings, permittivities)
    print(f"Casimir array pressure: {casimir_pressure:.2e} Pa")
    
    # Test dynamic Casimir
    dynamic = DynamicCasimir(5e9, 0.1)
    dynamic_density = dynamic.total_density()
    print(f"Dynamic Casimir energy density: {dynamic_density:.2e} J/m³")
    
    # Test squeezed vacuum
    squeezed = SqueezedVacuumResonator(resonator_frequency=1e12, squeezing_parameter=2.0)
    volume = 1e-6  # 1 cubic mm
    squeezed_density = squeezed.squeezed_energy_density(volume)
    print(f"Squeezed vacuum energy density: {squeezed_density:.2e} J/m³")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_vacuum_analysis()
