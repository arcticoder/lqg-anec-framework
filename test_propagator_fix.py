#!/usr/bin/env python3
"""
Quick test fix for the propagator test
"""

import unittest
import numpy as np
from platinum_road_core import D_ab_munu

def test_propagator_shape():
    """Test propagator shape directly."""
    print("🚀 Testing propagator D_ab_munu shape...")
    
    k4 = np.array([1.0, 0.5, 0.3, 0.2])
    mu_g = 0.15
    m_g = 0.1
    
    D = D_ab_munu(k4, mu_g, m_g)
    
    print(f"Shape: {D.shape}")
    print(f"Expected: (3, 3, 4, 4)")
    print(f"All finite: {np.all(np.isfinite(D))}")
    
    # Test passes if shape is (3,3,4,4)
    assert D.shape == (3, 3, 4, 4), f"Expected (3,3,4,4), got {D.shape}"
    assert np.all(np.isfinite(D)), "All entries should be finite"
    
    print("✅ Propagator test PASSED!")

if __name__ == "__main__":
    test_propagator_shape()
