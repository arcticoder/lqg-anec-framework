# src/__init__.py

"""
LQG-ANEC Framework

Loop Quantum Gravity - Averaged Null Energy Condition Analysis Framework

This package provides tools for analyzing ANEC violations in Loop Quantum Gravity
using coherent states, polymer corrections, and effective field theory.
"""

__version__ = "1.0.0"
__author__ = "LQG-ANEC Framework Team"

# Make key classes available at package level
try:
    from .coherent_states import CoherentState
    from .spin_network_utils import SpinNetwork, build_flat_graph
    from .stress_tensor_operator import StressTensorOperator, LocalT00, ScalarFieldStressTensor
    from .midisuperspace_model import MidiSuperspaceModel
    from .polymer_quantization import polymer_correction, PolymerOperator
    from .anec_violation_analysis import coherent_state_anec_violation
    
    __all__ = [
        'CoherentState',
        'SpinNetwork', 
        'build_flat_graph',
        'StressTensorOperator',
        'LocalT00',
        'ScalarFieldStressTensor', 
        'MidiSuperspaceModel',
        'polymer_correction',
        'PolymerOperator',
        'coherent_state_anec_violation'
    ]
    
except ImportError as e:
    # Graceful degradation if some modules are missing
    print(f"Warning: Some modules could not be imported: {e}")
    __all__ = []
