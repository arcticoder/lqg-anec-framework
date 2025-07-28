#!/usr/bin/env python3
import os
import json
import math
import sympy as sp
import pytest
from metric_ansatz_development import MetricAnsatzBuilder

def test_soliton_ansatz_against_reference():
    # Load the “golden” reference data for soliton profiles
    ref_path = os.path.join(os.path.dirname(__file__), "reference_soliton.json")
    with open(ref_path, "r") as f:
        ref = json.load(f)

    # Instantiate your ansatz builder
    builder = MetricAnsatzBuilder()

    # Predefine the actual Symbol objects used inside soliton_ansatz
    a_sym, b_sym, x0_sym, c_sym, r_sym = sp.symbols('a_0 b_0 x0_0 c_0 r', real=True)

    # Fetch the symbolic soliton expression once
    expr = builder.soliton_ansatz(variable='r', soliton_type='tanh', num_solitons=1)

    # Emit a header just like the 3nj test does
    print("\n=== Soliton Ansatz vs Reference Dataset ===")
    for key, expected in ref.items():
        # key format: "a,b,x0,c0"
        a_val, b_val, x0_val, c0_val = [sp.Rational(s) for s in key.split(",")]

        # Build a true symbol→number map
        subs_map = {
            a_sym:  a_val,
            b_sym:  b_val,
            x0_sym: x0_val,
            c_sym:  c0_val,
            r_sym:  sp.Rational(0)
        }

        # Perform the substitution and numeric evaluation
        numeric = expr.subs(subs_map).evalf()
        # Emit full table into the CI log
        print(f"{key}: result={numeric}, expected={expected}")

        # Compare with a small tolerance
        assert math.isclose(float(numeric), expected, rel_tol=1e-8), f"{key}: {numeric} != {expected}"
