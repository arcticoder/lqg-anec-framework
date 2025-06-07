#!/usr/bin/env python3
"""
Simple QI bounds comparison exactly as requested by the user.
"""
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from polymer_quantization import polymer_correction

def classical_qi(tau):
    return -3/(32 * np.pi**2 * tau**4)

def polymer_qi(tau, mu):
    # apply sinc correction multiplicatively
    return classical_qi(tau) * (np.sin(np.pi*mu) / (np.pi*mu) if mu!=0 else 1.0)

if __name__ == "__main__":
    taus = np.logspace(3, 7, num=20)    # 1e3‒1e7 s
    mus  = [0.0, 0.1, 0.5, 1.0]          # sample polymer scales
    header = ["tau(s)", "QI_classical"] + [f"QI_poly(mu={μ})" for μ in mus]
    print(", ".join(header))
    for τ in taus:
        row = [f"{τ:.1e}", f"{classical_qi(τ):.3e}"]
        row += [f"{polymer_qi(τ, μ):.3e}" for μ in mus]
        print(", ".join(row))
