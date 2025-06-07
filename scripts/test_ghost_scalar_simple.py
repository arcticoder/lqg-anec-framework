#!/usr/bin/env python3
"""
Simple Ghost Scalar ANEC Test - Exact format as requested
"""
import numpy as np

# Grid parameters
Nt, Nx = 201, 201
t_vals = np.linspace(-5, 5, Nt)
x_vals = np.linspace(-5, 5, Nx)
dt = t_vals[1] - t_vals[0]

# Ghost pulse φ(t=0, x)
sigma = 1.0
phi0 = np.exp(-x_vals**2/(2*sigma**2))

# Build static field array: φ(t,x) constant in t
phi = np.tile(phi0, (Nt, 1))

# Compute derivatives via centered differences (static, so ∂tφ=0)
dphi_dx = np.zeros_like(phi)
dphi_dx[:,1:-1] = (phi[:,2:] - phi[:,:-2])/(2*(x_vals[1]-x_vals[0]))
dphi_dt = np.zeros_like(phi)  # zero for static

# Null vector k^a = (1, 1)
# T_{uu} = T_{tt} + 2 T_{tx} + T_{xx}, but T_{tx}=0 here
# For ghost: T_{tt} = -½(∂tφ)^2 - ½(∂xφ)^2 - V
#            T_{xx} = -½(∂tφ)^2 - ½(∂xφ)^2 + V
#   ⇒ T_{uu} = - (∂tφ)^2 - (∂xφ)^2
T_uu = - (dphi_dt**2 + dphi_dx**2)

# Integrate along the null line t=x (i.e. indices i=j)
I = 0.0
for idx in range(min(Nt, Nx)):
    I += T_uu[idx, idx] * dt

print(f"ANEC integral T_uu dt along t=x: {I:.3e}  (J/m³·s)")
if I < 0:
    print("✓ Net negative ANEC violation achieved in this toy model.")
else:
    print("✗ No violation in this static profile.")
