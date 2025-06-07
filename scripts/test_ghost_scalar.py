#!/usr/bin/env python3
"""
Prototype Ghost-Scalar EFT with ANEC Violation

This script implements a simple 1+1D ghost scalar field theory to demonstrate
controlled NEC/ANEC violations. The Lagrangian is:

L = -½(∂φ)² - V(φ)

which gives the stress tensor:
T_ab = -∂_a φ ∂_b φ + g_ab(-½(∂φ)² - V)

We evolve a ghost pulse and compute the ANEC integral along null rays.

Author: LQG-ANEC Framework Development Team
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

class GhostScalarEFT:
    """
    1+1D Ghost Scalar Field Theory with controlled ANEC violations.
    """
    
    def __init__(self, Nt=201, Nx=201, t_range=(-5, 5), x_range=(-5, 5)):
        """
        Initialize the ghost scalar field system.
        
        Args:
            Nt, Nx: Grid points in time and space
            t_range, x_range: Domain ranges (t_min, t_max), (x_min, x_max)
        """
        self.Nt, self.Nx = Nt, Nx
        
        # Grid setup
        self.t_vals = np.linspace(t_range[0], t_range[1], Nt)
        self.x_vals = np.linspace(x_range[0], x_range[1], Nx)
        self.dt = self.t_vals[1] - self.t_vals[0]
        self.dx = self.x_vals[1] - self.x_vals[0]
        
        # Field arrays: phi(t,x), phi_dot(t,x)
        self.phi = np.zeros((Nt, Nx))
        self.phi_dot = np.zeros((Nt, Nx))
        
        # Derivative arrays
        self.dphi_dt = np.zeros((Nt, Nx))
        self.dphi_dx = np.zeros((Nt, Nx))
        
        # Stress tensor components
        self.T_tt = np.zeros((Nt, Nx))
        self.T_tx = np.zeros((Nt, Nx))
        self.T_xx = np.zeros((Nt, Nx))
        self.T_uu = np.zeros((Nt, Nx))  # Null component T_ab k^a k^b
        
        print(f"Initialized Ghost Scalar EFT: {Nt}×{Nx} grid")
        print(f"   Time range: {t_range}, Space range: {x_range}")
        print(f"   Grid spacing: dt={self.dt:.3f}, dx={self.dx:.3f}")
    
    def set_initial_conditions(self, profile_type="gaussian_pulse", **kwargs):
        """
        Set initial field configuration.
        
        Args:
            profile_type: Type of initial profile
            **kwargs: Parameters for the profile
        """
        if profile_type == "gaussian_pulse":
            amplitude = kwargs.get("amplitude", 1.0)
            width = kwargs.get("width", 1.0)
            center = kwargs.get("center", 0.0)
            velocity = kwargs.get("velocity", 0.0)
            
            # Initial field: Gaussian in space
            self.phi[0, :] = amplitude * np.exp(-(self.x_vals - center)**2 / (2*width**2))
            
            # Initial velocity: can be zero (static) or moving pulse
            self.phi_dot[0, :] = velocity * self.phi[0, :]
            
            print(f"Set Gaussian pulse: A={amplitude}, σ={width}, v={velocity}")
            
        elif profile_type == "sine_wave":
            amplitude = kwargs.get("amplitude", 1.0)
            wavelength = kwargs.get("wavelength", 2.0)
            phase = kwargs.get("phase", 0.0)
            
            k = 2*np.pi / wavelength
            self.phi[0, :] = amplitude * np.sin(k * self.x_vals + phase)
            self.phi_dot[0, :] = 0.0  # Start at rest
            
            print(f"Set sine wave: A={amplitude}, λ={wavelength}")
            
        elif profile_type == "soliton_like":
            amplitude = kwargs.get("amplitude", 1.0)
            width = kwargs.get("width", 1.0)
            center = kwargs.get("center", 0.0)
            
            # Sech-squared profile (soliton-like)
            self.phi[0, :] = amplitude / np.cosh((self.x_vals - center) / width)**2
            self.phi_dot[0, :] = 0.0
            
            print(f"Set soliton-like pulse: A={amplitude}, w={width}")
    
    def potential(self, phi, V_type="none", **kwargs):
        """
        Compute potential V(φ) and its derivative.
        
        Args:
            phi: Field values
            V_type: Type of potential
            **kwargs: Potential parameters
            
        Returns:
            V, dV_dphi
        """
        if V_type == "none":
            return np.zeros_like(phi), np.zeros_like(phi)
        
        elif V_type == "quadratic":
            m_squared = kwargs.get("m_squared", 0.1)
            V = 0.5 * m_squared * phi**2
            dV_dphi = m_squared * phi
            return V, dV_dphi
        
        elif V_type == "quartic":
            lambda_param = kwargs.get("lambda", 0.1)
            V = 0.25 * lambda_param * phi**4
            dV_dphi = lambda_param * phi**3
            return V, dV_dphi
        
        elif V_type == "mexican_hat":
            mu_squared = kwargs.get("mu_squared", -0.1)
            lambda_param = kwargs.get("lambda", 0.1)
            V = 0.5 * mu_squared * phi**2 + 0.25 * lambda_param * phi**4
            dV_dphi = mu_squared * phi + lambda_param * phi**3
            return V, dV_dphi
    
    def compute_derivatives(self, time_index):
        """
        Compute spatial and temporal derivatives using finite differences.
        
        Args:
            time_index: Which time slice to compute derivatives for
        """
        # Spatial derivative ∂φ/∂x (centered differences with periodic BC)
        self.dphi_dx[time_index, 1:-1] = (self.phi[time_index, 2:] - self.phi[time_index, :-2]) / (2*self.dx)
        # Boundary conditions (periodic)
        self.dphi_dx[time_index, 0] = (self.phi[time_index, 1] - self.phi[time_index, -1]) / (2*self.dx)
        self.dphi_dx[time_index, -1] = (self.phi[time_index, 0] - self.phi[time_index, -2]) / (2*self.dx)
        
        # Temporal derivative ∂φ/∂t (use stored phi_dot or compute from evolution)
        self.dphi_dt[time_index, :] = self.phi_dot[time_index, :]
    
    def compute_stress_tensor(self, time_index, V_type="none", **kwargs):
        """
        Compute stress tensor components for ghost scalar.
        
        For ghost scalar: T_ab = -∂_a φ ∂_b φ + g_ab(-½(∂φ)² - V)
        
        Args:
            time_index: Which time slice to compute for
            V_type: Potential type
            **kwargs: Potential parameters
        """
        # Get derivatives
        self.compute_derivatives(time_index)
        
        phi_t = self.dphi_dt[time_index, :]
        phi_x = self.dphi_dx[time_index, :]
        
        # Potential energy
        V, _ = self.potential(self.phi[time_index, :], V_type, **kwargs)
        
        # Kinetic and gradient energy densities
        kinetic = 0.5 * phi_t**2
        gradient = 0.5 * phi_x**2
        
        # Ghost scalar stress tensor components
        # T_tt = -∂_t φ ∂_t φ + (-½(∂φ)² - V) = -phi_t² - ½phi_t² - ½phi_x² - V
        self.T_tt[time_index, :] = -phi_t**2 - kinetic - gradient - V
        
        # T_tx = -∂_t φ ∂_x φ = -phi_t * phi_x
        self.T_tx[time_index, :] = -phi_t * phi_x
        
        # T_xx = -∂_x φ ∂_x φ + (-½(∂φ)² - V) = -phi_x² - ½phi_t² - ½phi_x² - V
        self.T_xx[time_index, :] = -phi_x**2 - kinetic - gradient - V
        
        # Null component T_uu = T_ab k^a k^b with k^a = (1, 1) (future null)
        # T_uu = T_tt + 2*T_tx + T_xx
        self.T_uu[time_index, :] = self.T_tt[time_index, :] + 2*self.T_tx[time_index, :] + self.T_xx[time_index, :]
    
    def evolve_static(self, V_type="none", **kwargs):
        """
        Keep field static (no time evolution) - useful for testing stress tensor.
        """
        print("Evolving (static case)...")
        
        # Copy initial conditions to all time slices
        for t_idx in range(self.Nt):
            self.phi[t_idx, :] = self.phi[0, :]
            self.phi_dot[t_idx, :] = self.phi_dot[0, :]
            
        # Compute stress tensor for all times
        for t_idx in range(self.Nt):
            self.compute_stress_tensor(t_idx, V_type, **kwargs)
        
        print("Static evolution complete.")
    
    def evolve_free_wave(self):
        """
        Evolve as free wave equation: ∂²φ/∂t² = ∂²φ/∂x²
        (This is for standard scalar, not ghost - included for comparison)
        """
        print("Evolving (free wave equation)...")
        
        # Use leapfrog integration
        for t_idx in range(1, self.Nt):
            if t_idx == 1:
                # First step: φ(t+dt) = φ(t) + dt*φ_dot(t) + ½dt²*∂²φ/∂x²
                d2phi_dx2 = np.zeros_like(self.phi[0, :])
                d2phi_dx2[1:-1] = (self.phi[0, 2:] - 2*self.phi[0, 1:-1] + self.phi[0, :-2]) / self.dx**2
                
                self.phi[1, :] = (self.phi[0, :] + self.dt * self.phi_dot[0, :] + 
                                 0.5 * self.dt**2 * d2phi_dx2)
            else:
                # Subsequent steps: φ(t+dt) = 2*φ(t) - φ(t-dt) + dt²*∂²φ/∂x²
                d2phi_dx2 = np.zeros_like(self.phi[t_idx-1, :])
                d2phi_dx2[1:-1] = (self.phi[t_idx-1, 2:] - 2*self.phi[t_idx-1, 1:-1] + 
                                  self.phi[t_idx-1, :-2]) / self.dx**2
                
                self.phi[t_idx, :] = (2*self.phi[t_idx-1, :] - self.phi[t_idx-2, :] + 
                                     self.dt**2 * d2phi_dx2)
            
            # Update phi_dot
            if t_idx > 0:
                self.phi_dot[t_idx, :] = (self.phi[t_idx, :] - self.phi[t_idx-1, :]) / self.dt
            
            # Compute stress tensor
            self.compute_stress_tensor(t_idx)
        
        print("Wave evolution complete.")
    
    def compute_anec_integral(self, null_direction="future"):
        """
        Compute ANEC integral along null geodesics.
        
        Args:
            null_direction: "future" (k^a = (1,1)) or "past" (k^a = (1,-1))
            
        Returns:
            Dictionary with ANEC results
        """
        print(f"Computing ANEC integral (null direction: {null_direction})...")
        
        if null_direction == "future":
            # Integrate along t = x + const lines
            anec_integrals = []
            
            # Different null rays (different constants)
            x_offsets = np.linspace(-2, 2, 11)  # Sample different null rays
            
            for x_offset in x_offsets:
                integral = 0.0
                count = 0
                
                for t_idx in range(self.Nt):
                    t = self.t_vals[t_idx]
                    x_target = t - x_offset  # Null line: x = t - offset
                    
                    # Find closest grid point
                    if self.x_vals[0] <= x_target <= self.x_vals[-1]:
                        x_idx = np.argmin(np.abs(self.x_vals - x_target))
                        integral += self.T_uu[t_idx, x_idx] * self.dt
                        count += 1
                
                if count > 0:
                    anec_integrals.append(integral)
            
            anec_integrals = np.array(anec_integrals)
            
        else:  # past null direction
            # Similar but along t = -x + const lines
            anec_integrals = []
            x_offsets = np.linspace(-2, 2, 11)
            
            for x_offset in x_offsets:
                integral = 0.0
                count = 0
                
                for t_idx in range(self.Nt):
                    t = self.t_vals[t_idx]
                    x_target = -t + x_offset  # Null line: x = -t + offset
                    
                    if self.x_vals[0] <= x_target <= self.x_vals[-1]:
                        x_idx = np.argmin(np.abs(self.x_vals - x_target))
                        integral += self.T_uu[t_idx, x_idx] * self.dt
                        count += 1
                
                if count > 0:
                    anec_integrals.append(integral)
            
            anec_integrals = np.array(anec_integrals)
        
        # Analysis
        results = {
            "anec_values": anec_integrals,
            "min_anec": np.min(anec_integrals) if len(anec_integrals) > 0 else 0,
            "max_anec": np.max(anec_integrals) if len(anec_integrals) > 0 else 0,
            "mean_anec": np.mean(anec_integrals) if len(anec_integrals) > 0 else 0,
            "violations": np.sum(anec_integrals < -1e-10),  # Count negative values
            "direction": null_direction
        }
        
        print(f"   ANEC range: [{results['min_anec']:.3e}, {results['max_anec']:.3e}]")
        print(f"   Mean ANEC: {results['mean_anec']:.3e}")
        print(f"   Violations: {results['violations']}/{len(anec_integrals)}")
        
        return results

def analyze_ghost_scalar_anec():
    """
    Comprehensive analysis of ANEC violations in ghost scalar theory.
    """
    print("=== Ghost Scalar EFT ANEC Analysis ===\n")
    
    # Test different configurations
    configs = [
        {
            "name": "Static Gaussian Pulse",
            "evolution": "static",
            "initial": {"profile_type": "gaussian_pulse", "amplitude": 2.0, "width": 1.0},
            "potential": {"V_type": "none"}
        },
        {
            "name": "Static Pulse with Quadratic Potential", 
            "evolution": "static",
            "initial": {"profile_type": "gaussian_pulse", "amplitude": 1.5, "width": 0.8},
            "potential": {"V_type": "quadratic", "m_squared": 0.1}
        },
        {
            "name": "Soliton-like Profile",
            "evolution": "static", 
            "initial": {"profile_type": "soliton_like", "amplitude": 1.0, "width": 1.2},
            "potential": {"V_type": "none"}
        },
        {
            "name": "Sine Wave with Mexican Hat",
            "evolution": "static",
            "initial": {"profile_type": "sine_wave", "amplitude": 1.0, "wavelength": 4.0},
            "potential": {"V_type": "mexican_hat", "mu_squared": -0.05, "lambda": 0.1}
        }
    ]
    
    results_summary = []
    
    for i, config in enumerate(configs):
        print(f"{i+1}. Testing: {config['name']}")
        
        # Initialize system
        eft = GhostScalarEFT(Nt=101, Nx=101, t_range=(-3, 3), x_range=(-5, 5))
        
        # Set initial conditions
        eft.set_initial_conditions(**config["initial"])
        
        # Evolve
        if config["evolution"] == "static":
            eft.evolve_static(**config["potential"])
        elif config["evolution"] == "wave":
            eft.evolve_free_wave()
        
        # Compute ANEC
        anec_future = eft.compute_anec_integral("future")
        anec_past = eft.compute_anec_integral("past")
        
        config_result = {
            "name": config["name"],
            "future_anec": anec_future,
            "past_anec": anec_past,
            "max_violation": min(anec_future["min_anec"], anec_past["min_anec"]),
            "system": eft
        }
        
        results_summary.append(config_result)
        
        print(f"   Future null ANEC: {anec_future['min_anec']:.3e} to {anec_future['max_anec']:.3e}")
        print(f"   Past null ANEC: {anec_past['min_anec']:.3e} to {anec_past['max_anec']:.3e}")
        print(f"   Max violation: {config_result['max_violation']:.3e}")
        print()
    
    return results_summary

def plot_ghost_analysis(results_summary):
    """
    Generate comprehensive plots of ghost scalar analysis.
    """
    print("Generating analysis plots...")
    
    n_configs = len(results_summary)
    fig, axes = plt.subplots(2, n_configs, figsize=(4*n_configs, 8))
    
    if n_configs == 1:
        axes = axes.reshape(2, 1)
    
    for i, result in enumerate(results_summary):
        eft = result["system"]
        name = result["name"]
        
        # Plot 1: Field and stress tensor at t=0
        ax1 = axes[0, i]
        
        ax1_twin = ax1.twinx()
        
        # Field
        line1 = ax1.plot(eft.x_vals, eft.phi[eft.Nt//2, :], 'b-', linewidth=2, label='φ(x)')
        ax1.set_xlabel("x")
        ax1.set_ylabel("φ(x)", color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        
        # T_uu
        line2 = ax1_twin.plot(eft.x_vals, eft.T_uu[eft.Nt//2, :], 'r-', linewidth=2, label='T_uu')
        ax1_twin.set_ylabel("T_uu", color='r')
        ax1_twin.tick_params(axis='y', labelcolor='r')
        
        ax1.set_title(f"{name}\nField & Stress Tensor")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: ANEC violation summary
        ax2 = axes[1, i]
        
        future_anec = result["future_anec"]["anec_values"]
        past_anec = result["past_anec"]["anec_values"]
        
        x_rays = range(len(future_anec))
        
        bars1 = ax2.bar([x - 0.2 for x in x_rays], future_anec, width=0.4, 
                       label='Future Null', alpha=0.7, color='blue')
        bars2 = ax2.bar([x + 0.2 for x in x_rays], past_anec, width=0.4, 
                       label='Past Null', alpha=0.7, color='red')
        
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
        ax2.set_xlabel("Null Ray Index")
        ax2.set_ylabel("ANEC Integral")
        ax2.set_title("ANEC Violations")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Highlight violations
        min_violation = min(np.min(future_anec), np.min(past_anec))
        if min_violation < -1e-10:
            ax2.text(0.5, 0.95, f"Max Violation:\n{min_violation:.2e}", 
                    transform=ax2.transAxes, ha='center', va='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    output_path = os.path.join(os.path.dirname(__file__), '..', 'results', 'ghost_scalar_anec.png')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   • Saved analysis plot: {output_path}")
    
    return fig

def main():
    """
    Main analysis routine for ghost scalar EFT.
    """
    # Run comprehensive analysis
    results = analyze_ghost_scalar_anec()
    
    # Generate plots
    fig = plot_ghost_analysis(results)
    
    # Summary
    print("=== Summary ===")
    print("Tested ghost scalar configurations:")
    
    max_violation_overall = 0
    best_config = None
    
    for result in results:
        violation = result["max_violation"]
        print(f"  • {result['name']}: {violation:.3e}")
        
        if violation < max_violation_overall:
            max_violation_overall = violation
            best_config = result["name"]
    
    print(f"\nBest violation: {max_violation_overall:.3e} in '{best_config}'")
    
    if max_violation_overall < -1e-10:
        print("✓ Successfully demonstrated ANEC violations in ghost scalar EFT")
        print("✓ Framework validated for controlled NEC violation studies")
    else:
        print("• No significant violations detected - may need stronger fields or different potentials")
    
    print("\n=== Ghost Scalar Analysis Complete ===")
    
    return results

if __name__ == "__main__":
    try:
        results = main()
        print(f"\nAnalysis completed successfully!")
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
