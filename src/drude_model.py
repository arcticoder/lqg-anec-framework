# src/drude_model.py

import numpy as np
from scipy.constants import c, epsilon_0, pi

class DrudeLorentzPermittivity:
    """
    Frequency-dependent permittivity using Drude-Lorentz model:
    ε(ω) = 1 - ω_p^2 / (ω^2 + iγω) + Σ_j f_j ω_pj^2 / (ω_j^2 - ω^2 - i γ_j ω)
    
    This provides realistic material response for Casimir calculations
    with proper frequency dispersion and losses.
    """
    def __init__(self, ωp, γ, oscillators=None):
        """
        Initialize Drude-Lorentz permittivity model.
        
        :param ωp: plasma frequency [rad/s]
        :param γ: collision rate [rad/s]
        :param oscillators: list of (fj, ωpj, ωj, γj) tuples for bound oscillators
        """
        self.ωp = ωp
        self.γ  = γ
        self.osc = oscillators or []

    def ε(self, ω):
        """
        Compute complex permittivity at frequency ω.
        
        :param ω: frequency [rad/s], can be array
        :return: complex permittivity ε(ω)
        """
        # Ensure ω is array for vectorized operations
        ω = np.asarray(ω)
        
        # Drude term (free electrons)
        term = 1 - self.ωp**2/(ω*(ω + 1j*self.γ))
        
        # Lorentz oscillator terms (bound electrons)
        for fj, ωpj, ωj, γj in self.osc:
            term += fj * ωpj**2 / (ωj**2 - ω**2 - 1j*γj*ω)
        return term

    def reflectivity(self, ω):
        """
        Simple normal-incidence reflectivity for a half-space:
        R = |(√ε - 1)/(√ε + 1)|^2
        
        :param ω: frequency [rad/s]
        :return: reflectivity R(ω)
        """
        eps = self.ε(ω)
        n   = np.sqrt(eps)
        return np.abs((n - 1)/(n + 1))**2

    def transmission(self, ω, thickness):
        """
        Transmission through finite thickness slab.
        
        :param ω: frequency [rad/s]
        :param thickness: slab thickness [m]
        :return: transmission coefficient
        """
        eps = self.ε(ω)
        n = np.sqrt(eps)
        k = ω * n / c
        
        # Simple Beer's law with reflection losses
        r = self.reflectivity(ω)
        t_bulk = np.exp(-np.imag(k) * thickness)
        
        return (1 - r)**2 * t_bulk / (1 - r**2 * t_bulk**2)

# Predefined material models
MATERIAL_MODELS = {
    'gold': DrudeLorentzPermittivity(
        ωp=1.36e16,  # Gold plasma frequency
        γ=1.45e14,   # Gold collision rate
        oscillators=[
            (0.76, 2.3e15, 2.4e15, 1.0e14),  # Interband transition
            (0.024, 2.8e15, 2.9e15, 5.0e13)  # Second transition
        ]
    ),
    
    'aluminum': DrudeLorentzPermittivity(
        ωp=2.24e16,  # Aluminum plasma frequency
        γ=1.22e14,   # Aluminum collision rate
        oscillators=[
            (0.523, 1.5e15, 1.6e15, 2.4e14)  # Interband transition
        ]
    ),
    
    'silver': DrudeLorentzPermittivity(
        ωp=1.38e16,  # Silver plasma frequency
        γ=2.73e13,   # Silver collision rate (lower loss than gold)
        oscillators=[
            (0.845, 6.5e15, 6.7e15, 9.6e14)  # Interband transition
        ]
    ),
    
    'silicon': DrudeLorentzPermittivity(
        ωp=0,  # No free carriers (intrinsic)
        γ=0,
        oscillators=[
            (11.68, 0, 3.4e15, 1e13)  # Simplified band gap model
        ]
    )
}

def get_material_model(material_name):
    """
    Get predefined material model by name.
    
    :param material_name: material identifier
    :return: DrudeLorentzPermittivity instance
    """
    if material_name.lower() in MATERIAL_MODELS:
        return MATERIAL_MODELS[material_name.lower()]
    else:
        raise ValueError(f"Unknown material: {material_name}")

def casimir_integrand_with_dispersion(ω, a, material_model):
    """
    Casimir force integrand with frequency-dependent permittivity.
    
    :param ω: frequency [rad/s]
    :param a: plate separation [m]
    :param material_model: DrudeLorentzPermittivity instance
    :return: integrand value
    """
    from scipy.constants import hbar
    
    # Get material response
    eps = material_model.ε(ω)
    R = material_model.reflectivity(ω)
    
    # Casimir integrand with material correction
    # ∝ ω³ R(ω) with geometric factors
    return (hbar * ω**3 * R) / (2 * pi**2 * c**3)

if __name__ == "__main__":
    # Test material models
    import matplotlib.pyplot as plt
    
    # Frequency range for testing
    ω_range = np.logspace(13, 17, 1000)  # 10 THz to 100 PHz
    
    plt.figure(figsize=(12, 8))
    
    # Plot permittivity for different materials
    for name, model in MATERIAL_MODELS.items():
        eps = model.ε(ω_range)
        
        plt.subplot(2, 2, 1)
        plt.loglog(ω_range/(2*pi*1e12), np.real(eps), label=f'{name} (real)')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Re[ε]')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.loglog(ω_range/(2*pi*1e12), np.abs(np.imag(eps)), label=f'{name} (imag)')
        plt.xlabel('Frequency (THz)')
        plt.ylabel('|Im[ε]|')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        R = model.reflectivity(ω_range)
        plt.semilogx(ω_range/(2*pi*1e12), R, label=name)
        plt.xlabel('Frequency (THz)')
        plt.ylabel('Reflectivity')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('results/material_dispersion_models.png', dpi=150)
    print("Material dispersion models saved to results/material_dispersion_models.png")
    
    # Test Casimir integrand
    gold_model = get_material_model('gold')
    a = 100e-9  # 100 nm gap
    
    casimir_vals = [casimir_integrand_with_dispersion(ω, a, gold_model) for ω in ω_range]
    
    plt.figure(figsize=(10, 6))
    plt.loglog(ω_range/(2*pi*1e12), casimir_vals)
    plt.xlabel('Frequency (THz)')
    plt.ylabel('Casimir integrand')
    plt.title(f'Gold Casimir integrand, a = {a*1e9:.0f} nm')
    plt.grid(True)
    plt.savefig('results/casimir_integrand_dispersion.png', dpi=150)
    print("Casimir integrand with dispersion saved to results/casimir_integrand_dispersion.png")
