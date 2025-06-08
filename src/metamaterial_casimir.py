# src/metamaterial_casimir.py

import numpy as np
from scipy.constants import hbar, c, pi
from scipy.integrate import quad
from typing import List, Dict, Optional, Tuple

# Import base classes (avoiding circular imports)
try:
    from vacuum_engineering import CasimirArray, MATERIAL_DATABASE
except ImportError:
    # Define minimal base class if vacuum_engineering not available
    class CasimirArray:
        def __init__(self, temperature=300.0):
            self.T = temperature
            self.c = c
            self.hbar = hbar
        
        def thermal_casimir_pressure(self, a, material_perm):
            return -(pi**2 * hbar * c) / (240 * a**4) * np.real(material_perm)
    
    MATERIAL_DATABASE = {}

try:
    from drude_model import DrudeLorentzPermittivity, casimir_integrand_with_dispersion
    DRUDE_AVAILABLE = True
except ImportError:
    DRUDE_AVAILABLE = False

class MetamaterialCasimir(CasimirArray):
    """
    Casimir array with alternating negative-index metamaterial layers.
    
    Implements enhanced Casimir forces through:
    - Negative permittivity/permeability materials
    - Alternating layer configurations 
    - Frequency-dependent material response
    - Super-Casimir force amplification
    """
    def __init__(self, spacings: List[float], eps_list: List[complex], 
                 mu_list: List[complex], temperature: float = 300.0):
        """
        Initialize metamaterial Casimir system.
        
        Args:
            spacings: List of layer spacings [m]
            eps_list: List of complex permittivities
            mu_list: List of complex permeabilities  
            temperature: Operating temperature [K]
        """
        super().__init__(temperature)
        self.spacings = np.array(spacings)
        self.eps = np.array(eps_list)
        self.mu = np.array(mu_list)
        
        # Validate inputs
        if len(spacings) != len(eps_list) or len(eps_list) != len(mu_list):
            raise ValueError("Spacing, permittivity, and permeability lists must have same length")
            
        self.n_layers = len(spacings)
        
        print(f"Metamaterial Casimir Array initialized:")
        print(f"  Number of layers: {self.n_layers}")
        print(f"  Spacing range: {np.min(spacings)*1e9:.1f} - {np.max(spacings)*1e9:.1f} nm")
        print(f"  Permittivity range: {np.min(np.real(eps_list)):.2f} - {np.max(np.real(eps_list)):.2f}")
        print(f"  Permeability range: {np.min(np.real(mu_list)):.2f} - {np.max(np.real(mu_list)):.2f}")

    def refractive_index(self, layer_idx: int) -> complex:
        """
        Compute effective refractive index for layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Complex refractive index n = √(εμ)
        """
        return np.sqrt(self.eps[layer_idx] * self.mu[layer_idx])

    def is_negative_index(self, layer_idx: int) -> bool:
        """
        Check if layer has negative refractive index.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            True if both ε and μ have negative real parts
        """
        return (np.real(self.eps[layer_idx]) < 0 and 
                np.real(self.mu[layer_idx]) < 0)

    def layer_energy_density(self, layer_idx: int, include_dispersion: bool = False) -> float:
        """
        Compute Casimir energy density for single layer with metamaterial corrections.
        
        Args:
            layer_idx: Layer index
            include_dispersion: Whether to include frequency dispersion
            
        Returns:
            Energy density [J/m³]
        """
        a = self.spacings[layer_idx]
        eps = self.eps[layer_idx]  
        mu = self.mu[layer_idx]
        
        if include_dispersion:
            # Use frequency-dependent calculation (placeholder - would need full Lifshitz formula)
            base_density = -(pi**2 * hbar * c) / (720 * a**4)
            
            # Metamaterial enhancement factor
            n_eff = self.refractive_index(layer_idx)
            
            if self.is_negative_index(layer_idx):
                # Negative index can reverse sign and amplify
                enhancement = -np.abs(n_eff)**3  # Cubic scaling for energy density
            else:
                enhancement = np.real(n_eff)**2
                
            return base_density * enhancement
        else:
            # Simple model without full dispersion
            return self.thermal_casimir_pressure(a, eps) * a  # Convert pressure to density

    def energy_density(self):
        """
        Enhanced energy density with metamaterial corrections.
        Simple model: density ∝ 1/(ε*μ) with sign corrections for negative-index materials.
        """
        base_densities = []
        
        for i in range(self.n_layers):
            # Base Casimir energy density
            a = self.spacings[i]
            base = -(pi**2 * hbar * c) / (720 * a**4)
            
            # Metamaterial correction
            eps = self.eps[i]
            mu = self.mu[i]
            
            # Material enhancement factor
            n_eff = np.sqrt(eps * mu)
            if np.real(n_eff) < 0:
                # Negative index - can amplify force significantly
                correction = np.abs(n_eff)**2
            else:
                # Normal materials
                correction = np.real(n_eff)
                
            # Apply metamaterial enhancement
            layer_density = base * correction / (eps * mu)
            base_densities.append(layer_density)
            
        return np.array(base_densities)
    
    def total_energy_density(self, include_dispersion: bool = False) -> float:
        """
        Compute total energy density from all layers.
        
        Args:
            include_dispersion: Whether to include frequency dispersion
            
        Returns:
            Total energy density [J/m³]
        """
        total = 0.0
        for i in range(self.n_layers):
            layer_density = self.layer_energy_density(i, include_dispersion)
            total += layer_density
        
        return total

    def force_amplification_factor(self) -> float:
        """
        Compute force amplification compared to vacuum Casimir.
        
        Returns:
            Amplification factor (>1 means enhanced force)
        """
        # Reference vacuum Casimir energy density
        ref_spacing = np.mean(self.spacings)
        vacuum_density = -(pi**2 * hbar * c) / (720 * ref_spacing**4)
        
        total_density = self.total_energy_density()
        
        return abs(total_density / vacuum_density)

    def optimize_metamaterial_stack(self, target_density: float, 
                                  n_layers: int = 10) -> Dict:
        """
        Optimize metamaterial stack configuration for target energy density.
        
        Args:
            target_density: Target negative energy density [J/m³]
            n_layers: Number of layers to optimize
            
        Returns:
            Optimization result dictionary
        """
        from scipy.optimize import differential_evolution
        
        def objective(params):
            # params = [spacings..., eps_real..., eps_imag..., mu_real..., mu_imag...]
            n = n_layers
            spacings = params[:n] * 1e-9  # Convert to meters
            eps_real = params[n:2*n]
            eps_imag = params[2*n:3*n] 
            mu_real = params[3*n:4*n]
            mu_imag = params[4*n:5*n]
            
            eps_list = eps_real + 1j*eps_imag
            mu_list = mu_real + 1j*mu_imag
            
            try:
                # Create temporary metamaterial system
                temp_system = MetamaterialCasimir(spacings, eps_list, mu_list, self.T)
                density = temp_system.total_energy_density()
                return abs(density - target_density)
            except:
                return 1e10  # Penalty for invalid configurations
        
        # Parameter bounds: [spacing_nm, eps_real, eps_imag, mu_real, mu_imag]
        bounds = (
            [(10, 1000)] * n_layers +     # Spacings in nm
            [(-10, 10)] * n_layers +      # eps real
            [(0, 5)] * n_layers +         # eps imag  
            [(-10, 10)] * n_layers +      # mu real
            [(0, 5)] * n_layers           # mu imag
        )
        
        result = differential_evolution(objective, bounds, seed=42, maxiter=100)
        
        if result.success:
            n = n_layers
            opt_spacings = result.x[:n] * 1e-9
            opt_eps = result.x[n:2*n] + 1j*result.x[2*n:3*n]
            opt_mu = result.x[3*n:4*n] + 1j*result.x[4*n:5*n]
            
            opt_system = MetamaterialCasimir(opt_spacings, opt_eps, opt_mu, self.T)
            achieved_density = opt_system.total_energy_density()
            
            return {
                'success': True,
                'spacings': opt_spacings,
                'permittivities': opt_eps,
                'permeabilities': opt_mu, 
                'achieved_density': achieved_density,
                'target_density': target_density,
                'error': result.fun,
                'amplification_factor': opt_system.force_amplification_factor()
            }
        else:
            return {'success': False, 'message': 'Optimization failed'}

class AlternatingMetamaterialStack(MetamaterialCasimir):
    """
    Specialized metamaterial stack with alternating positive/negative index layers.
    
    This configuration can produce constructive interference effects
    and enhanced Casimir forces.
    """
    
    def __init__(self, base_spacing: float, n_pairs: int, 
                 pos_material: Dict, neg_material: Dict, temperature: float = 300.0):
        """
        Initialize alternating stack.
        
        Args:
            base_spacing: Base layer spacing [m]
            n_pairs: Number of positive/negative pairs
            pos_material: {'eps': complex, 'mu': complex} for positive index
            neg_material: {'eps': complex, 'mu': complex} for negative index  
            temperature: Operating temperature [K]
        """
        # Create alternating pattern
        spacings = [base_spacing] * (2 * n_pairs)
        eps_list = []
        mu_list = []
        
        for i in range(n_pairs):
            # Positive index layer
            eps_list.append(pos_material['eps'])
            mu_list.append(pos_material['mu'])
            # Negative index layer  
            eps_list.append(neg_material['eps'])
            mu_list.append(neg_material['mu'])
            
        super().__init__(spacings, eps_list, mu_list, temperature)
        
        self.n_pairs = n_pairs
        self.pos_material = pos_material
        self.neg_material = neg_material
        
        print(f"Alternating metamaterial stack:")
        print(f"  Number of pairs: {n_pairs}")
        print(f"  Positive index: ε={pos_material['eps']:.2f}, μ={pos_material['mu']:.2f}")
        print(f"  Negative index: ε={neg_material['eps']:.2f}, μ={neg_material['mu']:.2f}")

    def interference_factor(self) -> complex:
        """
        Compute interference factor from alternating structure.
        
        Returns:
            Complex interference factor
        """
        # Simplified model - full calculation would require transfer matrix method
        n_pos = np.sqrt(self.pos_material['eps'] * self.pos_material['mu'])
        n_neg = np.sqrt(self.neg_material['eps'] * self.neg_material['mu'])
        
        # Phase difference between layers
        k_pos = 2*pi*n_pos/self.spacings[0]  
        k_neg = 2*pi*n_neg/self.spacings[0]
        
        # Constructive/destructive interference
        phase_diff = (k_pos - k_neg) * self.spacings[0]
        
        return np.exp(1j * phase_diff * self.n_pairs)

def create_optimized_metamaterial_casimir(target_enhancement: float = 100) -> MetamaterialCasimir:
    """
    Create optimized metamaterial Casimir system for maximum enhancement.
    
    Args:
        target_enhancement: Target force enhancement factor
        
    Returns:
        Optimized MetamaterialCasimir instance
    """
    # Start with known good metamaterial parameters
    base_spacing = 50e-9  # 50 nm
    n_layers = 20
    
    # Negative index metamaterial (typical values)
    eps_neg = -2.1 + 0.1j
    mu_neg = -1.2 + 0.05j
    
    # Conventional dielectric  
    eps_pos = 2.25 + 0.01j  # Fused silica
    mu_pos = 1.0 + 0.0j
    
    # Alternating pattern
    spacings = [base_spacing] * n_layers
    eps_list = [eps_neg if i%2==0 else eps_pos for i in range(n_layers)]
    mu_list = [mu_neg if i%2==0 else mu_pos for i in range(n_layers)]
    
    return MetamaterialCasimir(spacings, eps_list, mu_list)

if __name__ == "__main__":
    # Test metamaterial Casimir enhancement
    print("Testing Metamaterial Casimir Systems")
    print("=" * 40)
    
    # Test basic metamaterial system
    spacings = [50e-9] * 10  # 10 layers, 50 nm each
    eps_list = [(-2.0 + 0.1j) if i%2==0 else (2.25 + 0.01j) for i in range(10)]
    mu_list = [(-1.5 + 0.05j) if i%2==0 else (1.0 + 0.0j) for i in range(10)]
    
    meta_casimir = MetamaterialCasimir(spacings, eps_list, mu_list)
    
    print(f"\nBasic metamaterial system:")
    print(f"Total energy density: {meta_casimir.total_energy_density():.2e} J/m³")
    print(f"Force amplification: {meta_casimir.force_amplification_factor():.1f}x")
    
    # Test alternating stack
    pos_mat = {'eps': 2.25 + 0.01j, 'mu': 1.0 + 0.0j}
    neg_mat = {'eps': -2.0 + 0.1j, 'mu': -1.5 + 0.05j}
    
    alt_stack = AlternatingMetamaterialStack(
        base_spacing=50e-9, 
        n_pairs=5,
        pos_material=pos_mat,
        neg_material=neg_mat
    )
    
    print(f"\nAlternating stack:")
    print(f"Total energy density: {alt_stack.total_energy_density():.2e} J/m³") 
    print(f"Force amplification: {alt_stack.force_amplification_factor():.1f}x")
    print(f"Interference factor: {abs(alt_stack.interference_factor()):.2f}")
    
    # Test optimization
    print(f"\nTesting optimization...")
    target_density = -1e-10  # Target: -100 pJ/m³
    
    opt_result = meta_casimir.optimize_metamaterial_stack(target_density, n_layers=5)
    
    if opt_result['success']:
        print(f"Optimization successful!")
        print(f"Target density: {target_density:.2e} J/m³")
        print(f"Achieved density: {opt_result['achieved_density']:.2e} J/m³") 
        print(f"Amplification factor: {opt_result['amplification_factor']:.1f}x")
    else:
        print(f"Optimization failed: {opt_result.get('message', 'Unknown error')}")
    
    print(f"\nMetamaterial Casimir module ready!")
