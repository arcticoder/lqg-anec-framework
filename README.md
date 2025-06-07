# LQG-ANEC Framework

A comprehensive framework for analyzing Averaged Null Energy Condition (ANEC) violations in Loop Quantum Gravity (LQG) and related quantum field theory settings.

## Overview

This framework integrates multiple theoretical approaches to study ANEC violations:
- **Loop Quantum Gravity (LQG)**: Coherent states and spin network formulations
- **Polymer Quantization**: Modified dispersion relations and quantum corrections
- **Effective Field Theory (EFT)**: Higher-order curvature corrections
- **Warp Bubble Analysis**: Negative energy requirements and stability

## Features

- **Script-based Analysis**: All computations run via Python scripts (no Jupyter notebooks)
- **CLI Interface**: Command-line driven workflows with flexible parameter scanning
- **Multiple Analysis Types**: Single-point, parameter scans, EFT analysis, and warp bubble comparisons
- **Automated Reporting**: Results saved to JSON files with optional plot generation
- **Modular Architecture**: Extensible framework for adding new analysis methods

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd lqg-anec-framework
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Command Structure

```bash
python run_lqg_anec_analysis.py --analysis-type <type> [options]
```

### Analysis Types

#### 1. Single Point Analysis
Compute ANEC violation at specific parameter values:
```bash
python run_lqg_anec_analysis.py --analysis-type single --mu 0.1 --tau 1.0 --save-plots
```

#### 2. Parameter Space Scan
Scan over ranges of polymer parameter μ and timescale τ:
```bash
python run_lqg_anec_analysis.py --analysis-type scan --mu-range "0.01,0.5,20" --tau-range "0.1,10.0,20" --save-plots
```

#### 3. Effective Field Theory Analysis
Analyze ANEC violations using EFT corrections:
```bash
python run_lqg_anec_analysis.py --analysis-type eft --n-nodes 64 --save-plots
```

#### 4. Warp Bubble Comparison
Compare LQG results with warp bubble analysis:
```bash
python run_lqg_anec_analysis.py --analysis-type warp --save-plots
```

#### 5. Comprehensive Analysis
Run all analysis types in sequence:
```bash
python run_lqg_anec_analysis.py --analysis-type full --output-dir results_comprehensive --save-plots
```

### Command Line Options

- `--analysis-type`: Type of analysis (single, scan, eft, warp, full)
- `--n-nodes`: Number of spin network nodes (default: 64)
- `--alpha`: Coherent state spread parameter (default: 0.05)
- `--mu`: Polymer parameter (default: 0.1)
- `--tau`: Sampling timescale (default: 1.0)
- `--field-amplitude`: Matter field amplitude (default: 1.0)
- `--mu-range`: Polymer parameter range for scans: "min,max,steps"
- `--tau-range`: Timescale range for scans: "min,max,steps"
- `--output-dir`: Output directory for results (default: results)
- `--save-plots`: Save analysis plots to files
4. Compute ANEC integrals and analyze violations

## Structure

```
src/
├── coherent_states.py          # Weave/heat-kernel coherent states
├── stress_tensor_operator.py   # T_ab operator definitions
├── spin_network_utils.py       # Spin network graph utilities
├── midisuperspace_model.py     # Midisuperspace Hamiltonian dynamics
├── polymer_quantization.py     # LQG polymer corrections
├── effective_action.py         # EFT derivation from spin foams
├── anec_violation_analysis.py  # ANEC integral computation
└── utils.py                    # Common utilities

notebooks/
├── anec_violation_demo.ipynb   # Interactive ANEC analysis
└── coherent_states_demo.ipynb  # Coherent state visualization

docs/
├── theory/                     # Theoretical background
└── examples/                   # Usage examples

tests/
└── test_*.py                   # Unit tests
```

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.anec_violation_analysis import main
main()
```

## Key Results

The framework enables computation of:
- Polymer-modified quantum inequality bounds
- Time-dependent T^-4 smearing effects
- ANEC violations in discrete quantum geometry
- Backreaction effects in warp-drive geometries

## Framework Architecture

### Core Modules

- **`coherent_states.py`**: LQG coherent state construction and evolution
- **`spin_network_utils.py`**: Spin network graph construction and utilities
- **`stress_tensor_operator.py`**: Stress-energy tensor computations
- **`polymer_quantization.py`**: Polymer field theory and quantum corrections
- **`midisuperspace_model.py`**: Reduced phase space quantization
- **`effective_action.py`**: Higher-order curvature corrections and EFT analysis
- **`anec_violation_analysis.py`**: Main ANEC violation computation routines

### Analysis Modules

- **`negative_energy.py`**: Negative energy density computations and Ford-Roman bounds
- **`warp_bubble_analysis.py`**: Warp bubble feasibility and stability analysis
- **`numerical_integration.py`**: Specialized integration routines
- **`field_algebra.py`**: Polymer field algebra and commutation relations

### Utility Modules

- **`metric_ansatz.py`**: Spacetime metric parametrizations
- **`bubble_stability.py`**: Stability analysis for warp configurations
- **`shape_optimizer.py`**: Variational optimization of field configurations

## Output Structure

Results are saved in the specified output directory with the following structure:

```
results/
├── analysis_summary.json          # Main results file
├── anec_parameter_scan.png        # Parameter scan visualization (if applicable)
├── single_point/                  # Single-point analysis results
├── parameter_scan/                # Parameter scan results
├── eft_analysis/                  # EFT analysis results
└── warp_comparison/               # Warp bubble comparison results
```

### Result Files

Each analysis generates:
- **JSON Summary**: Complete numerical results and parameters
- **Plots**: Visualization of parameter scans and violation regions (PNG format)
- **Analysis Metadata**: Timestamps, parameter values, and computation details

## Physical Interpretation

### Key Quantities

- **ANEC Integral**: ∫ T_μν k^μ k^ν f(λ) dλ where k^μ is a null vector and f(λ) is a sampling function
- **Classical Bound**: Standard quantum field theory lower bound on the ANEC integral
- **Polymer Corrections**: Modifications due to discrete geometric structures in LQG
- **Violation Magnitude**: Ratio of computed integral to theoretical bound

### Parameter Ranges

- **Polymer Scale μ**: 10^-2 to 10^0 (quantum bounce to Planck scale)
- **Sampling Time τ**: 0.1 to 10.0 (short to long observation timescales)
- **Coherent State Width α**: 0.01 to 0.1 (localized to spread distributions)

## Theoretical Background

This framework implements state-of-the-art methods for studying ANEC violations:

1. **Loop Quantum Gravity**: Uses polymer quantization and discrete geometric structures
2. **Coherent State Methods**: Semiclassical states that capture both quantum and classical effects
3. **Effective Field Theory**: Systematic expansion in curvature invariants
4. **Warp Bubble Physics**: Alcubierre-style spacetimes requiring negative energy

### Key References

- Ashtekar, A. & Singh, P. "Loop Quantum Cosmology" (2011)
- Alcubierre, M. "The Warp Drive: Hyper-fast Travel Within General Relativity" (1994)
- Ford, L.H. & Roman, T.A. "Quantum Field Theory Constrains Traversable Wormhole Geometries" (1995)

## Development

### Adding New Analysis Methods

1. Create a new module in `src/`
2. Implement analysis functions with standard signatures
3. Add CLI integration in `run_lqg_anec_analysis.py`
4. Update this README with usage examples

### Testing

Run basic functionality tests:
```bash
python run_lqg_anec_analysis.py --analysis-type single --mu 0.1 --tau 1.0
```

### Dependencies

- `numpy`: Numerical computations
- `scipy`: Scientific computing and integration
- `matplotlib`: Plotting and visualization
- `networkx`: Graph structures for spin networks
- `argparse`: Command-line interface

## Recent Analysis Results

The framework has been successfully tested with the following capabilities:

✅ **Single-point ANEC analysis**: Computes classical and quantum-corrected ANEC integrals  
✅ **Parameter space scanning**: Maps violation regions across μ-τ parameter space  
✅ **EFT analysis**: Incorporates higher-order curvature corrections  
✅ **Warp bubble integration**: Compares LQG results with negative energy requirements  
✅ **Automated reporting**: Saves results to JSON and generates plots  
✅ **Polymer scale hierarchy**: Covers phenomenological to Planck scale physics  

### Example Output

Recent full analysis run shows:
- ANEC integral computed for multiple polymer scales
- Parameter optimization identifies optimal violation regimes
- EFT coefficients quantify curvature correction importance
- Stability analysis confirms polymer enhancement effects
- All results saved to structured file hierarchy

## License

[Add appropriate license information]

## Contributors

[Add contributor information]

## Citation

If you use this framework in your research, please cite:

```
@software{lqg_anec_framework,
  title={LQG-ANEC Framework: Comprehensive ANEC Violation Analysis},
  author={[Author Names]},
  year={2025},
  url={[Repository URL]}
}
```
