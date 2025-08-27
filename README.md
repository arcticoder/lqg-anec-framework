# LQG-ANEC Framework (research-stage)

## Related Repositories

- [energy](https://github.com/arcticoder/energy): Central meta-repo for all energy, quantum, and LQG research. This ANEC framework is integrated for energy condition analysis.
- [lqg-cosmological-constant-predictor](https://github.com/arcticoder/lqg-cosmological-constant-predictor): Shares theoretical and simulation infrastructure for LQG and cosmological analysis.
- [unified-lqg](https://github.com/arcticoder/unified-lqg): Provides core LQG framework for coherent states and spin network formulations.
- [warp-bubble-qft](https://github.com/arcticoder/warp-bubble-qft): Related for negative energy requirements and warp bubble stability analysis.
- [negative-energy-generator](https://github.com/arcticoder/negative-energy-generator): Shares ANEC violation analysis for negative energy generation applications.

All repositories are part of the [arcticoder](https://github.com/arcticoder) ecosystem and link back to the energy framework for unified documentation and integration.

This repository provides research-stage tools for exploring Averaged Null Energy Condition (ANEC) violations in Loop Quantum Gravity (LQG) and related quantum field theory settings. Results presented here are model-derived and intended for research and reproducibility; they require independent validation and V&V before any operational interpretation.

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
-- **Automated Reporting (research-stage)**: Results saved to JSON files with optional plot generation. Reported metrics should be interpreted as outputs of the implemented analyses and are subject to the assumptions and parameter choices documented in `docs/`.
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

#### 5. Analysis
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

Reported / provisional computational results (see `docs/` for methods, raw artifacts, and UQ plans):
- Polymer-modified quantum inequality (QI) estimates (model outputs)
- Observed T^-4-like smearing effects in selected parameterizations
- Indications of ANEC violation regions within the discrete-geometry model assumptions
- Preliminary backreaction estimates for simplified warp configurations

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
- **JSON Summary**: Numerical results and parameters
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

This framework implements methods for studying ANEC violations:

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

## Verification & Validation  
  
### Metric generation for billiard-ball–size warp bubble  
  
This V&V task (tracked in `VnV-TODO.ndson:1`) ensures that the warp-bubble metric builder implements the analytic soliton profile correctly README:  
  
- [`scripts/gen_reference_soliton.py`](scripts/gen_reference_soliton.py): generates an independent reference file `tests/reference_soliton.json` by evaluating

```math
    f(r)=c_0 + a\tanh\bigl[b,(r-x_0)\bigr]
```
at r=0

- [`tests/reference_soliton.json`](tests/reference_soliton.json): “golden” dataset mapping `(a,b,x0,c0)` tuples to expected values.  
- [`tests/test_metric_generation.py`](tests/test_metric_generation.py): loads the golden dataset, prints

    `=== Soliton Ansatz vs Reference Dataset ===`

    followed by each `key: result, expected`, and asserts the symbolic `soliton_ansatz` at `r=0` matches the reference within a relative tolerance of $10^{-8}$.  
- **CI workflow**: runs this test suite on every push via `.github/workflows/metric-generation.yml`.

## Analysis Results (June 2025)

The LQG-ANEC Framework has produced computational validation of quantum inequality analysis:


### Computational Results (reported / model-derived)
- **Large-scale sweep outcomes**: Model sweeps produced many candidate QI-violation events under the chosen parameter ranges; counts are sensitive to sampling strategy and post-processing filters (see `docs/` and `artifacts/`).
- **Performance metrics**: GPU utilization figures reflect the hardware and runtime configuration used in experiments; treat them as reproducibility notes rather than fixed performance guarantees.
- **Sustained negative-energy scenarios**: Some model configurations produce prolonged negative-energy flux in simulations; these are theoretical outputs that require rigorous physical interpretation and independent verification.
- **Field-configuration sensitivity**: Violation rates vary strongly with model choices and numerical tolerances; see `docs/UQ-TODO.ndjson` for recommended sensitivity analyses.

### Theoretical Results
- **Polymer-Enhanced Field Theory (model-derived):** Mathematical framework with week-scale modulation. Results are contingent on approximation choices and numerical methods; document assumptions and limitations in `docs/`.
- **Five QI Kernel Validation (methodology):** Validation methodology for different sampling approaches; include convergence tests and reproduce kernels in `docs/` when publishing claims.
- **Ghost Scalar EFT (theoretical observation):** Reported UV-finite negative-energy behavior is a model-derived observation. Verify analytically and numerically and include limitations and parameter-dependence in accompanying documentation.
- **Systematic ANEC Violation (provisional numbers):** Previously reported extreme numerical values are model outputs sensitive to discretization and sampling; provide raw artifacts, parameter sweeps, and uncertainty quantification before drawing physical conclusions.

### Documentation
Technical Documentation: See `docs/key_discoveries.tex` for mathematical details and `docs/computational_breakthrough_summary.tex` for performance analysis.

## Framework Capabilities (research-stage / validated on test data)

The repository implements analysis routines for the following research tasks. Where capabilities are listed, they indicate implemented features and test-scale validation; they are not guarantees of operational performance.

- **Single-point ANEC analysis**: Computes classical and quantum-corrected ANEC integrals for configured parameter sets. Validate with `tests/` and analytic checks before relying on outputs.
- **Parameter space scanning**: Supports parameter sweeps and result aggregation. Interpret maps as exploratory guides; attach uncertainty summaries for any claim derived from them.
- **EFT analysis**: Tools for incorporating higher-order curvature corrections; document assumptions in `docs/` for each run.
- **Warp bubble comparison**: Utilities to compare simplified warp metrics with LQG outputs; these comparisons are conceptual and require domain review for physical claims.
- **Automated reporting**: Saves structured JSON artifacts and plotting scripts to `results/` or `artifacts/`; include seeds and environment metadata to reproduce runs.
- **Polymer scale exploration**: Enables sweeps across polymer scales; link any phenomenological claims to UQ artifacts.
- **GPU-optimized workflows**: Experimental GPU acceleration paths included; profile and validate on your hardware.
- **Extended temporal sampling**: Supports long-duration integration for research experiments; results should include sensitivity checks and convergence diagnostics.
## Scope / Validation & Limitations

- **Maturity**: Research-stage code and analyses. Results are provisional and depend on model choices, numerical tolerances, and sampling strategies.
- **Reproducibility**: To reproduce reported outputs, run the scripts with the specified seeds and environment details; raw artifacts and run metadata should be published alongside any claim.
- **Uncertainty Quantification**: Add sensitivity sweeps and bootstrap/resampling where possible; consult `docs/UQ-TODO.ndjson` for suggested tasks.
- **Independent review**: Any claim with operational or physical implications should be reviewed by domain experts and accompanied by V&V artifacts.
- **Safety & communications**: Avoid presenting simulation outputs as operational capabilities in public summaries; use conservative phrasing and include limitations.

### Example Output

Representative analysis outputs (model/simulation artifacts) — interpret with caution and publish reproducibility artifacts alongside:
- **Counts of candidate QI-violation events:** Large counts can appear in parameter sweeps; these counts depend strongly on sampling and filtering choices. When reporting counts, publish the raw data, filtering scripts, and parameter settings.
- **Reported negative-energy flux cases:** Some simulations show prolonged negative-energy flux under certain model assumptions; these are theoretical outputs and require independent verification and physical interpretation by domain experts.
- **Field-configuration sensitivity:** Violation rates vary with model choices and numerical tolerances; include sensitivity analyses and CI-like summaries in `docs/`.
- **Performance & profiling:** GPU utilization and other profiling numbers are reproducibility notes tied to specific hardware and runtime configurations.
- **Repro artifacts:** Provide a short reproducible script/notebook and raw artifacts (seeds, environment, parameter files) that reproduce a representative result.

## License

The Unlicense

## Contributors

[Add contributor information]
