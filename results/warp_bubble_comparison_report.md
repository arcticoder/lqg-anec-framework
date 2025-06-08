# 3D Mesh-Based Warp Bubble Validation Report

Generated: 2025-06-07 21:46:29

## Summary

- Sources tested: 2
- Successful sources: 1
- Success rate: 50.0%

## Results by Source

### Metamaterial Casimir

- **Success**: False
- **Total Energy**: -3.14e-02 J
- **Stability**: 0.000
- **Max Negative Density**: -2.97e-04 J/m³
- **Execution Time**: 0.006 s
- **Mesh Nodes**: 3600
- **Parameters**: {'epsilon': -2.0, 'mu': -1.5, 'cell_size': 5e-08, 'n_layers': 100, 'R0': 5.0, 'shell_thickness': 0.5}

### Ghost/Phantom EFT

- **Success**: True
- **Total Energy**: -2.23e-13 J
- **Stability**: 0.997
- **Max Negative Density**: -1.37e-15 J/m³
- **Execution Time**: 0.005 s
- **Mesh Nodes**: 3600
- **Parameters**: {'M': 1000, 'alpha': 0.01, 'beta': 0.1, 'R0': 5.0, 'sigma': 0.2}

