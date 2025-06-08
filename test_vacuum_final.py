#!/usr/bin/env python3

from src.vacuum_engineering import *
print('Testing vacuum engineering module...')

# Test comprehensive vacuum analysis
print('Running comprehensive analysis...')
analysis = comprehensive_vacuum_analysis()

print('Results:')
for method, data in analysis.items():
    print(f'{method}: {data["target_ratio"]:.2e} target ratio, feasible: {data["feasible"]}')

# Test backward compatibility
print('\nTesting backward compatibility...')
legacy = build_lab_sources_legacy()
print(f'Legacy sources: {list(legacy.keys())}')
for name, source in legacy.items():
    print(f'{name}: {source.total_density():.2e} J/m³')

print('\nVacuum engineering module test completed successfully!')
