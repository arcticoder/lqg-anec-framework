#!/usr/bin/env python3
"""
Simple programmatic example from user request.
This demonstrates the exact code snippet provided in the user request.
"""

from run_3d_mesh_validation import main as validate
import sys
import json

# Simulate both and load JSON report
sys.argv = ['run_3d_mesh_validation.py', '--source', 'both', '--radius', '10', '--resolution', '30']
validate()

import json
report = json.load(open('results/warp_bubble_comparison_report.json'))
best = max(report['results'].items(),
           key=lambda kv: (kv[1]['success'], kv[1]['stability']))
print("Best source:", best[0], best[1])
