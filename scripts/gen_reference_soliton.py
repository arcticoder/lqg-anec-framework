#!/usr/bin/env python3
import json
import math
import os

# Define your test points: tuples of (a, b, x0, c0)
points = [
    (1.0, 1.0, 0.0, 0.0),
    (2.0, 0.5, 1.0, 3.0),
    (1.0, 2.0, 0.0, 4.0),
]

# Choose a fixed radius at which to evaluate (e.g. r0 = 0)
r0 = 0.0

# Analytic soliton profile: f(r) = c0 + a * tanh[b*(r - x0)]
def soliton_profile(r, a, b, x0, c0):
    return c0 + a * math.tanh(b * (r - x0))

# Build the reference dict
ref = {}
for a, b, x0, c0 in points:
    val = soliton_profile(r0, a, b, x0, c0)
    key = f"{a},{b},{x0},{c0}"
    ref[key] = val

# Write out to tests/reference_soliton.json
out_path = os.path.join("tests", "reference_soliton.json")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
with open(out_path, "w") as f:
    json.dump(ref, f, indent=2)

print(f"Written {len(ref)} entries to {out_path}")