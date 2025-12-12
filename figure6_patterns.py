import numpy as np
from tissue_deap import run_tissue_simulation

# Baseline and evolved params
baseline = [1.0, 1.0, 1.0]
evolved  = [1.6, 0.27, 1.7]  # from BEST PARAMS

baseline_pattern = run_tissue_simulation(baseline)
evolved_pattern  = run_tissue_simulation(evolved)

np.save("figure6_baseline_pattern.npy", baseline_pattern)
np.save("figure6_evolved_pattern.npy", evolved_pattern)

print("Saved figure6_baseline_pattern.npy and figure6_evolved_pattern.npy")
