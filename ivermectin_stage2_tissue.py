import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def bioelectric_model(y, t, k1, k2, k3):
    X, V = y
    dXdt = k1 * V - k2 * X * V
    dVdt = -k3 * V
    return [dXdt, dVdt]

# Tissue simulation parameters
N = 10  # 10x10 grid
t = np.linspace(0, 100, 2000)  # Longer time for tissue convergence
y0 = [1, 1]

# Stage 2: Fixed ivermectin-like perturbation (2x Cl conductance on left half)
ivm_factor = 2.0  # From Stage 1 dose-response sweet spot

# Simulate tissue: left 5 columns = ivermectin, right 5 = baseline
tissue_patterns = np.zeros((N, N))

for i in range(N):
    for j in range(N):
        # Left half: ivermectin (hyperpolarizing)
        if j < 5:
            k3_ivm = 0.01 * ivm_factor
        else:
            k3_ivm = 0.01  # Baseline
            
        sol = odeint(bioelectric_model, y0, t, args=(1, 0.1, k3_ivm))
        tissue_patterns[i, j] = sol[-1, 1]  # Steady-state Vnorm

# Save data
np.save('ivermectin_stage2_tissue_pattern.npy', tissue_patterns)

# Plot (Figure 3 style)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Baseline (all k3=0.01)
baseline = np.ones((N, N)) * 0.626  # From your Stage 1 baseline
im1 = ax1.imshow(baseline, cmap='Blues', vmin=0, vmax=1)
ax1.set_title('Untreated tissue')
ax1.set_ylabel('Normalized Vnorm (Vnorm)')
plt.colorbar(im1, ax=ax1)

# Ivermectin perturbed
im2 = ax2.imshow(tissue_patterns, cmap='coolwarm', vmin=0, vmax=1)
ax2.set_title(f'Ivermectin-like perturbation\n(×{ivm_factor} Cl conductance, left half)')
ax2.set_ylabel('Row')
plt.colorbar(im2, ax=ax2)

plt.tight_layout()
plt.savefig('ivermectin_stage2_tissue.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"✅ Ivermectin Stage 2 complete:")
print(f"   → ivermectin_stage2_tissue_pattern.npy")
print(f"   → ivermectin_stage2_tissue.png")
