import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def bioelectric_model(y, t, k1, k2, k3):
    X, V = y
    return [k1 * V - k2 * X * V, -k3 * V]

# Tissue: 100 cells with simple 1D spatial coupling
N_cells = 100
t = np.linspace(0, 50, 500)
dt = t[1] - t[0]

# Cell states: [X, V] for each cell
cells_baseline = np.ones((N_cells, 2))  # all start at X=1, V=1
cells_drug = np.ones((N_cells, 2))      # drug will hit cells 0–49

# Gap junction–like spatial coupling strength
coupling = 0.1

# Time stepping for a simple tissue approximation
for i in range(len(t)):
    # --- Untreated tissue (no drug anywhere) ---
    dV_base = bioelectric_model(cells_baseline[-1, :], t[i], 1, 0.1, 0.01)[1]
    for j in range(N_cells):
        neighbor_avg = np.mean(cells_baseline[max(0, j-1):min(N_cells, j+2), 1])
        cells_baseline[j, 1] += dt * (
            dV_base + coupling * (neighbor_avg - cells_baseline[j, 1])
        )

    # --- Drug tissue (amiloride hits half the cells) ---
    k3_drug = 0.001  # effective k3 under drug (stronger block)
    dV_drug_global = bioelectric_model(cells_drug[-1, :], t[i], 1, 0.1, k3_drug)[1]
    for j in range(N_cells):
        neighbor_avg = np.mean(cells_drug[max(0, j-1):min(N_cells, j+2), 1])

        if j < N_cells // 2:
            # First 50 cells: "drugged" with reduced k3
            dV_local = dV_drug_global
        else:
            # Remaining cells: untreated with baseline k3
            dV_local = bioelectric_model(cells_drug[j, :], t[i], 1, 0.1, 0.01)[1]

        cells_drug[j, 1] += dt * (
            dV_local + coupling * (neighbor_avg - cells_drug[j, 1])
        )

# Take final tissue voltages and reshape into a 10x10 grid
# Use column-major order so the first 50 cells occupy the left half of the image
V_baseline = cells_baseline[:, 1].reshape(10, 10, order='F')
V_drug = cells_drug[:, 1].reshape(10, 10, order='F')

# Plot tissue voltage maps
plt.figure(figsize=(10, 4))

# Panel A: untreated tissue
ax1 = plt.subplot(1, 2, 1)
im1 = ax1.imshow(V_baseline, cmap='RdBu_r', vmin=-1, vmax=1)
ax1.set_title('Untreated tissue')
ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_xlabel('Position in tissue')
ax1.set_ylabel('Position in tissue')
cbar1 = plt.colorbar(im1, ax=ax1)
cbar1.set_label('Normalized membrane potential')

# Panel B: tissue under amiloride (left half perturbed)
ax2 = plt.subplot(1, 2, 2)
im2 = ax2.imshow(V_drug, cmap='RdBu_r', vmin=-1, vmax=1)
ax2.set_title('Tissue under amiloride (left half perturbed)')
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel('Position in tissue')
cbar2 = plt.colorbar(im2, ax=ax2)
cbar2.set_label('Normalized membrane potential')

plt.tight_layout()
plt.savefig('figure4_tissue_abm.png', dpi=300)
plt.show()

print("Figure 4 saved: untreated vs amiloride-perturbed tissue patterns (left half treated).")
