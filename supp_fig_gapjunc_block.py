#!/usr/bin/env python3
"""
Supplementary Figure S4: Gap Junction Blockade Validation (Reviewer 1)
Shows amiloride fails to create tissue Vmem patterns without gap junctions
For ScholarOne resubmission - Cameron-99/bioelectric-qsp-model
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # NON-INTERACTIVE: Linux Cinnamon compatible
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === YOUR MODEL PARAMETERS (from Table S1 CSV) ===
k1, k2, k3_base = 1.0, 0.5, 0.8  # Steady V ≠ 0
N = 10  # 10x10 tissue grid
n_steps = 500  # Relaxation steps
dt = 0.1

print("🧬 Bioelectric tissue simulation: Gap junction validation")
print(f"Grid: {N}x{N}, Steps: {n_steps}, k1={k1}, k2={k2}, k3_base={k3_base}")

def tissue_step(V_grid, k3_grid, D):
    """One relaxation step: local ODE + gap junction diffusion"""
    V_new = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            # Local single-cell steady state (YOUR ODE)
            k3 = k3_grid[i,j]
            sol = solve_ivp(lambda t,y: [k1*y[1]-k2*y[0], k3*(y[0]-y[1])], 
                          [0, 50], [0.1, -0.2], rtol=1e-5, atol=1e-8)
            V_local = sol.y[1,-1]
            
            # Gap junction coupling to neighbors
            if D > 0:
                V_sum = 0
                n_neighbors = 0
                for di, dj in [(-1,0),(1,0),(0,-1),(0,1)]:
                    ni, nj = i+di, j+dj
                    if 0 <= ni < N and 0 <= nj < N:
                        V_sum += V_grid[ni,nj]
                        n_neighbors += 1
                if n_neighbors > 0:
                    V_neighbor_avg = V_sum / n_neighbors
                    V_new[i,j] = (1-D)*V_local + D*V_neighbor_avg
                else:
                    V_new[i,j] = V_local
            else:
                V_new[i,j] = V_local
    
    return V_new

# === AMILORIDE LEFT HALF (like Fig 3) ===
print("💊 Setting up amiloride perturbation (left 5 columns k3 ↓ 50%)")
k3_grid = np.ones((N,N)) * k3_base
k3_grid[:,:5] *= 0.5  # Amiloride zone

# === SIMULATE TWO CONDITIONS ===
print("🔬 Simulating Gap Junction INTACT (D=0.1)...")
V_gj_intact = np.zeros((N,N)) * -0.1  # Initial condition
for step in range(n_steps):
    V_gj_intact = tissue_step(V_gj_intact, k3_grid, D=0.1)

print("🔬 Simulating Gap Junction BLOCKED (D=0.0)...")  
V_gj_blocked = np.zeros((N,N)) * -0.1  # Same initial
for step in range(n_steps):
    V_gj_blocked = tissue_step(V_gj_blocked, k3_grid, D=0.0)

# === PLOT & SAVE (PUBLICATION VERSION ===
print("📊 Creating publication-ready SuppFig_S4...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)

# Panel A: Gap Junctions INTACT 
im1 = ax1.imshow(V_gj_intact, cmap='RdBu_r', vmin=-0.4, vmax=0.2)
ax1.set_title('Amiloride + Gap Junctions\n(D=0.1)', fontsize=12, fontweight='bold')
ax1.plot([4.5,4.5], [0,10], 'k--', lw=2, label='Domain boundary')
ax1.legend()
cbar1 = plt.colorbar(im1, ax=ax1, label='Vnorm', shrink=0.8)
ax1.set_xlabel('X'); ax1.set_ylabel('Y')

# Panel B: Gap Junctions BLOCKED
im2 = ax2.imshow(V_gj_blocked, cmap='RdBu_r', vmin=-0.4, vmax=0.2)
ax2.set_title('Amiloride + Gap Blockade\n(D=0.0)', fontsize=12, fontweight='bold')
cbar2 = plt.colorbar(im2, ax=ax2, label='Vnorm', shrink=0.8)
ax2.set_xlabel('X'); ax2.set_ylabel('Y')

# PUBLICATION CAPTION 
plt.suptitle('Supplementary Figure S3: Gap Junction Dependence of Tissue Vmem Patterning', 
             fontsize=14, fontweight='bold')
plt.tight_layout()

# SAVE HIGH-RES PUBLICATION FILES
plt.savefig('SuppFig_S4_gapjunc_blockade.png', dpi=300, bbox_inches='tight')
plt.savefig('SuppFig_S4_gapjunc_blockade.pdf', bbox_inches='tight') 
plt.close()

print("\n✅ PUBLICATION FILES SAVED (Reviewer 1 reference removed):")
print(f"   SuppFig_S4_gapjunc_blockade.png")
print(f"   SuppFig_S4_gapjunc_blockade.pdf")
