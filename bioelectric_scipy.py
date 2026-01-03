import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def bioelectric_model(y, t, k1, k2, k3):
    X, V = y
    dXdt = k1 * V - k2 * X * V
    dVdt = -k3 * V  # Voltage homeostasis
    return [dXdt, dVdt]

# Time points
t = np.linspace(0, 50, 1000)

# Baseline (no drug)
y0 = [1, 1]  # X=1, V=1 (normalized)
sol_baseline = odeint(bioelectric_model, y0, t, args=(1, 0.1, 0.01))

# Drug perturbation (k3 blocker - 90% inhibition)
sol_drug = odeint(bioelectric_model, y0, t, args=(1, 0.1, 0.001))

# PLOT FIGURE 1 - FIXED Y-AXIS + LEGEND
plt.figure(figsize=(6, 4))
plt.plot(t, sol_baseline[:, 1], 'k-', linewidth=2, label='Baseline (k₃ = 0.01)')
plt.plot(t, sol_drug[:, 1], 'r--', linewidth=2, label='Channel blocker (k₃ = 0.001)')
plt.xlabel('Time (arbitrary units)')
plt.ylabel('Normalized membrane potential (Vnorm)')  # ✅ FIXED
plt.title('Single-cell voltage dynamics under channel blockade')  # Polished title
plt.legend(title='Condition')  # Clean legend
plt.tight_layout()
plt.savefig('Figure1_single_cell_time_Vnorm.png', dpi=300, bbox_inches='tight')  # New filename
plt.show()

print("✅ Figure 1 updated: Figure1_single_cell_time_Vnorm.png")

# =============================================================================
# IVERMECTIN STAGE 1: Single-cell dose-response (Supplementary Figure S2)
# =============================================================================

print("\n🚀 Running Ivermectin Stage 1: Single-cell dose response...")

# Ivermectin: Cl- conductance scaling (hyperpolarizing effect on Vnorm)
cl_scales = np.logspace(-1, 0.7, 8)  # 0.1x to 5x Cl conductance
steady_states = []

for i, cl_scale in enumerate(cl_scales):
    # Simulate ivermectin as k3 increase (Cl- hyperpolarization pulls V toward 0)
    sol_ivm = odeint(bioelectric_model, y0, t, args=(1, 0.1, 0.01 * cl_scale))
    v_steady = sol_ivm[-1, 1]  # Final Vnorm
    steady_states.append(v_steady)
    print(f"Cl scale {cl_scale:.2f}x → Vnorm = {v_steady:.3f}")

# Save data (safe filename)
np.save('ivermectin_stage1_dose_response.npy', np.array([cl_scales, steady_states]))

# Plot dose-response (Figure S2 style)
plt.figure(figsize=(6, 4))
plt.semilogx(cl_scales, steady_states, 'mo-', linewidth=2, markersize=8, 
             label='Ivermectin-like Cl⁻ scaling')
plt.xlabel('Chloride conductance scale (× baseline)')
plt.ylabel('Steady-state Vnorm')
plt.title('Ivermectin-like single-cell dose response')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('ivermectin_stage1_dose_response.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Ivermectin Stage 1 complete:")
print("   → ivermectin_stage1_dose_response.npy")
print("   → ivermectin_stage1_dose_response.png")
