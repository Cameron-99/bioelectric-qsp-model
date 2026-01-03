import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random

# =========================
# Load baseline & perturbed
# =========================

target_pattern = np.load('figure6_baseline_pattern.npy')   # baseline pattern
perturbed_pattern = np.load('ivermectin_stage2_tissue_pattern.npy')  # 10x10

# Ensure target_pattern is 2D and matches ivermectin tissue shape
if target_pattern.ndim == 1:
    target_pattern = np.tile(target_pattern, (perturbed_pattern.shape[0], 1))

N = perturbed_pattern.shape[0]  # assumed 10

# ================
# Bioelectric model
# ================

def bioelectric_model(y, t, k1, k2, k3):
    X, V = y
    dXdt = k1 * V - k2 * X * V
    dVdt = -k3 * V
    return [dXdt, dVdt]

t = np.linspace(0, 100, 2000)
y0 = [1, 1]

# =========================
# DEAP evolutionary framework
# =========================

# Define fitness & individual
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0.5, 2.0)  # k1, k2, k3 bounds
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, 3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(individual, target=target_pattern):
    """Fitness = MSE between evolved tissue and target pattern (under fixed ivermectin)."""
    k1, k2, k3 = individual

    evolved_pattern = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            # Left half: ivermectin fixed (2x Cl conductance)
            if j < N // 2:
                k3_ivm = 0.01 * 2.0 * k3
            else:
                k3_ivm = 0.01 * k3

            sol = odeint(bioelectric_model, y0, t, args=(k1, k2, k3_ivm))
            evolved_pattern[i, j] = sol[-1, 1]

    mse = np.mean((evolved_pattern.flatten() - target.flatten())**2)
    return mse,

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

# ===========
# Run evolution
# ===========

pop = toolbox.population(n=100)
hof = tools.HallOfFame(3)
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)

pop, logbook = algorithms.eaSimple(
    pop, toolbox,
    cxpb=0.5, mutpb=0.2, ngen=50,
    stats=stats, halloffame=hof, verbose=True
)

# ==========================
# Reconstruct best pattern & save
# ==========================

def simulate_pattern_from_params(params):
    k1, k2, k3 = params
    pattern = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if j < N // 2:
                k3_ivm = 0.01 * 2.0 * k3
            else:
                k3_ivm = 0.01 * k3
            sol = odeint(bioelectric_model, y0, t, args=(k1, k2, k3_ivm))
            pattern[i, j] = sol[-1, 1]
    return pattern

best_params = np.array([list(ind) for ind in hof])
np.save('ivermectin_stage3_parameters.npy', best_params)

best_pattern = simulate_pattern_from_params(hof[0])
np.save('ivermectin_stage3_best_pattern.npy', best_pattern)

# ==========================
# Plot 3-panel evolution figure
# ==========================

# Shared color limits for all three panels
vmin = min(target_pattern.min(), perturbed_pattern.min(), best_pattern.min())
vmax = max(target_pattern.max(), perturbed_pattern.max(), best_pattern.max())
margin = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
vmin -= margin
vmax += margin
cmap = "viridis"

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

# Target (baseline)
im1 = ax1.imshow(target_pattern, cmap=cmap, vmin=vmin, vmax=vmax)
ax1.set_title('Target pattern\n(Baseline)')
ax1.set_xlabel('Column')
ax1.set_ylabel('Row')
plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

# Perturbed (ivermectin)
im2 = ax2.imshow(perturbed_pattern, cmap=cmap, vmin=vmin, vmax=vmax)
ax2.set_title('Ivermectin-perturbed')
ax2.set_xlabel('Column')
ax2.set_ylabel('Row')
plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

# Best evolved
im3 = ax3.imshow(best_pattern, cmap=cmap, vmin=vmin, vmax=vmax)
ax3.set_title(
    f'Best evolved\n[k1={hof[0][0]:.2f}, k2={hof[0][1]:.2f}, k3={hof[0][2]:.2f}]'
)
ax3.set_xlabel('Column')
ax3.set_ylabel('Row')
plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.savefig('ivermectin_stage3_evolution.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Ivermectin Stage 3 COMPLETE")
print("   → ivermectin_stage3_parameters.npy")
print("   → ivermectin_stage3_best_pattern.npy")
print("   → ivermectin_stage3_evolution.png")
