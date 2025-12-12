import random
import numpy as np
from deap import base, creator, tools
from scipy.integrate import odeint

# 1. Your single-cell model
def bioelectric_model(y, t, k1, k2, k3):
    X, V = y
    return [k1 * V - k2 * X * V, -k3 * V]

# 2. Tissue simulation using 10 independent cells
def run_tissue_simulation(params):
    """Use your ODE on 10 independent cells to build a 1D 'tissue'."""
    k1, k2, k3 = params
    n_cells = 10
    t = np.linspace(0, 100, 200)  # time points
    V_final = []

    for _ in range(n_cells):
        y0 = [1.0, -40.0]  # initial X, V (example)
        sol = odeint(bioelectric_model, y0, t, args=(k1, k2, k3))
        X_t, V_t = sol.T
        V_final.append(V_t[-1])  # final voltage of this cell

    return np.array(V_final)  # shape (10,)

# Baseline target: final voltages with default params [1,1,1]
BASELINE_PARAMS = [1.0, 1.0, 1.0]
TARGET_PATTERN = run_tissue_simulation(BASELINE_PARAMS)

# 3. DEAP types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

# 4. Toolbox
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

def evaluate(ind):
    """Fitness = how close final voltages match target pattern."""
    tissue = run_tissue_simulation(ind)            # shape (10,)
    error = np.mean((tissue - TARGET_PATTERN)**2)  # scalar
    return (-error,)  # DEAP maximizes fitness, so use negative error

toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)


# 4. Run evolution
if __name__ == "__main__":
    test_ind = [1.0, 1.0, 1.0]
    tissue = run_tissue_simulation(test_ind)
    error = np.mean((tissue - TARGET_PATTERN)**2)
    print("Test error at [1,1,1]:", error)


pop = toolbox.population(n=20)
print("Generation | Best Fitness")
print("-----------|------------")

gen_list = []
fit_list = []

for gen in range(15):
    # Evaluate
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Select
    pop[:] = toolbox.select(pop, len(pop))

    best_fitness = max(ind.fitness.values[0] for ind in pop)
    print(f"{gen:9d} | {best_fitness:10.4f}")

    gen_list.append(gen)
    fit_list.append(best_fitness)


# Best result
best = tools.selBest(pop, 1)[0]

import csv

with open("figure6_fitness.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["generation", "best_fitness"])
    writer.writerows(zip(gen_list, fit_list))

print("\nBEST PARAMS:", [round(x,2) for x in best])
print("Final fitness:", best.fitness.values[0])
