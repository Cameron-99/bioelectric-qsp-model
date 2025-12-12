import random
from deap import base, creator, tools

# 1. Create fitness and individual types
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

import numpy as np

def run_tissue_with_ivermectin(gj_scale, ch_scale, gain):
    """
    TEMPORARY placeholder.
    Replace the body of this function with your real tissue simulation.
    Must return a 2D numpy array: tissue_voltage[y, x]
    """
    # Example dummy tissue: a constant field that depends on parameters
    size = 16
    base = (gj_scale + ch_scale + gain) / 3.0
    tissue = np.full((size, size), base)
    return tissue

# Target pattern (for now, just all ones)
TARGET_PATTERN = np.ones((16, 16))

def evaluate(ind):
    gj_scale, ch_scale, gain = ind
    tissue = run_tissue_with_ivermectin(gj_scale, ch_scale, gain)
    error = np.mean((tissue - TARGET_PATTERN)**2)
    return (-error,)

# 2. Toolbox - FIXED: register select here
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, 0, 2)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=3)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=3)

# 3. Run simple evolution
pop = toolbox.population(n=20)
print("Starting evolution...")

for gen in range(10):
    # Evaluate all
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    # Select best for next generation
    pop[:] = toolbox.select(pop, len(pop))

# Get best result
best = tools.selBest(pop, 1)[0]
print("Best individual:", [round(x,2) for x in best])
print("Best fitness:", round(best.fitness.values[0], 4))
