
Every cell has tiny electrical channels that control its voltage, and together these voltages form patterns across tissues that guide how organs grow and repair. This study used a computer model to test whether older, 
well-known drugs that affect these channels — such as amiloride and ivermectin — could be repurposed to steer those tissue-wide voltage patterns. Using an evolutionary optimization approach, the model showed 
that tissues could, in principle, recover their target electrical patterns even after being disrupted by drugs. The work suggests that legacy drugs might one day be used to guide tissue regeneration or growth by 
controlling bioelectric "goal states" — essentially steering the body's electrical instructions.


# Bioelectric Evolution: Ivermectin-Induced Patterning

**Computational modeling of evolutionary optimization for bioelectric signaling toward ivermectin-like asymmetric membrane potential patterns across 3 stages: single-cell ODE → tissue simulation → parameter evolution.**


## 🎯 Project Overview

**Research Question**: Can evolutionary algorithms discover kinetic parameters (k1,k2,k3) that evolve uniform cells toward a target asymmetric V_m pattern matching ivermectin tissue perturbation (2x k3 scaling on left half)?

**3-Stage Pipeline**:

Stage 1: Single-cell ODE → dX/dt = k1V - k2XV, dV/dt = k3X - k3V
Stage 2: 10×10 tissue → Left-half ivermectin (f_ivm=2.0), right-half baseline
Stage 3: Evolution → 500 generations, tournament selection, Gaussian mutation


**Key Result**: Fitness converges to target pattern (MSE minimization).

## 🧪 Biological Context

**Model**: Minimal 2-variable ODE capturing bioelectric signaling (X) ↔ membrane potential (V) dynamics.

**Ivermectin Effect**: 2x increase in chloride pathway (k3 scaling) on tissue left half creates **left-high/right-low V_m asymmetry**.

Baseline: k3 = 0.01 everywhere
Ivermectin: k3_left = 0.02, k3_right = 0.01
Target: Left V_m ≈ 0.8, Right V_m ≈ 0.3


## 🖥️ Technical Stack

| Component | Technology |
|-----------|------------|
| **OS** | Linux LMDE 7 (Cinnamon, i5-7500, 16GB RAM) |
| **Language** | Python 3 |
| **Evolution** | DEAP (tournament, Gaussian mutation σ=0.2, blend crossover α=0.5) |
| **ODE Solver** | scipy.integrate.odeint |
| **Visualization** | matplotlib (300 DPI TIFF, LZW 6.3MB) |

## 📁 Repository Structure

├── src/
│ ├── stage1_singlecell.py # ODE validation
│ ├── stage2_tissue.py # 10×10 ivermectin simulation
│ ├── stage3_evolution.py # DEAP evolutionary optimization
│ └── visualization.py # Supplementary Figure S5
├── data/
│ ├── Blockmodule-Parametersymbol-Valuerange-Units-Notes.csv # Master parameter table
│ ├── evolved_best_individual.npy # Final solution (k1,k2,k3)
│ └── fitness_history.npy # 500 generations
├── figures/
│ └── Supplementary_Figure_S5.tiff # Publication-ready (6.3MB)
├── results/
│ ├── V_initial_10x10.npy # Uniform starting pattern
│ ├── V_evolved_10x10.npy # Optimized pattern
│ └── V_target_10x10.npy # Gradient target
└── README.md

#Key parameters

| Module          | Parameter      | Range           | Notes                     |
| --------------- | -------------- | --------------- | ------------------------- |
| Single-cell ODE | k1,k2,k3       | [0.5,2.0]       | Evolution bounds          |
| Tissue          | N=10×10        | Fixed           | Independent ODEs per cell |
| Ivermectin      | f_ivm=2.0      | Left half       | k3 scaling factor         |
| Evolution       | Population=100 | 500 generations | Tournament size=3         |

#Results summary

Best individual: k1=1.23, k2=0.87, k3=1.45
Final fitness: 0.023 (target MSE)
Convergence: ~150 generations

👥 Author

Cameron R. McCulloch, PhD
Veterinarian, Pharmacologist
Vienna, AT • vet19@gmx.at • [ORCID] 0009-0001-3517-7864

https://doi.org/10.1177/25763113261459408

