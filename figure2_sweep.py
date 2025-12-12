import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def bioelectric_model(y, t, k1, k2, k3):
    X, V = y
    return [k1 * V - k2 * X * V, -k3 * V]

# Parameters
k3_baseline = 0.01
IC50 = 1e-5      # example amiloride IC50 in M
hill_n = 1.0

def k3_effective(conc):
    # conc in M
    inhibition = (conc**hill_n) / (IC50**hill_n + conc**hill_n)
    return k3_baseline * (1.0 - inhibition)

t = np.linspace(0, 50, 1000)
y0 = [1, 1]

# Amiloride concentrations (M)
concs = np.logspace(-8, -3, 6)   # 1e-8 to 1e-3 M

plt.figure(figsize=(12, 5))

final_voltages = []

# Panel A: time courses for a few representative doses
plt.subplot(1, 2, 1)
for conc in [1e-8, 1e-6, 1e-4]:
    k3_eff = k3_effective(conc)
    sol = odeint(bioelectric_model, y0, t, args=(1, 0.1, k3_eff))
    plt.plot(t, sol[:, 1], label=f'[Amiloride] = {conc:.0e} M')

plt.xlabel('Time (s)')
plt.ylabel('Normalized membrane potential')
plt.title('Single-cell voltage over time\nfor selected amiloride concentrations')
plt.legend()
plt.grid(True)

# Panel B: steady-state voltage vs concentration
for conc in concs:
    k3_eff = k3_effective(conc)
    sol = odeint(bioelectric_model, y0, t, args=(1, 0.1, k3_eff))
    final_voltages.append(sol[-1, 1])

plt.subplot(1, 2, 2)
plt.semilogx(concs, final_voltages, 'o-')
plt.xlabel('Amiloride concentration (M, log scale)')
plt.ylabel('Steady-state normalized membrane potential')
plt.title('Dose–response of steady-state voltage to amiloride')
plt.grid(True)

plt.tight_layout()
plt.savefig('figure2_dose_response.png', dpi=300)
plt.show()

print("Figure 2 saved with realistic pharmacology axis.")
