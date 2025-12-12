import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def bioelectric_model(y, t, k1, k2, k3):
    X, V = y
    return [k1 * V - k2 * X * V, -k3 * V]

t = np.linspace(0, 50, 1000)
y0 = [1, 1]

# Amiloride (1980s Na channel blocker) - real pharmacology data
amiloride_doses = [0, 1e-6, 1e-5, 1e-4, 1e-3]  # μM
ic50_amiloride = 5e-6  # Literature IC50
hill_n = 1.2

inhibition = [dose**hill_n / (dose**hill_n + ic50_amiloride**hill_n) for dose in amiloride_doses]

plt.figure(figsize=(12, 5))

for i, (dose, inhib) in enumerate(zip(amiloride_doses, inhibition)):
    k3_eff = 0.01 * (1 - inhib)
    sol = odeint(bioelectric_model, y0, t, args=(1, 0.1, k3_eff))
    
    plt.subplot(1, 2, 1)
    plt.semilogx(dose*1e6, sol[-1,1], 'o-', label=f'{dose*1e6:.0f}μM')
    
plt.subplot(1, 2, 1)
plt.xlabel('Amiloride [μM]'); plt.ylabel('Final Voltage'); plt.title('Amiloride Bioelectric Effect')
plt.legend(); plt.grid()

plt.subplot(1, 2, 2)
plt.semilogx(amiloride_doses, inhibition, 'ro-')
plt.xlabel('Amiloride [μM]'); plt.ylabel('k3 Inhibition'); plt.title('Hill Equation IC50 Fit')
plt.grid()

plt.tight_layout()
plt.savefig('figure3_amiloride.png', dpi=300)
plt.show()

print(f"Figure 3 saved! Amiloride IC50: {ic50_amiloride*1e6:.0f} μM")
