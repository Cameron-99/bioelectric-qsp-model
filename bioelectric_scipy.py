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
y0 = [1, 1]  # X=1, V=1
sol_baseline = odeint(bioelectric_model, y0, t, args=(1, 0.1, 0.01))

# Drug perturbation (k3 blocker - 90% inhibition)
sol_drug = odeint(bioelectric_model, y0, t, args=(1, 0.1, 0.001))

# Plot
plt.figure(figsize=(6, 4))

plt.plot(t, sol_baseline[:, 1], label='No drug (k3 = 0.01)')
plt.plot(t, sol_drug[:, 1], 'r--', label='Channel blocker (k3 = 0.001)')

plt.xlabel('Time (arbitrary units)')
plt.ylabel('Membrane potential (mV)')
plt.title('Single-cell voltage over time\nwith and without channel blocker')
plt.legend()

plt.tight_layout()
plt.savefig('Figure1_single_cell_time.png')
plt.show()

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
y0 = [1, 1]  # X=1, V=1
sol_baseline = odeint(bioelectric_model, y0, t, args=(1, 0.1, 0.01))

# Drug perturbation (k3 blocker - 90% inhibition)
sol_drug = odeint(bioelectric_model, y0, t, args=(1, 0.1, 0.001))

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1,2,1)
plt.plot(t, sol_baseline[:,1], label='Baseline V')
plt.xlabel('Time'); plt.ylabel('Voltage'); plt.title('No Drug'); plt.legend()

plt.subplot(1,2,2)
plt.plot(t, sol_drug[:,1], 'r--', label='Drug (k3 blocked)')
plt.xlabel('Time'); plt.ylabel('Voltage'); plt.title('90% Channel Blocker'); plt.legend()
plt.tight_layout()
plt.xlabel("Amiloride concentration (log10 M)")
plt.ylabel("Membrane potential (mV)")
plt.title("Effect of amiloride (Na⁺ channel blocker) on single-cell voltage")
plt.savefig('Figure_1.png, dpi=300')
plt.show()
