import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def propranolol_pk(C, t, ke=0.231):
    # One-compartment first-order elimination
    return -ke * C

# Dose and PK parameters (illustrative)
D = 80e3          # dose in micrograms
F = 0.26          # bioavailability (fraction)
Vd = 5.11 * 70    # volume of distribution (L * kg), approximated

# Initial concentration
C0 = F * D / Vd

# Time in hours
t = np.linspace(0, 24, 100)

# Plasma concentration over time
C = odeint(propranolol_pk, C0, t)[:, 0]

# Simple Emax/Hill model for channel/receptor blockade
IC50 = 30.0       # concentration units consistent with C
n = 1.5
E = C**n / (IC50**n + C**n)   # fractional effect (0–1)

plt.figure(figsize=(6, 4))
plt.plot(t, E * 100)
plt.xlabel('Time (hours)')
plt.ylabel('Propranolol blockade (%)')
plt.title('Time course of propranolol-like blockade')
plt.grid(True)
plt.tight_layout()
plt.savefig('figure5.png', dpi=300)
plt.show()
