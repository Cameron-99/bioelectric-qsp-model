import numpy as np
import matplotlib.pyplot as plt

N=50
healthy = np.ones(N)
propranolol = np.ones(N); propranolol[:25] = 0.7

for _ in range(200):
    healthy += np.random.normal(0,0.01,N)
    diffusion = 0.05 * np.diff(propranolol, prepend=propranolol[0], append=propranolol[-1])
    propranolol[:25] *= 0.995
    propranolol += diffusion * 0.8

plt.figure(figsize=(12,4))
plt.subplot(121); plt.plot(healthy,'g-',lw=2); plt.title('Healthy Tissue')
plt.subplot(122); plt.plot(propranolol,'b-',lw=2); plt.title('Propranolol Effect')
plt.suptitle(f'Propranolol: {100*(1-np.mean(propranolol)/np.mean(healthy)):.1f}% disruption')
plt.savefig('figure5_propranolol.png',dpi=300); plt.show()
print('Figure 5 COMPLETE!')
