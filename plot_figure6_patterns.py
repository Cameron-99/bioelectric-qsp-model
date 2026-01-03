import numpy as np
import matplotlib.pyplot as plt

baseline = np.load("figure6_baseline_pattern.npy")
evolved  = np.load("figure6_evolved_pattern.npy")

x = range(len(baseline))

plt.figure(figsize=(4,3))
plt.plot(x, baseline, "o-", label="Baseline [1,1,1]")
plt.plot(x, evolved,  "s--", label="Evolved [1.6,0.27,1.7]")
plt.xlabel("Cell index")
plt.ylabel("Final voltage (arb. units)")  # or (mV) if appropriate

# ---- insert this block here ----
ymin = min(baseline.min(), evolved.min())
ymax = max(baseline.max(), evolved.max())
margin = 0.05 * (ymax - ymin) if ymax > ymin else 0.1 * abs(ymax or 1.0)
plt.ylim(ymin - margin, ymax + margin)
# ---- end inserted block ----

plt.title("Optimization restores the tissue pattern\nunder drug perturbation")
plt.legend()
plt.tight_layout()
plt.savefig("figure6_patterns.png", dpi=300)
print("Saved figure6_patterns.png")
