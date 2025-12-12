import csv
import matplotlib.pyplot as plt

gens = []
fits = []

with open("figure6_fitness.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        gens.append(int(row["generation"]))
        fits.append(float(row["best_fitness"]))

plt.figure(figsize=(4,3))
plt.plot(gens, fits, "o-", color="navy")
plt.xlabel("Generation")
plt.ylabel("Best fitness")
plt.title("Figure 6: Evolution of fitness")
plt.tight_layout()
plt.savefig("figure6_fitness.png", dpi=300)
