Bioelectric QSP Model
Bioelectric simulations of classic ion channel drugs using SciPy ODE solvers. Replicates tissue-level 
effects of HCN2 channel modulators (ivermectin, propranolol) with evolutionary optimization via DEAP.

# Bioelectric QSP Model
Computational pipeline for "Ion Channel Pharmacology Meets Bioelectric Pattern Control"
- ODE tissue simulations (10x10 gap-junction coupled)
- DEAP evolutionary optimization (cxBlend α=0.5, mutGaussian σ=0.2)
- Propranolol PK model (ke=0.231 h⁻¹)
See manuscript.pdf for full details.

Quick start
git clone https://github.com/Cameron-99/bioelectric-qsp-model.git
cd bioelectric-qsp-model
python3 -m venv venv && source venv/bin/activate
pip install scipy numpy matplotlib deap yaml
python bioelectric_scipy.py  # Core QSP model
python figure5_propranolol.py  # Drug sweeps

Requirements
# Core (LMDE 7 verified)
Python 3.10+ numpy scipy matplotlib deap pyyaml

# Optional visualization
seaborn pandas jupyter  # plot_figure6.py

Citation
@misc{cameron2026bioelectricqsp,
  title = {Bioelectric QSP Model: Ion Channel Drug Simulations},
  author = {McCulloch, CR},
  year = {2026},
  publisher = {GitHub},
  journal = {bioelectric-qsp-model},
  doi = {10.5281/zenodo.XXXXXXX}
}
Related Work

    Tufts Levin Lab bioelectricity

    ​

    BETSE neuroepithelium optimization (sister repo)

    HCN2 somite patterning (93% recovery demonstrated)

License: MIT. Contact: cameron.mcculloch@gmx.at
