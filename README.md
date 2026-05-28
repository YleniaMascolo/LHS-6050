# LHS 6050 — Global Transit/RV Fitting 

This repository implements a Global Fit framework (incorporating both circular and eccentric models) that combines TESS photometry, ground-based follow-up photometry, and ESPRESSO radial velocity (RV) data. It also includes tools to evaluate the bulk composition of the planet against theoretical interior models (Zeng et al. 2019).

The repository is organized as follows:

```text
├── GlobalFit/
│   ├── GlobalFit_circular.py      # MCMC global fit assuming a circular orbit (e=0)
│   └── GlobalFit_eccentric.py     # MCMC global fit allowing eccentric orbital parameters
│
├── functions/                     # Core numerical and statistical modules
│   ├── models_circular.py         # Light curve and RV forward models for circular orbits
│   ├── models_eccentric.py        # Forward models accounting for eccentricity
│   ├── emcee_run.py               # Wrapper to execute MCMC sampling via 'emcee' package
│   ├── multiGP.py                 # Multi-dimensional GPs for stellar activity mitigation
│   ├── stats_circular.py          # Statistical module for circular fits (prior, likelihood, posterior)
│   └── stats_eccentric.py         # Statistical module for eccentric fits (prior, likelihood, posterior)
│
├── datasets/                      # Raw and processed observational data
│   ├── TESS/                      # Discovery TESS FITS files (Sectors 4, 31, 42, 70, 71)
│   ├── LCO/                       # Ground-based follow-up photometry from Las Cumbres Observatory 
│   ├── TRAPPIST/                  # Ground-based transit photometry from TRAPPIST-South
│   ├── NGTS/                      # Ground-based light curves from NGTS
│   ├── MINERVA/                   # Ground-based MINERVA-Australis data 
│   ├── ESPRESSO/                  # RV measurements from ESPRESSO (multiple reduction pipelines)
│   ├── RV_best_fit_pipelines/     # Best-fit RV extractions (LBL, S-BART, ESP, GZ our pipeline)
│   ├── zeng_curves/               # Theoretical composition tracks (Zeng et al. 2019)
│   ├── processed_data/            # Scripts and pickle files containing processed data
│   └── best_fit_results.json      # Final joint-fit parameters from the best-reported model 
│
├── plot/                          # Visualization and diagnostic Jupyter notebooks
│   ├── transit_plot.ipynb         # Photometric phased transits and residuals
│   ├── radial_velocity_plot.ipynb # Radial velocity Keplerian curves and phase-folds
│   ├── gls_periodogram.ipynb      # Generalized Lomb-Scargle periodograms 
│   ├── pipeline_comparison_plot.ipynb # Visual assessment of different RV extraction techniques
│   └── diagrams.ipynb             # M-R and M-density diagrams, HZ and system overview
│
└── tests/
    └── recovery_injection.py      # Injection-recovery tests to validate GP sensitivity
```

To run the notebooks and scripts, please install the following dependencies:

* [numpy](https://github.com/numpy/numpy) 
* [pandas](https://github.com/pandas-dev/pandas) 
* [matplotlib](https://github.com/matplotlib/matplotlib) 
* [scipy](https://github.com/scipy/scipy) 
* [emcee](https://github.com/dfm/emcee) 
* [corner](https://github.com/dfm/corner.py) 
* [astropy](https://github.com/astropy/astropy) 
* [batman-package](https://github.com/lkreidberg/batman)
* [radvel](https://github.com/California-Planet-Search/radvel)
* [seaborn](https://github.com/mwaskom/seaborn)

```bash
pip install numpy pandas matplotlib scipy emcee corner astropy batman-package radvel seaborn
```



