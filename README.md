# LHS 6050 — Global Transit/RV Fitting 

This repository implements a Global Fit framework (incorporating both circular and eccentric models) that combines TESS photometry, ground-based follow-up photometry, and ESPRESSO radial velocity (RV) data. It also includes tools to evaluate the bulk composition of the planet against theoretical interior models.

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
├── tests/
│   └── recovery_injection.py      # Injection-recovery tests to validate GP sensitivity
|
└── interior_structure_model/      # Folder with PDF plots and Jupyter notebooks for interior structure model using ExoMDN
    ├── lhs6050b_samples.parquet   # Output from ExoMDN model in .ipynb 
    ├── plot_ridge_lhs6050b.ipynb  # Ridgeplot and the standard ExoMDN cornerplot
    └── run_exomdn_lhs6050b.ipynb  # Inference for ExoMDN model using mass radius and equilibrium temperature of LHS 6050b
```
Additional data [here](https://drive.google.com/drive/folders/14kszrij8drW1fSgP8EOUyYQ7x352nvvN?usp=share_link):
- MCMC joint-fit posteriors from the best-reported model (used in radial_velocity_plot.ipynb)
- NASA Exoplanet Archive DataFrame (used in diagrams.ipynb) 

## System Requirements and Installation Guide

Please see the 'requirements.txt' file for version numbers of all packages associated with the Python environment used to run codes in this repository. This software has been tested on MacOS Tahoe 26.4.1 operating systems. Using a HPC is preferred.
There is no non-standard hardware required, although this code can make use of multiprocessing.

To run the notebooks and scripts, please install the following dependencies. 
Typical install time: 30 minutes.

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
* [ExoMDN](https://github.com/philippbaumeister/ExoMDN.git)

```bash
pip install numpy pandas matplotlib scipy emcee corner astropy batman-package radvel seaborn exomdn
```

## Instructions for Use

Before running the code, ensure to have an active Python environment with the required dependencies installed.

To run the analysis, you must keep the directory structure of the repository intact. The scripts interact with each other as follows:
* `datasets/`: Contains all the raw and processed observational data. 
* `functions/`: Contains the numerical, statistical, and MCMC modules. These python files are core libraries automatically imported by the main execution scripts.
* `GlobalFit/`: Contains the main executable scripts that orchestrate the entire analysis.
* `tests/`: Contains a test file for Gaussian Process sensitivity. 
You do not need to run the files inside `functions/` individually. You only need to execute one of the main scripts inside the `GlobalFit/` or `tests/` directories, depending on the orbital model assumption.

The script is configured to save all outputs automatically inside the directory specified by `base_dir` (by default configured within the script). Upon successful completion of the MCMC run, the following files will be generated:
* `chain_plot_circular.pdf`: A multi-panel plot containing the MCMC chains for all free parameters.
* `corner_plot_circular.pdf`: The final publication-ready corner plot showing the multi-dimensional posterior probability distributions, covariance contours, and median best-fit values overlaid on the histograms.
* `retrieval_chain_circular.h5`: The main HDF5 backend file that stores the entire state of the sampler.
* `mcmc_results_circular.npz`: A compressed NumPy binary file containing the raw flattened chains (`chain`) and their corresponding log-probabilities (`log_prob`).
* `best_fit_results_circular.json`: A structured JSON file containing the complete final data of the system fit:
    * `fitted_parameters`: Median values, lower bounds (`err_minus`), and upper bounds (`err_plus`) for all the initialised parameters.
    * `derived_parameters`: Physically calculated planetary properties with propagated uncertainties.
    * `model_statistics`: The final Bayesian Information Criterion (BIC) score.

Once the global fit is complete or to analyze the results, the interactive Jupyter Notebooks can be used.
* Plotting Results: Open the notebooks in the `plot/` folder to generate phased transit light curves (`transit_plot.ipynb`), Keplerian RV curves (`radial_velocity_plot.ipynb`), or Mass-Radius diagrams (`diagrams.ipynb`).
* Interior Structure (`interior_structure_model/`): Run `run_exomdn_lhs6050b.ipynb` to infer the core-mantle-atmosphere composition of the planet based on the best-fit mass and radius, using the ExoMDN machine learning model framework.

## Demo

To test the pipeline or run a quick validation, you can follow the instructions above and modify the `niter` (MCMC steps) parameter inside the main script.
Open `GlobalFit/GlobalFit_circular.py` and temporarily lower the number of steps to around 1,000 (instead of the production value 100,000).
On a standard desktop computer, running 1,000 steps with 8-10 parallel processes (`nproc`) takes approximately 4 to 5 hours. 
