import numpy as np
import matplotlib.pyplot as plt
import corner
import json
import seaborn as sns
import pickle

import os,sys
from pathlib import Path
current_dir = Path(__file__).resolve()
project_root = current_dir.parent.parent
functions_path = project_root / "functions"
sys.path.append(str(functions_path))
import emcee_run
import stats_circular as stats



"Global fit script for system LHS 6050 assuming a circular orbit. Joint analisys of photometric (TESS and ground based) and ESPRESSO datasets"
"Outputs: corner and chains plots, chains and posteriors files, best fit parameters json file"



data_dir = "/home/ymascolo/Desktop/github/datasets" # path datsets
base_dir = "/home/ymascolo/Desktop/github/GlobalFit" # path for saving outputs

# import datasets
with open(os.path.join(data_dir, "processed_data/LHS6050b_global_data.pkl"), "rb") as f:
    data_loaded = pickle.load(f)

tess_data     = data_loaded["tess_data"]
lco_data      = data_loaded["lco_data"]
lcog_data     = data_loaded["lcog_data"]
trap_data     = data_loaded["trap_data"]
minerva_data  = data_loaded["minerva_data"]
ngts_data     = data_loaded["ngts_data"]
espresso_data = data_loaded["espresso_data"]

t0 = 2459164.3524 - 2457000  # mid-transit time (BTJD)
per  = 40.3834187              # period (days)

if __name__ == "__main__":

    # -----------------------------
    # Initialization and MCMC
    # -----------------------------
    ndim = 34
    niter = 100000
    nwalkers = 2*ndim

    init_guess = []
    while len(init_guess) < nwalkers:
        theta = [
            np.random.normal(t0, 1e-4),         ### t0
            np.random.normal(per, 1e-4),        ### per
            np.random.uniform(0.07, 0.10),      ### rp
            np.random.uniform(0.0, 1.0),        ### b
            np.random.normal(0.22,0.01),        ### mstar
            np.random.normal(0.25,0.01),        ### rstar
            np.random.uniform(0.0, 1.0),        ### q1_tess
            np.random.uniform(0.0, 1.0),        ### q2_tess
            np.random.uniform(0.0, 1.0),        ### q1_lco  
            np.random.uniform(0.0, 1.0),        ### q2_lco
            np.random.uniform(0.0, 1.0),        ### q1_lcog
            np.random.uniform(0.0, 1.0),        ### q2_lcog
            np.random.uniform(0.0, 1.0),        ### q1_minerva
            np.random.uniform(0.0, 1.0),        ### q2_minerva
            np.random.uniform(0.0, 1.0),        ### q1_ngts
            np.random.uniform(0.0, 1.0),        ### q2_ngts
            np.random.uniform(0.0, 1.0),        ### q1_trap
            np.random.uniform(0.0, 1.0),        ### q2_trap    
            np.random.normal(0.003,0.001),      ### krv
            np.random.normal(0,0.001),          ### gamma
            np.random.normal(100,2),            ### lam
            np.random.normal(6.2,0.1),          ### prot
            np.random.normal(0.5,0.1),          ### w
            np.random.normal(0.01,0.001),       ### A_rv
            np.random.normal(0.01,0.001),       ### A_fwhm
            np.random.normal(0.01,0.001),       ### A_bis
            np.random.normal(0.01,0.001),       ### A_asym
            np.random.normal(0.01,0.001),       ### A_RV_dGdt
            np.random.normal(0.01,0.001),       ### A_fwhm_dGdt
            np.random.normal(0.01,0.001),       ### A_asym_dGdt
            np.random.normal(0.01,0.001),       ### jitter_RV
            np.random.normal(0.01,0.001),       ### jitter_FWHM
            np.random.normal(0.01,0.001),       ### jitter_bis
            np.random.normal(0.01,0.001),       ### jitter_asym
            ]
                
                
        theta_lnlike = stats.log_probability(theta, tess_data, lco_data, lcog_data, minerva_data, ngts_data, trap_data, espresso_data)
        if abs(theta_lnlike) < np.inf:
            init_guess.append(theta)

    chain_file = os.path.join(base_dir, "retrieval_chain_circular.h5")        

    sampler, samples = emcee_run.emcee_adaptive_run(init_guess, nwalkers, niter, stats.log_probability, 
                                                       (tess_data, lco_data, lcog_data, minerva_data, ngts_data, trap_data, espresso_data), 
                                                       filename=chain_file, nproc=nwalkers,
                                                       resume=True)


    # -----------------------------
    # Chains plot
    # -----------------------------
    fig, axes = plt.subplots(ndim, figsize=(10, 25), sharex=True)
    samples = sampler.get_chain()
    labels = ["$t_0$", "P", "$R_p/R_s$", "b", "mstar", "rstar", "q1_tess", "q2_tess", "q1_lco", "q2_lco", "q1_lcog", "q2_lcog", "q1_minerva", "q2_minerva", "q1_ngts", "q2_ngts", "q1_trap", "q2_trap",
            "krv", "gamma", "lam", "prot", "w", "RV_scale", "fwhm_scale", "bis_scale", "asym_scale", "RV_dGdt_scale", "FWHM_dGdt_scale", "asym_dGdt_scale", "rv_jitter", "fwhm_jitter", "bis_jitter", "asym_jitter"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3) # [:, :, i] only i is related to number of free parameters
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")
    chain_plot = os.path.join(base_dir, "chain_plot_circular.pdf")
    plt.savefig(chain_plot)
    plt.show()


    # -----------------------------
    # Corner plot
    # -----------------------------
    samples = sampler.get_chain(discard=1900, flat=True)

    colors = sns.color_palette("Greens", 10) 
    sns.palplot(colors) 
    theta_best = np.zeros(ndim)

    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        theta_best[i] = mcmc[1]

    figure = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        smooth=1.0,
        smooth1d=1.0,
        levels = (0.393, 0.68, 0.86, 0.95),  
        color=colors[9],
        fill_contours=True,
        plot_datapoints=False,
        contour_kwargs={
            "colors": "black",
            "linewidths": 1.2,},
        hist_kwargs={
            "linewidth": 1.5,
            "color": "black",
            "alpha": 0.6},
    )

      # --- best-fit solution ---
    corner.overplot_lines(figure, theta_best, color="k", lw=1.5)
    corner.overplot_points(
        figure,
        theta_best[None],
        marker="s",
        color="k",
        markersize=4
    )

    for ax in figure.get_axes():
        ax.tick_params(direction="in", top=True, right=True)

    plt.tight_layout()
    corner_plot = os.path.join(base_dir, "corner_plot_circular.pdf")
    plt.savefig(corner_plot)
    plt.show()

    mcmc_results = os.path.join(base_dir, "mcmc_results_circular.npz")
    np.savez(mcmc_results,
            chain=sampler.get_chain(),
            log_prob=sampler.get_log_prob())


    # -----------------------------
    # Best-fit parameters
    # -----------------------------
    labels = ["t_0", "P", "R_p/R_s", "b", "mstar", "rstar", "q1_{tess}", "q2_{tess}", "q1_{lco}", "q2_{lco}", "q1_{lcog}", "q2_{lcog}", "q1_{minerva}", "q2_{minerva}", "q1_{ngts}", "q2_{ngts}", "q1_{trap}", "q2_{trap}",
            "krv", "gamma", "lam", "P_{rot}", "w", "RV_{scale}", "fwhm_{scale}", "bis_{scale}", "asym_{scale}", "RV_{dGdt\,scale}", "FWHM_{dGdt \,scale}", "asym_{dGdt \,scale}", "rv_{jitter}", "fwhm_{jitter}", "bis_{jitter}", "asym_{jitter}"]

    from IPython.display import display, Math

    theta_best = np.zeros(ndim)

    results = {
        "fitted_parameters": {},
        "derived_parameters": {}
    }

    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)

        theta_best[i] = mcmc[1]

        results["fitted_parameters"][labels[i]] = {
            "median": float(mcmc[1]),
            "err_minus": float(q[0]),
            "err_plus": float(q[1])
        }

        txt = "\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{{2:.5f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))

        
    # -----------------------------
    # Derived parameters (from chains)
    # -----------------------------
    t0, per, rp, b, mstar, rstar, _, _, _, _, _, _, _, _, _, _, _, _, krv, *_= samples.T

    Rsun_to_Rearth = 109.076
    G = 6.67430e-11
    M_sun = 1.9885e30
    day2sec = 86400
    M_earth = 5.972e24
    M_star = mstar * M_sun
    per_sec = per * day2sec
    R_earth = 6.3781e6
    teff_sun = 5778 

    # radius of planet 
    Rp_Rearth = rp * rstar * Rsun_to_Rearth 

    #semimajor axis
    a = ((per / 365.25)**(2/3) * mstar**(1/3)) 
    a_rstar = ((per / 365.25)**(2/3) * mstar**(1/3) * 215.032) / rstar

    cosi = b / a_rstar
    inc = np.degrees(np.arccos(cosi))
    sini = np.sin(np.radians(inc))

    #mass of planet 
    Mp = (krv * (M_star**(2/3)) * ((per_sec / (2*np.pi*G))**(1/3))) / sini
    Mpe = (Mp / M_earth)*1e3

    #density
    rhoe = Mpe / (Rp_Rearth**3)
    rho_cgs = rhoe * 5.51

    # equilibrium temperature
    teff = 3220 # star lhs6050
    teq = teff * (1 / (2 * a_rstar)**0.5) * (1 - 0.3)**0.25

    #surface gravity in m/s2
    gp = (G * Mp*1e3) / (Rp_Rearth * R_earth)**2

    # Transit duration (ore)
    term_sqrt = np.sqrt((1 + rp)**2 - b**2)
    tdur_days = (per / np.pi) * np.arcsin(term_sqrt / (a_rstar * sini))
    tdur_hrs = tdur_days * 24

    #insolation in Earth unit 
    S = (rstar**2) * (teff / teff_sun)**4 / (a**2)

    derived_params = [
            (Rp_Rearth, r"R_p\,[R_\oplus]"),
            (inc,       r"i\,[^\circ]"),
            (a_rstar,   r"a/R_\star\,"),
            (a,         r"a\,(AU)"),
            (Mpe,       r"M_p\,[M_\oplus]"),
            (a,         r"a(AU)"),
            (a_rstar,   r"a/r_{star}"),
            (rhoe,      r"\rho (\rho_e)"),
            (rho_cgs,   r"\rho (g\,cm^{-3})"),
            (teq,       r"T_{eq} (K)"),
            (gp,        r"g_p (m/s^2)"),
            (tdur_hrs,  r"T_{dur} (h)"),
            (S,         r"S (S_\oplus)")
        ]

    for values, label in derived_params:

        mcmc = np.percentile(values, [16, 50, 84])
        q = np.diff(mcmc)

        results["derived_parameters"][label] = {
            "median": float(mcmc[1]),
            "err_minus": float(q[0]),
            "err_plus": float(q[1])
        }

        txt = (
            r"{} = {:.3f}_{{-{:.5f}}}^{{+{:.5f}}}"
            .format(label, mcmc[1], q[0], q[1])
        )

        display(Math(txt))


    #==============================
    # BIC
    #==============================
    n = sum(len(sd["time"]) for sd in tess_data) + sum(len(sd["time"]) for sd in minerva_data) + sum(len(sd["time"]) for sd in lco_data) + sum(len(sd["time"]) for sd in lcog_data) \
        + sum(len(sd["time"]) for sd in ngts_data) + sum(len(sd["time"]) for sd in trap_data) +  len(espresso_data["bis"]) + len(espresso_data["fwhm"]) +len(espresso_data["rv"]) + len(espresso_data["asym"]) 
   
    # best log-likelihood
    best_ll = stats.log_likelihood(theta_best, tess_data, lco_data, lcog_data, minerva_data, ngts_data, trap_data, espresso_data)
    k_trend = len(labels) + 3*len(tess_data) + 4*len(lco_data) + 4*len(lcog_data) + 4*len(minerva_data) + 4*len(ngts_data) + 3*len(trap_data)

    BIC_trend = k_trend * np.log(n) - 2 * best_ll
    print("BIC (trend counted):", BIC_trend)

    results["model_statistics"] = {
        "BIC": float(BIC_trend),
    }
    
    best_fit_results = os.path.join(base_dir, "best_fit_results_circular.json")
    with open(best_fit_results, "w") as f:
        json.dump(results, f, indent=4)
