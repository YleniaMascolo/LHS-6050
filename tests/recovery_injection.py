import numpy as np
import matplotlib.pyplot as plt
import corner
import pandas as pd
import json
import seaborn as sns

import os,sys
from pathlib import Path
current_dir = Path(__file__).resolve()
project_root = current_dir.parent.parent
functions_path = project_root / "functions"
sys.path.append(str(functions_path))
import emcee_run
import multiGP

data_dir = "/home/ymascolo/Desktop/github/datasets" # path datsets
base_dir = "/home/ymascolo/Desktop/github/tests" # path for saving outputs

# -----------------------------
# Sinusoidal planet model function 
# -----------------------------
def planetmodel(krv, t, t0, per):
    phase = (t-t0)/per
    rvmodel = -1 * krv * np.sin(2*np.pi * phase)

    return rvmodel

# ============================================================
# Log-prior 
# ============================================================
def log_prior(theta):

    krv, gamma, lam, prot, w, RV_scale, fwhm_scale, bis_scale, asym_scale, RV_dGdt_scale, FWHM_dGdt_scale, asym_dGdt_scale, rv_jitter, fwhm_jitter, bis_jitter, asym_jitter = theta 

    if not (
        0.0 < krv < 0.1 and            
        -0.002 < gamma < 0.002 and     
        0.0 < lam < 600.0 and          
        6.0 < prot < 6.5 and            
        0.0 < w < 100 and               
        0 < RV_scale < 10 and        ### rv_GP_scale
        0 < fwhm_scale < 10 and      ### fwhm_GP_scale
        0 < bis_scale < 10 and       ### bis_GP_scale
        0 < asym_scale < 10 and       ### asym_GP_scale
        0 < RV_dGdt_scale < 10 and   ### rv_dGdT_GP_scale
        0 < FWHM_dGdt_scale < 10 and ### fwhm_dGdT_GP_scale
        0 < asym_dGdt_scale < 10 and ### asym_dGdT_GP_scale
        0 < rv_jitter < 0.1 and      ### rv_jitter
        0 < fwhm_jitter < 0.1 and    ### fwhm_jitter
        0 < bis_jitter < 0.1 and    ### bis_jitter
        0 < asym_jitter < 0.1         ### asym_jitter      
    ):
        return -np.inf
    
    return 0.0

# ============================================================
# Log-likelihood Multidimension GP
# ============================================================
def log_likelihood(theta, espresso_data, t0, per):

    krv, gamma, lam, prot, w, RV_scale, fwhm_scale,  bis_scale, asym_scale, RV_dGdt_scale, FWHM_dGdt_scale, asym_dGdt_scale, rv_jitter, fwhm_jitter, bis_jitter, asym_jitter = theta 
    
    t = espresso_data["time"]
    rv = espresso_data["rv"]
    rv_err = espresso_data["rv_err"]
    fwhm = espresso_data["fwhm"]
    fwhm_err = espresso_data["fwhm_err"]
    bis = espresso_data["bis"]
    bis_err = espresso_data["bis_err"]
    asym = espresso_data["asym"]
    asym_err = espresso_data["asym_err"]
    
    rvmodel = planetmodel(krv, t, t0, per)
    hp = multiGP.QPHyperParams(amp=1, lam=lam, P=prot, w=w)  # amp=1 because absorbed in A and B coefficients
    A = np.array([RV_scale, fwhm_scale,bis_scale,  asym_scale])      # e.g. RV and BIS scale with G
    B = np.array([RV_dGdt_scale, FWHM_dGdt_scale, 0.0, asym_dGdt_scale])      # e.g. RV also scales with dG/dt, BIS not

    jitter = np.array([rv_jitter, fwhm_jitter, bis_jitter, asym_jitter]) # per observable

    ll_rv = multiGP.lnlike_multidim_qp_gp(
        t_list=[t,t,t, t],
        r_list=[rv-rvmodel-gamma,fwhm, bis, asym],
        err_list=[rv_err,fwhm_err, bis_err,  asym_err],
        A=A, B=B, hp=hp, jitter=jitter
    )  

    if ll_rv != ll_rv:
        ll_rv = -1*np.inf
        
    return ll_rv

# ============================================================
# Log-posterior 
# ============================================================
def log_probability(theta, espresso_data, t0, per):
    
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta, espresso_data, t0, per)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll



# =================================
# ESPRESSO dataset 
# =================================
df_all = pd.read_csv(os.path.join(data_dir,"ESPRESSO/RV_data.txt"), sep='\s+')

df = df_all.dropna(subset=['rv_bart', 'rv_bart_err'])

t = df['bjd'] - 2457000

# true values
t0 = 2164.354
per = 40.383
true_krv = 0.0027
true_planet_signal = planetmodel(true_krv, t, t0, per)

df['rv_bart'] -= true_planet_signal

# synthetic planet signal
test_krv = 0.003
fake_signal = planetmodel(test_krv, t, t0, per)
# Inject RV signal
rv = (df['rv_bart'] - np.mean(df['rv_bart'])) + fake_signal
rv_err = df['rv_bart_err']

fwhm = df['fwhm_esp'] - np.mean(df['fwhm_esp'])
fwhm_err = df['fwhm_esp_err']

bis = df['bis'] - np.mean(df['bis'])
bis_err = df['bis_err']

asym = df['asym'] - np.mean(df['asym'])
asym_err = df['asym_err']


espresso_data = ({
        "time": t,
        "rv": rv,
        "rv_err": rv_err,
        "fwhm": fwhm,
        "fwhm_err": fwhm_err,
        "bis": bis, 
        "bis_err": bis_err,
        "asym": asym, 
        "asym_err": asym_err
    })



if __name__ == "__main__":


    # -----------------------------
    # Initialization and MCMC
    # -----------------------------
    ndim = 16
    niter = 100000
    nwalkers = 4*ndim

    init_guess = []
    while len(init_guess) < nwalkers:
        theta = [
            np.random.normal(0.002,0.001),      ### krv
            np.random.normal(0,0.001),          ### gamma
            np.random.normal(40,2),             ### lam
            np.random.normal(6.2,0.1),          ### prot
            np.random.normal(6.2,0.1),          ### w
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
                
                
        theta_lnlike = log_probability(theta, espresso_data, t0, per)
        if abs(theta_lnlike) < np.inf:
            init_guess.append(theta)

    chain_file = os.path.join(base_dir, "retrieval_chain.h5")        

    sampler, samples = emcee_run.emcee_adaptive_run(init_guess, nwalkers, niter, log_probability, 
                                                       (espresso_data, t0, per), 
                                                       filename=chain_file, nproc=nwalkers,
                                                       resume=True)


    # -----------------------------
    # Chains plot
    # -----------------------------
    fig, axes = plt.subplots(ndim, figsize=(10, 25), sharex=True)
    samples = sampler.get_chain()
    labels = ["krv", "gamma", "lam", "prot", "w", "RV_scale", "fwhm_scale",  "bis_scale", "asym_scale", 
              "RV_dGdt_scale", "FWHM_dGdt_scale", "asym_dGdt_scale", "rv_jitter", "fwhm_jitter", "bis_jitter", "asym_jitter"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3) 
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    chain_plot = os.path.join(base_dir, "chain_plot.pdf")
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
            "linewidths": 1.2, },
        hist_kwargs={
            "linewidth": 1.5,
            "color": "black",
            #"fill": True,
            "alpha": 0.6 },
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
    corner_plot = os.path.join(base_dir, "corner_plot.pdf")
    plt.savefig(corner_plot)
    plt.show()

    mcmc_results = os.path.join(base_dir, "mcmc_results.npz")
    np.savez(mcmc_results,
            chain=sampler.get_chain(),
            log_prob=sampler.get_log_prob())


    # -----------------------------
    # Best-fit parameters
    # -----------------------------
    labels = ["krv", "gamma", "lam", "P_{rot}", "w", "RV_{scale}", "fwhm_{scale}", "bis_{scale}", "asym_{scale}",
            "RV_{dGdt\,scale}", "FWHM_{dGdt \,scale}", "asym_{dGdt \,scale}", "rv_{jitter}", "fwhm_{jitter}",  "bis_{jitter}", "asym_{jitter}"]

    from IPython.display import display, Math

    theta_best = np.zeros(ndim)

    results = {
        "fitted_parameters": {}
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

        txt = "\mathrm{{{3}}} = {0:.7f}_{{-{1:.7f}}}^{{{2:.7f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))


    #==============================
    # BIC
    #==============================
    n = len(espresso_data["rv"]) + len(espresso_data["fwhm"]) +  len(espresso_data["bis"]) + len(espresso_data["asym"])
    # best log-likelihood
    best_ll = log_likelihood(theta_best, espresso_data, t0, per)
    k_trend = len(labels) 

    BIC_trend = k_trend * np.log(n) - 2 * best_ll
    print("BIC (trend counted):", BIC_trend)

    results["model_statistics"] = {
        "BIC": float(BIC_trend),
    }
    
    best_fit_results = os.path.join(base_dir, "best_fit_results.json")
    with open(best_fit_results, "w") as f:
        json.dump(results, f, indent=4)
