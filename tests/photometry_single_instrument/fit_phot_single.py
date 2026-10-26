import numpy as np
import matplotlib.pyplot as plt
import corner
import json
import seaborn as sns
import pickle
import os,sys
import emcee_run
import stats_phot_single as stats



"Global fit script for system LHS 6050 assuming a circular orbit. Joint analisys of photometric (TESS and ground based) and ESPRESSO datasets"
"Outputs: corner and chains plots, chains and posteriors files, best fit parameters json file"



data_dir = "/home/data" # path datsets
base_dir = "/home/hotometry_single_instrument/results" # path for saving outputs

# import datasets
with open(os.path.join(data_dir, "LHS6050b_complete_dataset.pkl"), "rb") as f:
    data_loaded = pickle.load(f)

data = data_loaded["lco_data"] # define here the instrument

t0 = 2459164.3524 - 2457000  # mid-transit time (BTJD)
per  = 40.3834187              # period (days)

if __name__ == "__main__":

    # -----------------------------
    # Initialization and MCMC
    # -----------------------------
    ndim = len(stats.PARAM_NAMES) 
    niter = 1000000
    nwalkers = 2*ndim

    init_guess = []
    while len(init_guess) < nwalkers:
        theta = [
            np.random.normal(t0, 1e-4),         ### t0
            np.random.normal(per, 1e-4),        ### per
            np.random.uniform(0.06, 0.10),      ### rp
            np.random.uniform(0.01, 0.9),        ### b
            np.random.uniform(50,200),        ### a/rs
            np.random.uniform(0.01, 0.9),        ### q1
            np.random.uniform(0.01, 0.9),        ### q2

            # add or remove line code based on the number of jitter
            np.random.uniform(0.0, 0.05), 
            np.random.uniform(0.0, 0.05),
            np.random.uniform(0.0, 0.05),
            np.random.uniform(0.0, 0.05),

            ]
                
                
        theta_lnlike = stats.log_posterior(theta, data, use_airmass=True, use_fwhm=False, use_centroid=False)
        if np.isfinite(theta_lnlike):
            init_guess.append(theta)

    chain_file = os.path.join(base_dir, "retrieval_chain.h5")        

    sampler, flat_samples = emcee_run.emcee_adaptive_run(init_guess, nwalkers, niter, stats.log_posterior, 
                                                       (data,), 
                                                       filename=chain_file, nproc=2*nwalkers,
                                                       resume=True)


    # -----------------------------
    # Chains plot
    # -----------------------------
    fig, axes = plt.subplots(ndim, figsize=(10, 25), sharex=True)
    samples = sampler.get_chain()
    labels = ["$t_0$", "P", "$R_p/R_s$", "b", "a", "q1", "q2", 
            # uncomment these parameters based on the instrument used
            'SS','Teid','CTIO', 'McD'  # LCOGT sloan i'
            # 'CTIOg'                  # LCOGT sloan g'
            # 'T2', 'T4', 'T5'         # Minerva-Australis
            # 'TRAPPIST'               # TRAPPIST-South
            # '20250812_412358', '20250812_412361', '20250812_412365', '20250812_412369', '20250812_412372', '20250812_412379',    # NGTS
            # '20251101_419475', '20251101_419478', '20251101_419481', '20251101_419484', '20251101_419487', '20251101_419492'
            ]
            
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

    colors = sns.color_palette("Greens", 10) 
    sns.palplot(colors) 
    theta_best = np.zeros(ndim)

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        theta_best[i] = mcmc[1]

    figure = corner.corner(
        flat_samples,
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

    from IPython.display import display, Math

    theta_best = np.zeros(ndim)

    results = {
        "fitted_parameters": {},
        "derived_parameters": {}
    }

    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
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
    t0 = flat_samples[:, 0]
    per = flat_samples[:, 1]
    rp = flat_samples[:, 2]
    b = flat_samples[:, 3]
    a_rstar = flat_samples[:, 4]


    cosi = b / a_rstar
    inc = np.degrees(np.arccos(cosi))


    derived_params = [
            (inc,       r"i [^\circ]")
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
    n = sum(len(sd["time"]) for sd in data) 
   
    # best log-likelihood
    best_ll = stats.log_likelihood(theta_best, data, use_airmass=True, use_fwhm=False, use_centroid=False)
    k_trend = len(labels) + 3*len(data) 

    BIC_trend = k_trend * np.log(n) - 2 * best_ll
    print("BIC (trend counted):", BIC_trend)

    results["model_statistics"] = {
        "BIC": float(BIC_trend),
    }
    
    best_fit_results = os.path.join(base_dir, "best_fit_results.json")
    with open(best_fit_results, "w") as f:
        json.dump(results, f, indent=4)

    

    #==============================
    # Final plot
    #==============================
    import matplotlib.gridspec as gridspec

    numb = ndim - 7 # number of datasets

    fig = plt.figure(figsize=(8, 4 * numb))

    outer_gs = gridspec.GridSpec(nrows=numb, ncols=1, hspace=0.35)

    for i, sector in enumerate(data):

        inner_gs = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer_gs[i], height_ratios=[numb, 1], hspace=0.0)

        ll, transit_bf, trend_bf = stats.ground_photometry_log_likelihood_dataset(theta_best, sector, 'q1', 'q2',use_airmass=True,use_fwhm=False,use_centroid=False)

        time = sector["time"]
        flux = sector["flux"]
        model = transit_bf * trend_bf
        residuals = flux - model

        # -----------------
        # Main panel
        # -----------------
        ax1 = fig.add_subplot(inner_gs[0])
        ax1.plot(time, flux, ".", ms=4, alpha=0.6, label="data")
        ax1.plot(time, model, "r-", lw=2, label="Best-fit")
        ax1.set_ylabel("Normalized Flux")
        ax1.set_title(f"Sector {sector['dataset']}")
        ax1.set_ylim(0.96, 1.03)
        ax1.legend()

        # -----------------
        # Residuals panel
        # -----------------
        ax2 = fig.add_subplot(inner_gs[1], sharex=ax1)
        ax2.plot(time, residuals, ".", ms=4, alpha=0.6)
        ax2.axhline(0, color="k", ls="--", lw=1)
        ax2.set_ylabel("Residuals")
        ax2.set_xlabel("Time (BTJD)")
        ax2.set_ylim(-0.01,0.01)

        plt.setp(ax1.get_xticklabels(), visible=False)

    final_plot = os.path.join(base_dir, "transit_final.png")
    plt.savefig(final_plot)
    plt.show()

