import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import corner
import seaborn as sns
from IPython.display import display, Math
import stats_only_photometry as stats
import dynesty_run



data_dir = "/home/data"  # path datasets
base_dir = "/home/tests/eccentricity_contribution/photometry/results"  # path for saving outputs

# Import datasets
with open(os.path.join(data_dir, "LHS6050b_complete_dataset.pkl"), "rb") as f:
    data_loaded = pickle.load(f)

tess_data = data_loaded["tess_data"]
lco_data = data_loaded["lco_data"]
lcog_data = data_loaded["lcog_data"]
trap_data = data_loaded["trap_data"]
ngts_data = data_loaded["ngts_data"]
minerva_data = data_loaded["minerva_data"]



if __name__ == "__main__":

    ndim = len(stats.PARAM_NAMES) 
    nlive = 1000  # Number of live points for dynesty
    checkpoint_file = os.path.join(base_dir, "dynesty_checkpoint.pkl")        

    # Run dynesty sampling
    results, samples = dynesty_run.dynesty_dynamic_run(
        ptform=stats.prior_transform, 
        lnlike=stats.log_likelihood, 
        ndim=ndim,
        args=(tess_data, lco_data, lcog_data, ngts_data, trap_data, minerva_data,), 
        filename=checkpoint_file, 
        nproc=2 * ndim,
        nlive=nlive,
        dlogz_threshold=0.01,
        resume=True
    )

    print("\n" + "=" * 40)
    print(" DYNESTY SAMPLING SUMMARY")
    print("=" * 40)
    results.summary() 
    print("=" * 40 + "\n")

    # -----------------------------
    # Trace plot (Dynesty built-in)
    # -----------------------------
    from dynesty import plotting as dyplot
    
    labels = [
        "$t_0$", "P", "$R_p/R_s$", "b", "h", "k", "$m_*$", "$r_*$",
        "$q1_{tess}$", "$q2_{tess}$", "$q1_{lco}$", "$q2_{lco}$", "$q1_{lcog}$", "$q2_{lcog}$",
        "$q1_{ngts}$", "$q2_{ngts}$", "$q1_{trap}$", "$q2_{trap}$", "$q1_{minerva}$", "$q2_{minerva}$",
        "$\sigma_{TESS,31}$", "$\sigma_{TESS,70}$", "$\sigma_{TESS,71}$",
        "$\sigma_{SS}$", "$\sigma_{Teid}$", "$\sigma_{CTIO}$", "$\sigma_{McD}$", "$\sigma_{CTIOg}$", 
        "$\sigma_{20250812\_412358}$", "$\sigma_{20250812\_412361}$", "$\sigma_{20250812\_412365}$",
        "$\sigma_{20250812\_412369}$", "$\sigma_{20250812\_412372}$", "$\sigma_{20250812\_412379}$", 
        "$\sigma_{20251101\_419475}$", "$\sigma_{20251101\_419478}$", "$\sigma_{20251101\_419481}$",
        "$\sigma_{20251101\_419484}$", "$\sigma_{20251101\_419487}$", "$\sigma_{20251101\_419492}$",
        "$\sigma_{TRAPPIST}$", "$\sigma_{T2}$", "$\sigma_{T4}$", "$\sigma_{T5}$"
    ]

    fig, axes = dyplot.traceplot(results, labels=labels, fig=plt.subplots(ndim, 2, figsize=(12, 2 * ndim)))
    plt.tight_layout()
    chain_plot = os.path.join(base_dir, "chain_plot_eccentric.pdf")
    plt.savefig(chain_plot)
    plt.show()

    # -----------------------------
    # Corner plot
    # -----------------------------
    colors = sns.color_palette("Greens", 10) 
    
    theta_best = np.zeros(ndim)
    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        theta_best[i] = mcmc[1]

    # Plot only main parameters
    n_main = len(stats.PARAM_NAMES) 
    samples_main = samples[:, :n_main]
    labels_main = labels[:n_main]
    theta_best_main = theta_best[:n_main]

    figure = corner.corner(
        samples_main,
        labels=labels_main,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f", 
        smooth=1.0,
        smooth1d=1.0,
        levels=(0.393, 0.68, 0.86, 0.95),  
        color=colors[9],
        fill_contours=True,
        plot_datapoints=False,
        contour_kwargs={"colors": "black", "linewidths": 1.2},
        hist_kwargs={"linewidth": 1.5, "color": "black", "alpha": 0.6},
    )

    corner.overplot_lines(figure, theta_best_main, color="k", lw=1.5)
    corner.overplot_points(figure, theta_best_main[None], marker="s", color="k", markersize=4)

    for ax in figure.get_axes():
        ax.tick_params(direction="in", top=True, right=True)

    plt.tight_layout()
    corner_plot = os.path.join(base_dir, "corner_plot_eccentric.pdf")
    plt.savefig(corner_plot)
    plt.show()

    dynesty_results_file = os.path.join(base_dir, "dynesty_results.npz")
    np.savez(dynesty_results_file,
             samples=results.samples,
             logl=results.logl,
             logwt=results.logwt,
             flat_samples=samples)

    # -----------------------------
    # Best-fit parameters JSON
    # -----------------------------
    json_labels = stats.PARAM_NAMES

    results_fit = {
        "fitted_parameters": {},
        "derived_parameters": {},
        "statistic_parameters": {
            "logZ": float(results.logz[-1]),
            "dlogZ": float(results.logzerr[-1]),
            "nlive": nlive,
            "niter": int(results.niter),
            "eff_pct": float(results.eff)
        }
    }

    for i in range(ndim):
        mcmc = np.percentile(samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)

        results_fit["fitted_parameters"][json_labels[i]] = {
            "median": float(mcmc[1]),
            "err_minus": float(q[0]),
            "err_plus": float(q[1])
        }

        txt = r"\mathrm{{{3}}} = {0:.5f}_{{-{1:.5f}}}^{{+{2:.5f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], json_labels[i])
        display(Math(txt))

    # -----------------------------
    # Derived parameters (from chains)
    # -----------------------------
    t0, per, rp, b, h, k, mstar, rstar = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3], samples[:, 4], samples[:, 5], samples[:, 6], samples[:, 7]

    a = ((per / 365.25)**(2/3) * mstar**(1/3))
    a_rstar = (a * 215.032) / rstar

    Rsun_to_Rearth = 109.076
    Rp_Rearth = rp * rstar * Rsun_to_Rearth

    e = h**2 + k**2
    w = np.degrees(np.arctan2(h, k))


    derived_params = [
        (Rp_Rearth, r"R_p\,[R_\oplus]"),
        (a,         r"a (AU)"),
        (a_rstar,   r"a/R_\star"),
        (e,         r"e"),
        (w,         r"\omega\,[^\circ]")
    ]

    for values, label in derived_params:
        mcmc = np.percentile(values, [16, 50, 84])
        q = np.diff(mcmc)

        results_fit["derived_parameters"][label] = {
            "median": float(mcmc[1]),
            "err_minus": float(q[0]),
            "err_plus": float(q[1])
        }

        txt = r"{} = {:.3f}_{{-{:.5f}}}^{{+{:.5f}}}".format(label, mcmc[1], q[0], q[1])
        display(Math(txt))

    best_fit_results = os.path.join(base_dir, "best_fit_results_photometry.json")
    with open(best_fit_results, "w") as f:
        json.dump(results_fit, f, indent=4)