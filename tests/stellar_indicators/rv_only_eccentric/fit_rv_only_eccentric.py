import json
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import corner
import seaborn as sns
from IPython.display import display, Math
import stats_rv_only_eccentric as stats
import dynesty_run


data_dir = "/home/data"  # path datasets
base_dir = "/home/stellar_indicators_rv_only_eccentric/results"  # path for saving outputs

# Import datasets
with open(os.path.join(data_dir, "LHS6050b_complete_dataset.pkl"), "rb") as f:
    data_loaded = pickle.load(f)

espresso_data = data_loaded["espresso_data"]


if __name__ == "__main__":

    ndim = len(stats.PARAM_NAMES)  
    nlive = 1500  # Number of live points for dynesty
    checkpoint_file = os.path.join(base_dir, "dynesty_checkpoint.pkl")        

    # Run dynesty sampling
    results, samples = dynesty_run.dynesty_dynamic_run(
        ptform=stats.prior_transform, 
        lnlike=stats.log_likelihood, 
        ndim=ndim,
        args=(espresso_data,), 
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
    

    labels = stats.PARAM_NAMES

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
    n_main = ndim
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

    # save results
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
            "eff_pct": float(results.eff),
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
    h, k = samples[:, 0], samples[:, 1]

    e = h**2 + k**2
    w = np.degrees(np.arctan2(h, k))

    derived_params = [
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


    mgic_value = stats.compute_mgic(theta_best, espresso_data)

    results_fit["statistic_parameters"]["MGIC"] = {
         "value": float(mgic_value),
    }

    best_fit_results = os.path.join(base_dir, "best_fit_results_rv.json")
    with open(best_fit_results, "w") as f:
        json.dump(results_fit, f, indent=4)