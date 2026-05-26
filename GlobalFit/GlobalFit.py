import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import corner
import pandas as pd
import json
import models
import emcee_run
import stats
import seaborn as sns
import os



# ================================
# Load TESS sectors function
# ================================
def tess_sector(filename, t0, P, duration):
    """Upload TESS _lc.fits file and applies crowding correction, normalization and mask."""
    
    hdul = fits.open(filename)
    
    if "LIGHTCURVE" in hdul[1].name:
        hdu = hdul[1]
    else:
        raise ValueError("Light curve HDU not found")
    
    data = hdu.data
    hdr  = hdu.header

    quality_mask = (data["QUALITY"] == 0)
    
    # time, sap flux and sap flux error
    time = data["TIME"][quality_mask] # BTJD = BJD -2457000
    sap_flux = data["SAP_FLUX"][quality_mask]
    # sap_err  = data["SAP_FLUX_ERR"]
    
    # crowding correction
    crowdsap = hdr["CROWDSAP"]
    flux_corr = sap_flux / crowdsap
    # err_corr  = sap_err / crowdsap

    nanmask = np.isnan(time) + np.isnan(flux_corr) 
    time = time[~nanmask]
    flux_corr = flux_corr[~nanmask]

    # normalization
    norm = np.nanmedian(flux_corr)
    flux = flux_corr / norm
    # flux_err = err_corr / norm

    # # median of the absolute deviations as error
    # abs_dev = np.abs(flux_corr - norm)
    # mad = np.nanmedian(abs_dev)
    # flux_err = np.full_like(flux, 1.4826 * mad) / norm

    sigma = np.median(np.abs(np.diff(flux_corr)))
    flux_err = np.full_like(flux, 1.4826 * sigma) / norm 

    # mask transit 
    mask = mask_transits(time, t0, P, duration)
    time = time[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]

    return time, flux, flux_err

# ================================
# Mask transit function
# ================================
def mask_transits(time, t0, P, duration, factor=3):
   
    phase = (time - t0) % P
    window = factor * duration

    mask = (phase < window) | (phase > P - window)
    return mask


# ------------- TESS data ------------- 

sector_files = [
    "/home/u1182374/LHS6050/data/TESS/tess2020294194027-s0031-0000000337217173-0198-s/tess2020294194027-s0031-0000000337217173-0198-s_lc.fits",
    "/home/u1182374/LHS6050/data/TESS/tess2023263165758-s0070-0000000337217173-0265-s/tess2023263165758-s0070-0000000337217173-0265-s_lc.fits",
    "/home/u1182374/LHS6050/data/TESS/tess2023289093419-s0071-0000000337217173-0266-s/tess2023289093419-s0071-0000000337217173-0266-s_lc.fits"
]

t0 = 2459164.3524 - 2457000  # mid-transit time (BTJD)
per  = 40.3834187              # period (days)
duration = 2.148/24          # transit duration (days)

labels = [31,70,71]

tess_data = []
for  i, f in enumerate(sector_files, start=1):
    time_tess, flux_tess, flux_err_tess = tess_sector(f, t0, per, duration)

    tess_data.append({
        "sector": labels[i-1],
        "time": time_tess,
        "flux": flux_tess,
        "err": flux_err_tess
    })


# ==================================
# LCO and TRAPPIST: Load dataset function
# ==================================
def ground_dataset(filename):

    # load datasets
    df = pd.read_csv(filename, sep='\s+') 

    # Define data array 
    time = np.array(df['BJD_TDB']-2457000) # time in BJD - 2457000
    flux_raw = np.array(df['rel_flux_T1'])
    airmass = np.array(df['AIRMASS'])

    # mask nan data 
    nanmask = np.isnan(time) + np.isnan(flux_raw) + np.isnan(airmass)
    time = time[~nanmask]
    flux_mask = flux_raw[~nanmask]
    airmass = airmass[~nanmask]

    median_flux = np.nanmedian(flux_mask) 

    # # normalize data using median
    flux = flux_mask/median_flux

    # # median of the absolute deviations as error
    # abs_dev = np.abs(flux_mask - median_flux)
    # mad = np.nanmedian(abs_dev)
    sigma = np.median(np.abs(np.diff(flux_mask)))
    flux_err = np.full_like(flux, 1.4826 * sigma) / median_flux 

    return time, flux, flux_err, airmass  


# ----------- LCO data ---------------

lco_files = [
    "/home/u1182374/LHS6050/data/LCO/TIC_337217173-01_20250922_LCO-SS-1.0m_ip_15pix_measurements.tbl",
    "/home/u1182374/LHS6050/data/LCO/TIC337217173-11_20240924_LCO-Teid-1m0_ip_measurements.tbl",
    "/home/u1182374/LHS6050/data/LCO/TIC337217173-11_20241214_LCO-CTIO-1m0_ip_12px_Measurements.tbl",
    "/home/u1182374/LHS6050/data/LCO/TIC337217173-11_20241214_LCO-McD-1m0_ip_9px_Measurements.tbl"
]

labels = ['SS','Teid','CTIO', 'McD']

lco_data = []
for  i, f in enumerate(lco_files, start=1):
    time_lco, flux_lco, flux_err_lco, airmass_lco = ground_dataset(f)

    lco_data.append({
    "dataset": labels[i-1],
    "time": time_lco,
    "flux": flux_lco,
    "err": flux_err_lco,
    "airmass": airmass_lco
    })


# ------------- LCO data g filter ---------------

lcog_files = [
    "/home/u1182374/LHS6050/data/LCO/TIC_337217173-01_20251102_LCO-CTIO-1.0m_gp_12pix_measurements.tbl",
]

labels = ['CTIOg']

lcog_data = []
for  i, f in enumerate(lcog_files, start=1):
    time_lcog, flux_lcog, flux_err_lcog, airmass_lcog = ground_dataset(f)

    lcog_data.append({
    "dataset": labels[i-1],
    "time": time_lcog,
    "flux": flux_lcog,
    "err": flux_err_lcog,
    "airmass": airmass_lcog
    })


# --------------- TRAPPIST data ----------------

trap_files = [
    "/home/u1182374/LHS6050/data/TIC_337217173-01_20251102_TRAPPIST-South-0.6m_Rc_12pix_measurements.tbl",
]

labels = ['TRAPPIST']

trap_data = []
for  i, f in enumerate(trap_files, start=1):
    time_trap, flux_trap, flux_err_trap, _ = ground_dataset(f)

    trap_data.append({
    "dataset": labels[i-1],
    "time": time_trap,
    "flux": flux_trap,
    "err": flux_err_trap
    })



# ================================
# NGTS and MINERVA: Load data function
# ================================
def ground2_dataset(filename):
    
    data = np.genfromtxt(filename)  
        
    time = data[:,0] - 2457000 # time in BJD - 2457000
    flux_raw = data[:,1]
    airmass = data[:,3]

    # mask nan data 
    nanmask = np.isnan(time) + np.isnan(flux_raw) + np.isnan(airmass)
    time = time[~nanmask]
    flux_mask = flux_raw[~nanmask]
    airmass = airmass[~nanmask]

    median_flux = np.nanmedian(flux_mask) 

    # # normalize data using median
    flux = flux_mask/median_flux

    sigma = np.median(np.abs(np.diff(flux_mask)))
    flux_err = np.full_like(flux, 1.4826 * sigma) / median_flux 

    return time, flux, flux_err, airmass


# ---------------- Minerva data ------------------

files = [
    "/home/u1182374/LHS6050/data/MINERVA/minerva_T2.dat",
    "/home/u1182374/LHS6050/data/MINERVA/minerva_T4.dat",
    "/home/u1182374/LHS6050/data/MINERVA/minerva_T5.dat"
]

labels = ['T2','T4','T5']

minerva_data = []
for  i, f in enumerate(files, start=1):
    time, flux, err, airmass = ground2_dataset(f)

    minerva_data.append({
        "dataset": labels[i-1],
        "time": time,
        "flux": flux,
        "err": err,
        "airmass": airmass
    })


# ------------- NGTS data ----------------

files = [
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20250812_412358.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20250812_412361.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20250812_412365.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20250812_412369.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20250812_412372.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20250812_412379.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20251101_419475.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20251101_419478.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20251101_419481.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20251101_419484.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20251101_419487.dat",
    "/home/u1182374/LHS6050/data/NGTS/NGTS_20251101_419492.dat"
]

labels = ['20250812_412358', '20250812_412361', '20250812_412365', '20250812_412369', '20250812_412372', '20250812_412379', 
          '20251101_419475', '20251101_419478', '20251101_419481', '20251101_419484', '20251101_419487', '20251101_419492']

ngts_data = []
for  i, f in enumerate(files, start=1):
    time, flux, err, airmass = ground2_dataset(f)

    ngts_data.append({
        "dataset": labels[i-1],
        "time": time,
        "flux": flux,
        "err": err,
        "airmass": airmass
    })



# =================================
# ESPRESSO dataset 
# =================================
df_all = pd.read_csv("/home/u1182374/LHS6050/data/ESPRESSO/RV_data.txt", sep='\s+')

df = df_all.dropna(subset=['rv_bart', 'rv_bart_err'])

t = df['bjd'] - 2457000

rv = df['rv_bart'] - np.mean(df['rv_bart'])
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

    base_dir = "/home/u1182374/LHS6050/GlobalFit/pyGlobalFit_circular_uniform_ld_600_new"

    # -----------------------------
    # Initialization and MCMC
    # -----------------------------
    ndim = 34
    niter = 120000
    nwalkers = 2*ndim

    init_guess = []
    while len(init_guess) < nwalkers:
        theta = [
            np.random.normal(t0, 1e-4),
            np.random.normal(per, 1e-4),
            np.random.uniform(0.07, 0.10),
            np.random.uniform(0.0, 1.0),
            np.random.normal(0.22,0.01),   
            np.random.normal(0.25,0.01), 
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.uniform(0.0, 1.0),
            np.random.normal(0.003,0.001),      ### krv
            np.random.normal(0,0.001),          ### gamma
            np.random.normal(100,2),             ### lam
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

    chain_file = os.path.join(base_dir, "retrieval_chain.h5")        

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

    plt.rcParams.update({
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
    })

    figure = corner.corner(
        samples,
        labels=labels,

        # confidence intervals standard
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",

        # smoothing 
        smooth=1.0,
        smooth1d=1.0,

        # levels = 0.5σ, 1σ, 1.5σ, 2σ
        levels = (0.393, 0.68, 0.86, 0.95),  

        # colors and contours
        color=colors[9],
        fill_contours=True,
        plot_datapoints=False,

        contour_kwargs={
            "colors": "black",
            "linewidths": 1.2,
        },

        hist_kwargs={
            "linewidth": 1.5,
            "color": "black",
            #"fill": True,
            "alpha": 0.6
        },
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

    a = ((per / 365.25)**(2/3) * mstar**(1/3) * 215.032) / rstar

    Rsun_to_Rearth = 109.076
    Rp_Rearth = rp * rstar * Rsun_to_Rearth 

    cosi = b / a
    inc = np.degrees(np.arccos(cosi))
    sini = np.sin(np.radians(inc))

    G = 6.67430e-11
    M_sun = 1.9885e30
    day2sec = 86400
    M_earth = 5.972e24

    M_star = mstar * M_sun
    per_sec = per * day2sec

    Mp = (krv * (M_star**(2/3)) * ((per_sec / (2*np.pi*G))**(1/3))) / sini

    Mpe = (Mp / M_earth)*1e3

    derived_params = [
        (Rp_Rearth, r"R_p\,[R_\oplus]"),
        (inc,       r"i\,[^\circ]"),
        (a,         r"a/R_\star\,"),
        (Mpe,       r"M_p\,[M_\oplus]")
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
    
    best_fit_results = os.path.join(base_dir, "best_fit_results.json")
    with open(best_fit_results, "w") as f:
        json.dump(results, f, indent=4)
