
import numpy as np
from numpy.linalg import lstsq
import multiGP
import models_circular as models
from scipy.special import ndtri


"""
Statistical modules for the joint global fit of the LHS 6050 system.

This module presents the Bayesian core of the pipeline under the assumption of 
a circular orbit. It calculates boundary conditions through a log-prior, multi-instrument 
log-likelihoods, and combines them into the final log-posterior function. 

"""


PARAM_NAMES = [
    "t0", "per", "rp", "b", "mstar", "rstar",
    "q1_tess", "q2_tess", "q1_lco", "q2_lco", "q1_lcog", "q2_lcog", "q1_ngts", "q2_ngts", "q1_trap", "q2_trap", "q1_minerva", "q2_minerva",
    "krv", "rv_gamma", "lam", "prot", "w","rv_scale", "fwhm_scale", "rv_dGdt_scale","rv_jitter", "fwhm_jitter", "fwhm_gamma", 
    "31", "70", "71", 'SS','Teid','CTIO', 'McD', 'CTIOg', '20250812_412358', '20250812_412361', '20250812_412365', '20250812_412369', '20250812_412372', '20250812_412379', 
    '20251101_419475', '20251101_419478', '20251101_419481', '20251101_419484', '20251101_419487', '20251101_419492', 'TRAPPIST', "T2", "T4", "T5"
]



mstar_obs = 0.224
rstar_obs = 0.254
mstar_obs_err = 0.007
rstar_obs_err = 0.013
prot_transit = 6.2326
prot_transit_err = 0.1314


# ============================================================
# Prior Transform for Dynesty
# ============================================================

def prior_transform(u):

    p = np.empty_like(u)

    p[0] = 2164.32 + u[0] * (2164.38 - 2164.32)  # t0
    p[1] = 40.3825 + u[1] * (40.384 - 40.3825)  # per
    p[2] = u[2] * 0.1  # rp 
    p[3] = u[3] * 1.0  # b (0 < b < 1)

    # Gaussian priors on mstar and rstar
    p[4] = mstar_obs + mstar_obs_err * ndtri(u[4])  # mstar
    p[5] = rstar_obs + rstar_obs_err * ndtri(u[5])  # rstar

    # --- LD photometry ---
    for i in range(6, 18):
        p[i] = u[i] 

    p[18] = u[18] * 0.1  # krv (0 < krv < 0.1)
    p[19] = -0.05 + u[19] * (0.05 - (-0.05))  # rv_gamma

    p[20] = 6.5 + u[20] * (600.0 - 6.5)  # lam 
    
    # Gaussian prior on prot
    p[21] = prot_transit + prot_transit_err * ndtri(u[21])  # prot

    p[22] = 0.01 + u[22] * (6.5 - 0.01)  # w

    p[23] = -0.05 + u[23] * (0.05 - (-0.05))  # rv_scale
    p[24] = u[24] * 0.1  # fwhm_scale
    p[25] = -0.1 + u[25] * (0.1 - (-0.1))  # rv_dGdt_scale

    # --- Jitter RV ---
    p[26] = u[26] * 0.02  # rv_jitter
    p[27] = u[27] * 0.02  # fwhm_jitter

    # --- Gamma Indicator ---
    p[28] = -0.1 + u[28] * (0.1 - (-0.1))  # fwhm_gamma

    # --- Jitter photometry ---
    for i in range(29, len(u)):
        p[i] = u[i] * 0.1

    return p

# -----------Normalization function -----------------
def normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# ============================================================
# TESS: Transit + trend log-likelihood
# ============================================================
def tess_transit_log_likelihood_sector(theta, sector_data):

    theta_tess = (
        theta["t0"], theta["per"], theta["rp"], theta["b"],
        theta["mstar"], theta["rstar"], theta["q1_tess"], theta["q2_tess"]
    )

    t = sector_data["time"]
    y = sector_data["flux"]
    yerr = sector_data["err"]
    
    jitter_key = str(sector_data["sector"])
    flux_jitter = theta[jitter_key]

    t_norm = normalization(t)
    A = np.column_stack([t_norm, np.ones_like(t_norm)])

    transit = models.transit_model(theta_tess, t, supersample_factor=5, exp_time=0.0013888889)
    if (np.std(transit) < 1e-5) or not np.all(np.isfinite(transit)) or np.any(transit <= 0):
        return -np.inf, None, None

    residuals = y / transit
    if not np.all(np.isfinite(residuals)):
        return -np.inf, None, None

    coeffs_fit, _, _, _ = lstsq(A, residuals, rcond=None)
    if not np.all(np.isfinite(coeffs_fit)):
        return -np.inf, None, None

    trend = A @ coeffs_fit
    model = transit * trend

    if not np.all(np.isfinite(model)):
        return -np.inf, None, None

    var = yerr**2 + flux_jitter**2

    ll_transit_sector = -0.5 * np.sum(((y - model)**2) / var + np.log(2 * np.pi * var))

    return ll_transit_sector, transit, trend


def tess_transit_log_likelihood(theta, tess_data):
    tess_ll_transit = 0.0
    for sector in tess_data:
        tess_ll_sec, _, _ = tess_transit_log_likelihood_sector(theta, sector)
        tess_ll_transit += tess_ll_sec
    return tess_ll_transit


# ============================================================
# Groud Based: Transit + trend log-likelihood
# ============================================================
def ground_photometry_log_likelihood_dataset(theta, dataset, ld_q1_key, ld_q2_key, use_airmass=True, quadratic_polynomial=True):
    
    theta_inst = (
        theta["t0"], theta["per"], theta["rp"], theta["b"], 
        theta["mstar"], theta["rstar"], theta[ld_q1_key], theta[ld_q2_key]
    )

    t = dataset["time"]
    y = dataset["flux"]
    yerr = dataset["err"]
    
    jitter_key = str(dataset["dataset"])
    flux_jitter = theta[jitter_key]

    t_norm = normalization(t)

    # check for time polynomial
    if quadratic_polynomial:
        cols = [t_norm**2, t_norm, np.ones_like(t_norm)]
    else:
        cols = [t_norm, np.ones_like(t_norm)]

    # check for airmass
    if use_airmass and ("airmass" in dataset):
        cols.append(normalization(dataset["airmass"]))

    A = np.column_stack(cols)

    if not np.all(np.isfinite(A)):
        return -np.inf, None, None

    transit = models.transit_model(theta_inst, t)
    if (np.std(transit) < 1e-5) or not np.all(np.isfinite(transit)) or np.any(transit <= 0):
        return -np.inf, None, None

    residuals = y / transit
    if not np.all(np.isfinite(residuals)):
        return -np.inf, None, None

    coeffs_fit, _, _, _ = lstsq(A, residuals, rcond=None)
    if not np.all(np.isfinite(coeffs_fit)):
        return -np.inf, None, None

    trend = A @ coeffs_fit
    model = transit * trend

    if not np.all(np.isfinite(model)):
        return -np.inf, None, None

    # Likelihood with jitter
    var = yerr**2 + flux_jitter**2

    ll_dataset = -0.5 * np.sum(((y - model)**2) / var + np.log(2 * np.pi * var))

    return ll_dataset, transit, trend


# ============================================================
# LCO: log-likelihood
# ============================================================

def lco_transit_log_likelihood(theta, lco_data):
    ll = 0.0
    for dataset in lco_data:
        ll_sec, _, _ = ground_photometry_log_likelihood_dataset(
            theta, dataset, ld_q1_key="q1_lco", ld_q2_key="q2_lco", use_airmass=False,quadratic_polynomial=False
        )
        ll += ll_sec
    return ll

# ============================================================
# LCOg: log-likelihood
# ============================================================

def lcog_transit_log_likelihood(theta, lcog_data):
    ll = 0.0
    for dataset in lcog_data:
        ll_sec, _, _ = ground_photometry_log_likelihood_dataset(
            theta, dataset, ld_q1_key="q1_lcog", ld_q2_key="q2_lcog", use_airmass=False, quadratic_polynomial=True
        )
        ll += ll_sec
    return ll

# ============================================================
# NGTS: log-likelihood
# ============================================================

def ngts_transit_log_likelihood(theta, ngts_data):
    ll = 0.0
    for dataset in ngts_data:
        ll_sec, _, _ = ground_photometry_log_likelihood_dataset(
            theta, dataset, ld_q1_key="q1_ngts", ld_q2_key="q2_ngts", use_airmass=True, quadratic_polynomial=True
        )
        ll += ll_sec
    return ll

# ============================================================
# TRAPPIST: log-likelihood
# ============================================================

def trap_transit_log_likelihood(theta, trap_data):
    ll = 0.0
    for dataset in trap_data:
        ll_sec, _, _ = ground_photometry_log_likelihood_dataset(
            theta, dataset, ld_q1_key="q1_trap", ld_q2_key="q2_trap", use_airmass=False, quadratic_polynomial=False
        )
        ll += ll_sec
    return ll

# ============================================================
# MINERVA: log-likelihood
# ============================================================

def minerva_transit_log_likelihood(theta, minerva_data):
    ll = 0.0
    for dataset in minerva_data:
        ll_sec, _, _ = ground_photometry_log_likelihood_dataset(
            theta, dataset, ld_q1_key="q1_minerva", ld_q2_key="q2_minerva", use_airmass=True, quadratic_polynomial=False
        )
        ll += ll_sec
    return ll


# ============================================================
# ESPRESSO: Multidimension GP log-likelihood
# ============================================================
def espresso_rv_log_likelihood(theta, espresso_data, indicators=["rv", "fwhm"]):
    if not isinstance(theta, dict):
        theta = {name: val for name, val in zip(PARAM_NAMES, theta)}

    theta_espresso = (
        theta["t0"], theta["per"], theta["krv"], theta["rv_gamma"]
    )

    t = espresso_data["time"]
    
    rvmodel = models.planetmodel(theta_espresso,  t)

    t_list = []
    r_list = []
    err_list = []
    A_list = []
    B_list = []
    jitter_list = []

    for ind in indicators:
        t_list.append(t)
        err_list.append(espresso_data[f"{ind}_err"])
        A_list.append(theta[f"{ind}_scale"])
        B_list.append(theta.get(f"{ind}_dGdt_scale", 0.0))
        jitter_list.append(theta[f"{ind}_jitter"])

        if ind == "rv":
            r_list.append(espresso_data["rv"] - rvmodel)
        else:
            r_list.append(espresso_data[ind] - theta[f"{ind}_gamma"])

    hp = multiGP.QPHyperParams(amp=1, lam=theta["lam"], P=theta["prot"], w=theta["w"])

    # Likelihood 
    ll = multiGP.lnlike_multidim_qp_gp(
        t_list=t_list,
        r_list=r_list,
        err_list=err_list,
        A=np.array(A_list), 
        B=np.array(B_list), 
        hp=hp, 
        jitter=np.array(jitter_list)
    )

    return -np.inf if np.isnan(ll) else ll


def compute_mgic(theta, espresso_data, indicators=["rv", "fwhm"]):
    if not isinstance(theta, dict):
        theta = {name: val for name, val in zip(PARAM_NAMES, theta)}
    
    theta_espresso = (
        theta["t0"], theta["per"], theta["krv"], theta["rv_gamma"]
    )

    t = espresso_data["time"]
    
    rvmodel = models.planetmodel(theta_espresso,  t)

    t_list, r_list, err_list = [], [], []
    A_list, B_list, jitter_list = [], [], []

    for ind in indicators:
        t_list.append(t)
        err_list.append(espresso_data[f"{ind}_err"])
        A_list.append(theta[f"{ind}_scale"])
        jitter_list.append(theta[f"{ind}_jitter"])
        B_list.append(theta.get(f"{ind}_dGdt_scale", 0.0))

        if ind == "rv":
            r_list.append(espresso_data["rv"] - rvmodel)
        else:
            r_list.append(espresso_data[ind] - theta[f"{ind}_gamma"])

    hp = multiGP.QPHyperParams(amp=1, lam=theta["lam"], P=theta["prot"], w=theta["w"])

    return multiGP.compute_mgic_rv(
        t_list=t_list,
        r_list=r_list,
        err_list=err_list,
        A=np.array(A_list),
        B=np.array(B_list),
        hp=hp,
        jitter=np.array(jitter_list),
        k_pars=13 # number of RV parameters
    )


# ============================================================
# Total log-likelihood = transits + RV
# ============================================================
def log_likelihood(theta, tess_data, lco_data, lcog_data, ngts_data, trap_data, minerva_data, espresso_data):

    theta = dict(zip(PARAM_NAMES, theta))

    if theta["rp"] <= 0.0 or theta["mstar"] <= 0.0 or theta["rstar"] <= 0.0:
        return -np.inf

    tess_ll_transit = tess_transit_log_likelihood(theta, tess_data)
    lco_ll_transit  = lco_transit_log_likelihood(theta, lco_data)
    lcog_ll_transit = lcog_transit_log_likelihood(theta, lcog_data)
    ngts_ll_transit = ngts_transit_log_likelihood(theta, ngts_data)
    trap_ll_transit = trap_transit_log_likelihood(theta, trap_data)
    minerva_ll_transit = minerva_transit_log_likelihood(theta, minerva_data)
    
    ll_rv   = espresso_rv_log_likelihood(theta, espresso_data)  

    return tess_ll_transit + lco_ll_transit + lcog_ll_transit + ngts_ll_transit + trap_ll_transit + minerva_ll_transit + ll_rv 
