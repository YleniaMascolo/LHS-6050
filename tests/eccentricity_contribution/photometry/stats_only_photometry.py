
import numpy as np
from numpy.linalg import lstsq
import multiGP
import batman
from scipy.special import ndtri



"""Statistical modules for the photometry fit of the LHS 6050 system.

This module presents the Bayesian core of the pipeline under the assumption of 
a eccentric orbit. It calculates boundary conditions through a log-prior, multi-instrument 
log-likelihoods, and combines them into the final log-posterior function. 
"""

PARAM_NAMES = [
    "t0", "per", "rp", "b", "h", "k", "mstar", "rstar",
    "q1_tess", "q2_tess", "q1_lco", "q2_lco", "q1_lcog", "q2_lcog", "q1_ngts", "q2_ngts", "q1_trap", "q2_trap", "q1_minerva", "q2_minerva", 
    "31", "70", "71", 'SS','Teid','CTIO', 'McD', 'CTIOg', '20250812_412358', '20250812_412361', '20250812_412365', '20250812_412369', '20250812_412372', '20250812_412379', 
    '20251101_419475', '20251101_419478', '20251101_419481', '20251101_419484', '20251101_419487', '20251101_419492', 'TRAPPIST', 'T2','T4','T5'
]

mstar_obs = 0.224
rstar_obs = 0.254
mstar_obs_err = 0.007
rstar_obs_err = 0.013

# -----------------------------
#  Quadratic limb darkening function (Kipping 2013)
# -----------------------------
def limb_darkening(q1, q2):

    sqrt_q1 = np.sqrt(q1)
    u1 = 2 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1 - 2*q2)
    
    return u1, u2


# -----------------------------
# BATMAN model function 
# -----------------------------
def transit_model(theta, t, ld_q1_key, ld_q2_key, supersample_factor=1, exp_time=0.0):

    # parameterization of eccentricity and omega
    e = theta["h"]**2 + theta["k"]**2
    w = np.degrees(np.arctan2(theta["h"], theta["k"]))  # batman expects degrees

    # period in years
    per_yr = theta["per"] / 365.25

    # a/Rs (dimensionless)
    a = 215.032 * (theta["mstar"]**(1/3)) * (per_yr**(2/3)) / theta["rstar"]

    # limb darkening 
    u1, u2 = limb_darkening(theta[ld_q1_key], theta[ld_q2_key])

    # parameterization of impact parameter for inclination
    cosi = theta["b"] / a
    cosi = np.clip(cosi, -1.0, 1.0)
    inc = np.degrees(np.arccos(cosi))

    params = batman.TransitParams()
    params.t0 = 0.0         #time of inferior conjunction (BJD)
    params.per = theta["per"]        #orbital period
    params.rp = theta["rp"]          #planet radius (in units of stellar radii Rp/Rs)
    params.a = a            #semi-major axis (in units of stellar radii a/Rs)
    params.inc = inc        #orbital inclination (in degrees)
    params.ecc = e          #eccentricity
    params.w = w            #longitude of periastron (in degrees)
    params.u = [u1,  u2]    #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic" #limb darkening model

    m = batman.TransitModel(params, t-theta["t0"], supersample_factor=supersample_factor, exp_time=exp_time) 
    return m.light_curve(params) 




# ============================================================
# Prior Transform for Dynesty
# ============================================================

def prior_transform(u):

    p = np.empty_like(u)

    p[0] = 2164.32 + u[0] * (2164.38 - 2164.32)  # t0
    p[1] = 40.3825 + u[1] * (40.384 - 40.3825)  # per
    p[2] = u[2] * 0.1  # rp 
    p[3] = u[3] * 1.0  # b (0 < b < 1)

    e_max = 0.9
    e = u[4] * e_max                 # e ~ Unif(0, e_max)
    omega_rad = u[5] * (2 * np.pi)   # omega ~ Unif(0, 2pi)
    r = np.sqrt(e)                   # e = h**2 + k**2 = r**2

    p[4] = r * np.sin(omega_rad)     # h
    p[5] = r * np.cos(omega_rad)     # k

    # Gaussian priors on mstar and rstar
    p[6] = mstar_obs + mstar_obs_err * ndtri(u[6])  # mstar
    p[7] = rstar_obs + rstar_obs_err * ndtri(u[7])  # rstar


    p[8] = u[8]  # q1_tess ([0,1])
    p[9] = u[9]  # q2_tess
    p[10] = u[10]  # q1_lco
    p[11] = u[11]  # q2_lco
    p[12] = u[12]  # q1_lcog
    p[13] = u[13]  # q2_lcog
    p[14] = u[14]  # q1_ngts
    p[15] = u[15]  # q2_ngts
    p[16] = u[16]  # q1_trap
    p[17] = u[17]  # q2_trap
    p[18] = u[18]  # q1_minerva
    p[19] = u[19]  # q2_minerva

    # --- Jitter photometry ---
    for i in range(20, len(u)):
        p[i] = u[i] * 0.1

    return p

# -----------Normalization function -----------------
def normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

# ============================================================
# TESS: Transit + trend log-likelihood
# ============================================================
def tess_transit_log_likelihood_sector(theta, sector_data):

    t = sector_data["time"]
    y = sector_data["flux"]
    yerr = sector_data["err"]
    
    jitter_key = str(sector_data["sector"])
    flux_jitter = theta[jitter_key]

    t_norm = normalization(t)
    A = np.column_stack([t_norm, np.ones_like(t_norm)])

    transit = transit_model(theta, t, ld_q1_key="q1_tess", ld_q2_key="q2_tess", supersample_factor=5, exp_time=0.0013888889)
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
    
    t = dataset["time"]
    y = dataset["flux"]
    yerr = dataset["err"]
    
    jitter_key = str(dataset["dataset"])
    flux_jitter = theta[jitter_key]

    t_norm = normalization(t)

    if quadratic_polynomial:
        cols = [t_norm**2, t_norm, np.ones_like(t_norm)]
    else:
        cols = [t_norm, np.ones_like(t_norm)]

    if use_airmass and ("airmass" in dataset):
        cols.append(normalization(dataset["airmass"]))

    A = np.column_stack(cols)

    if not np.all(np.isfinite(A)):
        return -np.inf, None, None

    transit = transit_model(theta, t, ld_q1_key=ld_q1_key, ld_q2_key=ld_q2_key)
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

    ll_dataset = -0.5 * np.sum(((y - model)**2) / var + np.log(2 * np.pi * var))

    return ll_dataset, transit, trend


# ============================================================
# LCO: Transit + trend log-likelihood
# ============================================================

def lco_transit_log_likelihood(theta, lco_data):
    ll = 0.0
    for dataset in lco_data:
        ll_sec, _, _ = ground_photometry_log_likelihood_dataset(
            theta, dataset, ld_q1_key="q1_lco", ld_q2_key="q2_lco", use_airmass=False, quadratic_polynomial=False
        )
        ll += ll_sec
    return ll

# ============================================================
# LCOg: Transit + trend log-likelihood
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
# NGTS: Transit + trend log-likelihood
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
# TRAPPIST: Transit + trend log-likelihood
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
# MINERVA: Transit + trend log-likelihood
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
# Total log-likelihood = transits
# ============================================================
def log_likelihood(theta, tess_data, lco_data, lcog_data, ngts_data, trap_data, minerva_data):

    theta = dict(zip(PARAM_NAMES, theta))

    if (theta["h"]**2 + theta["k"]**2) >= 0.9 or \
       theta["rp"] <= 0.0 or theta["mstar"] <= 0.0 or theta["rstar"] <= 0.0:
        return -np.inf

    tess_ll_transit = tess_transit_log_likelihood(theta, tess_data)
    lco_ll_transit  = lco_transit_log_likelihood(theta, lco_data)
    lcog_ll_transit = lcog_transit_log_likelihood(theta, lcog_data)
    ngts_ll_transit = ngts_transit_log_likelihood(theta, ngts_data)
    trap_ll_transit = trap_transit_log_likelihood(theta, trap_data)
    minerva_ll_transit = minerva_transit_log_likelihood(theta, minerva_data)

    return tess_ll_transit + lco_ll_transit + lcog_ll_transit + ngts_ll_transit + trap_ll_transit + minerva_ll_transit 
