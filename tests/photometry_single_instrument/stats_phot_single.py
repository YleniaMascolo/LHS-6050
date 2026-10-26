
import numpy as np
from numpy.linalg import lstsq
import batman



"""Statistical modules for the photometric fit using only one instrument of the LHS 6050 system.

This module presents the Bayesian core of the pipeline under the assumption of 
a eccentric orbit. It calculates boundary conditions through a log-prior, multi-instrument 
log-likelihoods, and combines them into the final log-posterior function. 
"""

PARAM_NAMES = [
    "t0", "per", "rp", "b", 'a', "q1", "q2", 
    # uncomment these parameters based on the instrument used
    'SS','Teid','CTIO', 'McD'  # LCOGT sloan i'
    # 'CTIOg'                  # LCOGT sloan g'
    # 'T2', 'T4', 'T5'         # Minerva-Australis
    # 'TRAPPIST'               # TRAPPIST-South
    # '20250812_412358', '20250812_412361', '20250812_412365', '20250812_412369', '20250812_412372', '20250812_412379',    # NGTS
    # '20251101_419475', '20251101_419478', '20251101_419481', '20251101_419484', '20251101_419487', '20251101_419492'
]

t0_obs = 2164.3548
t0_obs_err = 0.001
per_obs = 40.383386
per_obs_err = 0.000028

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


    # limb darkening 
    u1, u2 = limb_darkening(theta[ld_q1_key], theta[ld_q2_key])

    # parameterization of impact parameter for inclination
    cosi = theta["b"] / theta["a"]
    cosi = np.clip(cosi, -1.0, 1.0)
    inc = np.degrees(np.arccos(cosi))

    params = batman.TransitParams()
    params.t0 = 0.0         #time of inferior conjunction (BJD)
    params.per = theta["per"]        #orbital period
    params.rp = theta["rp"]          #planet radius (in units of stellar radii Rp/Rs)
    params.a = theta["a"]            #semi-major axis (in units of stellar radii a/Rs)
    params.inc = inc        #orbital inclination (in degrees)
    params.ecc = 0          #eccentricity
    params.w = 90            #longitude of periastron (in degrees)
    params.u = [u1,  u2]    #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic" #limb darkening model

    m = batman.TransitModel(params, t-theta["t0"], supersample_factor=supersample_factor, exp_time=exp_time) 
    return m.light_curve(params) 


def log_prior(p):
    p = np.asarray(p)
    t0, per, rp, b, a, q1, q2 = p[:7]
    jitter = p[7:]

    # uniform prior    
    if not (0.0 <= rp <= 0.2): return -np.inf
    if not (0.0 <= b <= 1.0): return -np.inf
    if not (a > 0.0): return -np.inf
    if not (0.0 <= q1 <= 1.0): return -np.inf
    if not (0.0 <= q2 <= 1.0): return -np.inf
    if np.any(jitter < 0.0) or np.any(jitter > 0.1): return -np.inf

    # gaussian prior
    lp_t0 = -0.5 * ((t0 - t0_obs) / t0_obs_err) ** 2 
    lp_per = -0.5 * ((per - per_obs) / per_obs_err) ** 2 

    return lp_t0 + lp_per

# -----------Normalization function -----------------
def normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# ============================================================
# Groud Based: Transit + trend log-likelihood
# ============================================================
def ground_photometry_log_likelihood_dataset(theta, dataset, ld_q1_key, ld_q2_key, use_airmass=True, use_fwhm=False, use_centroid=False):

    if not isinstance(theta, dict):
        theta = dict(zip(PARAM_NAMES, theta))
    
    t = dataset["time"]
    y = dataset["flux"]
    yerr = dataset["err"]

    
    jitter_key = str(dataset["dataset"])
    flux_jitter = theta[jitter_key]

    t_norm = normalization(t)

    cols = [t_norm, np.ones_like(t_norm)]

    if use_airmass and ("airmass" in dataset):
        cols.append(normalization(dataset["airmass"]))
    if use_fwhm and ("fwhm" in dataset):
        cols.append(normalization(dataset["fwhm"]))
    if use_centroid and ("x_pos" in dataset) and ("y_pos" in dataset):
        cols.append(normalization(dataset["x_pos"]))
        cols.append(normalization(dataset["y_pos"]))

    A = np.column_stack(cols)

    if not np.all(np.isfinite(A)):
        return -np.inf, None, None

    # transit model
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

    # Likelihood with jitter 
    var = yerr**2 + flux_jitter**2

    ll_dataset = -0.5 * np.sum(((y - model)**2) / var + np.log(2 * np.pi * var))

    return ll_dataset, transit, trend


# ============================================================
# Transit + trend log-likelihood
# ============================================================

def log_likelihood(theta, data, use_airmass=True, use_fwhm=False, use_centroid=False):
    ll = 0.0
    for dataset in data:
        ll_sec, _, _ = ground_photometry_log_likelihood_dataset(
            theta, dataset, ld_q1_key="q1", ld_q2_key="q2", use_airmass=use_airmass, use_fwhm=use_fwhm, use_centroid=use_centroid)
        ll += ll_sec
    return ll

# ============================================================
# Psterior = likelihood +prior
# ============================================================ 

def log_posterior(theta, data, use_airmass=True, use_fwhm=False, use_centroid=False):
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, data, use_airmass=use_airmass, use_fwhm=use_fwhm, use_centroid=use_centroid)