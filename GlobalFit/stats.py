
import numpy as np
from numpy.linalg import lstsq
import multiGP
import models
from ldtk import LDPSetCreator, BoxcarFilter
from ldtk.filters import tess, sdss_i, sdss_g




# stellar parameters 
mstar_obs = 0.224
rstar_obs = 0.254
mstar_obs_err = 0.007
rstar_obs_err = 0.013
prot_transit = 6.20
prot_transit_err = 0.01


# ============================================================
# Log-prior 
# ============================================================
def log_prior(theta):

    t0, per, rp, b, mstar, rstar, q1_tess, q2_tess, q1_lco, q2_lco, q1_lcog, q2_lcog, q1_minerva, q2_minerva, q1_ngts, q2_ngts, q1_trap, q2_trap,\
    krv, gamma, lam, prot, w, RV_scale, fwhm_scale, bis_scale, asym_scale, \
    RV_dGdt_scale, FWHM_dGdt_scale, asym_dGdt_scale, rv_jitter, fwhm_jitter, bis_jitter, asym_jitter = theta

    if not (
        2164.32  < t0 < 2164.38 and
        40.3825 < per < 40.384 and
        rp > 0.0 and
        0. < b < 1.0 and
        mstar > 0.0 and
        rstar > 0.0 and
        0.0 < q1_tess < 1.0 and 0.0 < q2_tess < 1.0 and
        0.0 < q1_lco < 1.0 and 0.0 < q2_lco < 1.0 and 
        0.0 < q1_lcog < 1.0 and 0.0 < q2_lcog < 1.0 and
        0.0 < q1_minerva < 1.0 and 0.0 < q2_minerva < 1.0 and 
        0.0 < q1_ngts < 1.0 and 0.0 < q2_ngts < 1.0 and
        0.0 < q1_trap < 1.0 and 0.0 < q2_trap < 1.0 and
        -0.1 < krv < 0.1 and         ### rv amp
        0 < lam < 600 and           ### lam  (radial timescale factor)
        6 < prot < 6.5 and           ### prot
        0 < w < 100 and              ### w (periodic timescale factor)
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
# normalization function 
# ============================================================
def normalization(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))

# ============================================================
# TESS: Transit + trend log-likelihood
# ============================================================
def tess_transit_log_likelihood_sector(theta, tess_data):

    t0, per, rp, b, mstar, rstar,  q1_tess, q2_tess, *_ = theta

    theta_tess = t0, per, rp, b, mstar, rstar, q1_tess, q2_tess

    t = tess_data["time"]
    y = tess_data["flux"]
    yerr = tess_data["err"]

    # normalization function for time
    t_norm = normalization(t)

    # A matrix for polinomial of 2 order
    A = np.column_stack([ t_norm**2, t_norm, np.ones_like(t_norm)])

    # transit model
    transit = models.transit_model(theta_tess, t, supersample_factor=5, exp_time=0.0013888889)
    if np.std(transit) < 1e-5:
        #transit *= np.nan
        return -np.inf, None, None
    residuals = y - transit

    # fit trend using lstsq
    coeffs_fit, _, _, _ = lstsq(A, residuals, rcond=None)

    # trend model
    trend = A @ coeffs_fit # @ useful for multiply matrix
    model = transit + trend

    if not np.all(np.isfinite(model)):
        return -np.inf, None

    # log-likelihood
    ll_transit_sector = -0.5 * np.sum(((y - model) / yerr) ** 2)

    return ll_transit_sector, transit, trend

# logL for all transits
def tess_transit_log_likelihood(theta, tess_data):
    tess_ll_transit = 0.0
    for sector in tess_data:
        tess_ll_sec, _, _ = tess_transit_log_likelihood_sector(theta, sector)
        tess_ll_transit += tess_ll_sec
    return tess_ll_transit


# ============================================================
# LCO: Transit + trend log-likelihood
# ============================================================
def lco_transit_log_likelihood_dataset(theta, lco_data):

    t0, per, rp, b, mstar, rstar,  _, _, q1_lco, q2_lco, *_ = theta

    theta_lco = t0, per, rp, b, mstar, rstar,  q1_lco, q2_lco

    t = lco_data["time"]
    y = lco_data["flux"]
    yerr = lco_data["err"]
    airmass = lco_data["airmass"]

    t_norm = normalization(t)
    airmass_norm = normalization(airmass)

    # A matrix for polinomial of 2 order
    A = np.column_stack([ t_norm**2, t_norm, np.ones_like(t_norm), airmass_norm]) 

    # transit model
    transit = models.transit_model(theta_lco, t)
    if np.std(transit) < 1e-5:
        #transit *= np.nan
        return -np.inf, None, None
    residuals = y - transit

    # fit trend using lstsq
    coeffs_fit, _, _, _ = lstsq(A, residuals, rcond=None)

    # trend model
    trend = A @ coeffs_fit # @ useful for multiply matrix
    model = transit + trend

    if not np.all(np.isfinite(model)):
        return -np.inf, None

    # log-likelihood
    lco_ll_transit = -0.5 * np.sum(((y - model) / yerr) ** 2)

    return lco_ll_transit, transit, trend

# logL for all transits
def lco_transit_log_likelihood(theta, lco_data):
    lco_ll_transit = 0.0
    for dataset in lco_data:
        lco_ll_sec, _, _ = lco_transit_log_likelihood_dataset(theta, dataset)
        lco_ll_transit += lco_ll_sec
    return lco_ll_transit

# ============================================================
# LCO filter g: Transit + trend log-likelihood
# ============================================================
def lcog_transit_log_likelihood_dataset(theta, lcog_data):

    t0, per, rp, b, mstar, rstar,  _, _, _, _, q1_lcog, q2_lcog, *_ = theta

    theta_lcog = t0, per, rp, b, mstar, rstar,  q1_lcog, q2_lcog

    t = lcog_data["time"]
    y = lcog_data["flux"]
    yerr = lcog_data["err"]
    airmass = lcog_data["airmass"]

    t_norm = normalization(t)
    airmass_norm = normalization(airmass)

    # A matrix for polinomial of 2 order
    A = np.column_stack([ t_norm**2, t_norm, np.ones_like(t_norm), airmass_norm]) 

    # transit model
    transit = models.transit_model(theta_lcog, t)
    if np.std(transit) < 1e-5:
        #transit *= np.nan
        return -np.inf, None, None
    residuals = y - transit

    # fit trend using lstsq
    coeffs_fit, _, _, _ = lstsq(A, residuals, rcond=None)

    # trend model
    trend = A @ coeffs_fit # @ useful for multiply matrix
    model = transit + trend

    if not np.all(np.isfinite(model)):
        return -np.inf, None

    # log-likelihood
    lcog_ll_transit = -0.5 * np.sum(((y - model) / yerr) ** 2)

    return lcog_ll_transit, transit, trend

# logL for all transits
def lcog_transit_log_likelihood(theta, lcog_data):
    lcog_ll_transit = 0.0
    for dataset in lcog_data:
        lcog_ll_sec, _, _ = lcog_transit_log_likelihood_dataset(theta, dataset)
        lcog_ll_transit += lcog_ll_sec
    return lcog_ll_transit

# ============================================================
# Minerva: Transit + trend log-likelihood
# ============================================================
def minerva_transit_log_likelihood_dataset(theta, minerva_data):

    t0, per, rp, b, mstar, rstar,   _, _, _, _, q1_minerva, q2_minerva, *_ = theta

    theta_minerva = t0, per, rp, b, mstar, rstar,  q1_minerva, q2_minerva

    t = minerva_data["time"]
    y = minerva_data["flux"]
    yerr = minerva_data["err"]
    airmass = minerva_data["airmass"]

    t_norm = normalization(t)
    airmass_norm = normalization(airmass)

    # A matrix for polinomial of 2 order
    A = np.column_stack([ t_norm**2, t_norm, np.ones_like(t_norm), airmass_norm])

    # transit model
    transit = models.transit_model(theta_minerva, t)
    if np.std(transit) < 1e-5:
        #transit *= np.nan
        return -np.inf, None, None
    residuals = y - transit

    # fit trend using lstsq
    coeffs_fit, _, _, _ = lstsq(A, residuals, rcond=None)

    # trend model
    trend = A @ coeffs_fit # @ useful for multiply matrix
    model = transit + trend

    if not np.all(np.isfinite(model)):
        return -np.inf, None

    # log-likelihood
    minerva_ll_transit = -0.5 * np.sum(((y - model) / yerr) ** 2)

    return minerva_ll_transit, transit, trend

# logL for all transits
def minerva_transit_log_likelihood(theta, minerva_data):
    minerva_ll_transit = 0.0
    for dataset in minerva_data:
        minerva_ll_sec, _, _ = minerva_transit_log_likelihood_dataset(theta, dataset)
        minerva_ll_transit += minerva_ll_sec
    return minerva_ll_transit


# ============================================================
# NGTS: Transit + trend log-likelihood
# ============================================================
def ngts_transit_log_likelihood_dataset(theta, ngts_data):

    t0, per, rp, b, mstar, rstar,   _, _, _, _, _, _, q1_ngts, q2_ngts, *_ = theta

    theta_ngts = t0, per, rp, b, mstar, rstar,  q1_ngts, q2_ngts

    t = ngts_data["time"]
    y = ngts_data["flux"]
    yerr = ngts_data["err"]
    airmass = ngts_data["airmass"]

    t_norm = normalization(t)
    airmass_norm = normalization(airmass)

    # A matrix for polinomial of 2 order
    A = np.column_stack([ t_norm**2, t_norm, np.ones_like(t_norm), airmass_norm]) 

    if not np.all(np.isfinite(A)):
        return -np.inf

    # transit model
    transit = models.transit_model(theta_ngts, t)
    if np.std(transit) < 1e-5:
        #transit *= np.nan
        return -np.inf, None, None
    residuals = y - transit

    if not np.all(np.isfinite(residuals)):
        return -np.inf

    # fit trend using lstsq
    coeffs_fit, _, _, _ = lstsq(A, residuals, rcond=None)

    # trend model
    trend = A @ coeffs_fit # @ useful for multiply matrix
    model = transit + trend

    if not np.all(np.isfinite(model)):
        return -np.inf, None

    # log-likelihood
    ll_transit_dataset = -0.5 * np.sum(((y - model) / yerr) ** 2)

    return ll_transit_dataset, transit, trend

# logL for all transits
def ngts_transit_log_likelihood(theta, ngts_data):
    ll_transit = 0.0
    for dataset in ngts_data:
        ll_sec, _, _ = ngts_transit_log_likelihood_dataset(theta, dataset)
        ll_transit += ll_sec
    return ll_transit

# ============================================================
# TRAPPIST: Transit + trend log-likelihood
# ============================================================
def trap_transit_log_likelihood_dataset(theta, trap_data):

    t0, per, rp, b, mstar, rstar, _, _, _, _, _, _, _, _, _, _, q1_trap, q2_trap, *_ = theta

    theta_trap = t0, per, rp, b, mstar, rstar, q1_trap, q2_trap

    t = trap_data["time"]
    y = trap_data["flux"]
    yerr = trap_data["err"]

    t_norm = normalization(t)

    # A matrix for polinomial of 2 order
    A = np.column_stack([ t_norm**2, t_norm, np.ones_like(t_norm)]) 

    if not np.all(np.isfinite(A)):
        return -np.inf

    # transit model
    transit = models.transit_model(theta_trap, t)
    if np.std(transit) < 1e-5:
        #transit *= np.nan
        return -np.inf, None, None
    residuals = y - transit

    if not np.all(np.isfinite(residuals)):
        return -np.inf

    # fit trend using lstsq
    coeffs_fit, _, _, _ = lstsq(A, residuals, rcond=None)

    # trend model
    trend = A @ coeffs_fit # @ useful for multiply matrix
    model = transit + trend

    if not np.all(np.isfinite(model)):
        return -np.inf, None

    # log-likelihood
    ll_trap_transit_dataset = -0.5 * np.sum(((y - model) / yerr) ** 2)

    return ll_trap_transit_dataset, transit, trend

# logL for all transits
def trap_transit_log_likelihood(theta, trap_data):
    ll_transit = 0.0
    for dataset in trap_data:
        ll_sec, _, _ = trap_transit_log_likelihood_dataset(theta, dataset)
        ll_transit += ll_sec
    return ll_transit


# ============================================================
# ESPRESSO: Multidimension GP log-likelihood
# ============================================================
def espresso_rv_log_likelihood(theta,espresso_data):

    t0, per, _,  _,  _, _, _, _, _, _, _, _, _, _, _, _, _, _,  \
    krv, gamma, lam, prot, w, RV_scale, fwhm_scale, bis_scale, asym_scale, \
    RV_dGdt_scale, FWHM_dGdt_scale, asym_dGdt_scale, rv_jitter, fwhm_jitter, bis_jitter, asym_jitter = theta

    theta_espresso = t0, per, krv, gamma, lam, prot, w, RV_scale, fwhm_scale, bis_scale, asym_scale, \
    RV_dGdt_scale, FWHM_dGdt_scale, asym_dGdt_scale, rv_jitter, fwhm_jitter, bis_jitter, asym_jitter

    t = espresso_data["time"]
    rv = espresso_data["rv"]
    rv_err = espresso_data["rv_err"]
    fwhm = espresso_data["fwhm"]
    fwhm_err = espresso_data["fwhm_err"]
    bis = espresso_data["bis"]
    bis_err = espresso_data["bis_err"]
    asym = espresso_data["asym"]
    asym_err = espresso_data["asym_err"]
    
    rvmodel = models.planetmodel(theta_espresso,t)
    hp = multiGP.QPHyperParams(amp=1, lam=lam, P=prot, w=w)  # amp=1 because absorbed in A and B coefficients
    A = np.array([RV_scale, fwhm_scale, bis_scale, asym_scale])      # e.g. RV and BIS scale with G
    B = np.array([RV_dGdt_scale, FWHM_dGdt_scale, 0.0, asym_dGdt_scale])      # e.g. RV also scales with dG/dt, BIS not

    jitter = np.array([rv_jitter, fwhm_jitter, bis_jitter, asym_jitter]) # per observable

    ll_rv = multiGP.lnlike_multidim_qp_gp(
        t_list=[t,t,t, t],
        r_list=[rv-rvmodel,fwhm, bis, asym],
        err_list=[rv_err,fwhm_err, bis_err, asym_err],
        A=A, B=B, hp=hp, jitter=jitter
    )  

    if ll_rv != ll_rv:
        ll_rv = -1*np.inf
        
    return ll_rv

# ============================================================
# Stellar log-likelihood
# ============================================================
def star_log_likelihood(theta):
    _, _, _, _, mstar, rstar, _, _, _, _, _, _, _, _, _, _, _, _, _, _, _, prot, *_ = theta

    ll_m = -0.5 * ((mstar - mstar_obs) / mstar_obs_err) ** 2
    ll_r = -0.5 * ((rstar - rstar_obs) / rstar_obs_err) ** 2
    ll_prot = -0.5 * ((prot - prot_transit) / prot_transit_err) ** 2

    ll_star = ll_m + ll_r + ll_prot

    return ll_star



# ============================================================
# Total log-likelihood = transits + stellar
# ============================================================
def log_likelihood(theta, tess_data, lco_data, lcog_data, minerva_data, ngts_data, trap_data, espresso_data):

    tess_ll_transit = tess_transit_log_likelihood(theta, tess_data)
    lco_ll_transit = lco_transit_log_likelihood(theta, lco_data)
    lcog_ll_transit = lcog_transit_log_likelihood(theta, lcog_data)
    minerva_ll_transit = minerva_transit_log_likelihood(theta, minerva_data)
    ngts_ll_transit = ngts_transit_log_likelihood(theta, ngts_data)
    trap_ll_transit = trap_transit_log_likelihood(theta, trap_data)
    ll_rv = espresso_rv_log_likelihood(theta,espresso_data)
    ll_star = star_log_likelihood(theta)

    return tess_ll_transit + lco_ll_transit + lcog_ll_transit + minerva_ll_transit + ngts_ll_transit + trap_ll_transit + ll_rv + ll_star


# ============================================================
# Log-posterior 
# ============================================================
def log_probability(theta, tess_data, lco_data, lcog_data,  minerva_data, ngts_data, trap_data, espresso_data):
    
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf

    ll = log_likelihood(theta, tess_data, lco_data, lcog_data, minerva_data, ngts_data, trap_data, espresso_data)
    if not np.isfinite(ll):
        return -np.inf

    return lp + ll