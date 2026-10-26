
import numpy as np
from numpy.linalg import lstsq
import multiGP
from scipy.special import ndtri

PARAM_NAMES = [
    "lam", "prot", "w",
    "krv", "rv_scale", "rv_dGdt_scale", "rv_gamma", "rv_jitter",
    "fwhm_scale", "fwhm_gamma", "fwhm_jitter",
    
    ## add or remove indicators
    # "asym_scale", "asym_dGdt_scale", "asym_gamma", "asym_jitter",
    # "bis_scale", "bis_dGdt_scale", "bis_gamma", "bis_jitter"
    # "dlw_scale", "dlw_dGdt_scale", "dlw_gamma", "dlw_jitter"
    # "ca_scale", "ca_dGdt_scale", "ca_gamma", "ca_jitter"
    # "ha_scale", "ha_dGdt_scale", "ha_gamma", "ha_jitter"
    # "na_scale", "na_dGdt_scale", "na_gamma", "na_jitter"
    # "s_scale", "s_dGdt_scale", "s_gamma", "s_jitter"
    # "contrast_scale", "contrast_dGdt_scale", "contrast_gamma", "contrast_jitter"
]

prot_transit = 6.2326
prot_transit_err = 0.1314
t0 = 2164.35447598362
per  = 40.3834              # period (days)

# -----------------------------
# Sinusoidal planet model function 
# -----------------------------
def planetmodel(theta, t0, per, t):

    # Calculate the orbital phase relative to the reference transit time
    phase = (t - t0) / per
    
    rvmodel = -1 * theta["krv"] * np.sin(2 * np.pi * phase) + theta["rv_gamma"]

    return rvmodel



# ============================================================
# Prior Transform for Dynesty
# ============================================================

def prior_transform(u):

    p = np.empty_like(u)

    p[0] = 6.5 + u[0] * (600.0 - 6.5)  # lam 
    p[1] = prot_transit + prot_transit_err * ndtri(u[1]) # prot gaussian
    p[2] = 0.01 + u[2] * (6.5 - 0.01)  # w

    p[3] = u[3] * 0.1  # krv (0 < krv < 0.1)
    p[4] = -0.05 + u[4] * (0.05 - (-0.05))  # rv_scale
    p[5] = -0.1 + u[5] * (0.1 - (-0.1))  # rv_dGdt_scale
    p[6] = -0.05 + u[6] * (0.05 - (-0.05))  # rv_gamma
    p[7] = u[7] * 0.02  # rv_jitter

    # --- Indicators ---
    p[8] = u[8] * 0.1  # fwhm_scale
    p[9] = -0.1 + u[9] * (0.1 - (-0.1))  # fwhm_gamma
    p[10] = u[10] * 0.02  # fwhm_jitter

    # p[11] = -0.01 + u[11] * (0.01 - (-0.01))  # asym_scale
    # p[12] = -0.01 + u[12] * (0.01 - (-0.01))  # asym_dGdt_scale
    # p[13] = -0.01 + u[13] * (0.01 - (-0.01))  # asym_gamma
    # p[14] = u[14] * 0.002  # asym_jitter

    # p[15] = -0.03 + u[15] * (0.03 - (-0.03))  # bis_scale
    # p[16] = -0.03 + u[16] * (0.03 - (-0.03))  # bis_dGdt_scale
    # p[17] = -0.03 + u[17] * (0.03 - (-0.03))  # bis_gamma
    # p[18] = u[18] * 0.007  # bis_jitter

    # p[11] = -0.0002 + u[11] * (0.0002 - (-0.0002))  # dlw_scale
    # p[12] = -0.0002 + u[12] * (0.0002 - (-0.0002))  # dlw_dGdt_scale
    # p[13] = -0.0002 + u[13] * (0.0002 - (-0.0002))  # dlw_gamma
    # p[14] = u[14] * 0.0001  # dlw_jitter

    # p[11] = -18 + u[11] * (18 - (-18))  # ca_scale
    # p[12] = -18 + u[12] * (18 - (-18))  # ca_dGdt_scale
    # p[13] = -18 + u[13] * (18 - (-18))  # ca_gamma
    # p[14] = u[14] * 2.5  # ca_jitter

    # p[11] = -5 + u[11] * (5 - (-5))    # ha_scale
    # p[12] = -5 + u[12] * (5 - (-5))   # ha_dGdt_scale
    # p[13] = -5 + u[13] * (5 - (-5))   # ha_gamma
    # p[14] = u[14] * 3  # ha_jitter

    # p[11] = 0.5 + u[11] * (0.5 - (-0.5))  # na_scale
    # p[12] = -0.5 + u[12] * (0.5 - (-0.5))  # na_dGdt_scale
    # p[13] = -0.5 + u[13] * (0.5 - (-0.5))  # na_gamma
    # p[14] = u[14] * 0.1  # na_jitter

    # p[11] = -18500 + u[11] * (18500 - (-18500))  # s_scale
    # p[12] = -18500 + u[12] * (18500 - (-18500))  # s_dGdt_scale
    # p[13] = -18500 + u[13] * (18500 - (-18500))  # s_gamma
    # p[14] = u[14] * 3000 # s_jitter

    # p[11] = -3 + u[11] * (3 - (-3))  # contrast_scale
    # p[12] = -3 + u[12] * (3 - (-3))  # contrast_dGdt_scale
    # p[13] = -3 + u[13] * (3 - (-3))  # contrast_gamma
    # p[14] = u[14] * 0.5  # contrast_jitter

    return p

# -----------Normalization function -----------------
def normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# ============================================================
# ESPRESSO: Multidimension GP log-likelihood
# ============================================================

def log_likelihood(theta_vec, espresso_data, indicators=["rv", "fwhm", "asym", "bis"]): # add or remove indicators 
    theta = {name: val for name, val in zip(PARAM_NAMES, theta_vec)}

    t = espresso_data["time"]
    
    rvmodel = planetmodel(theta, t0, per, t)

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
        jitter_list.append(theta[f"{ind}_jitter"])
        
        # B scale (es. dG/dt)
        B_list.append(theta.get(f"{ind}_dGdt_scale", 0.0))

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


# ============================================================
# MGIC computation
# ============================================================

def compute_mgic(theta_vec, espresso_data, indicators=["rv", "fwhm", "asym", "bis"]): # add or remove indicators 
    theta = {name: val for name, val in zip(PARAM_NAMES, theta_vec)}
    t = espresso_data["time"]
    rvmodel = planetmodel(theta, t0, per, t)

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
        k_pars=len(PARAM_NAMES)
    )




