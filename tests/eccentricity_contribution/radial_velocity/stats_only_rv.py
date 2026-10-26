
import numpy as np
from numpy.linalg import lstsq
import multiGP
import radvel
from scipy.special import ndtri

PARAM_NAMES = [
    "h", "k", 
    "lam", "prot", "w",
    "krv", "rv_scale", "rv_dGdt_scale", "rv_gamma", "rv_jitter",
    "fwhm_scale", "fwhm_gamma", "fwhm_jitter"
]

prot_transit = 6.2326
prot_transit_err = 0.1314
t0 = 2164.35447598362  # mid-transit time (BTJD)
per  = 40.3834              # period (days)


def planetmodel(theta, t0, per, t):
    
    ecc = theta["h"]**2 + theta["k"]**2
    w_pl = np.arctan2(theta["h"], theta["k"]) 

    tp = radvel.orbit.timetrans_to_timeperi(t0, per, ecc, w_pl)
    t_array = np.array(t)
    params = [per, tp, ecc, w_pl, theta["krv"]]
    rv_model = radvel.kepler.rv_drive(t_array, params) + theta["rv_gamma"]
    
    return rv_model


# ============================================================
# Prior Transform for Dynesty
# ============================================================

def prior_transform(u):

    p = np.empty_like(u)

    e_max = 0.9
    e = u[0] * e_max                 # e ~ Unif(0, e_max)
    omega_rad = u[1] * (2 * np.pi)   # omega ~ Unif(0, 2pi)
    r = np.sqrt(e)                   # e = h**2 + k**2 = r**2

    p[0] = r * np.sin(omega_rad)     # h
    p[1] = r * np.cos(omega_rad)     # k

    p[2] = 6.5 + u[2] * (600.0 - 6.5)  # lam 
    p[3] = prot_transit + prot_transit_err * ndtri(u[3]) # prot gaussian
    p[4] = 0.01 + u[4] * (6.5 - 0.01)  # w

    p[5] = u[5] * 0.1  # krv (0 < krv < 0.1)
    p[6] = -0.05 + u[6] * (0.05 - (-0.05))  # rv_scale
    p[7] = -0.1 + u[7] * (0.1 - (-0.1))  # rv_dGdt_scale
    p[8] = -0.05 + u[8] * (0.05 - (-0.05))  # rv_gamma
    p[9] = u[9] * 0.02  # rv_jitter

    # --- Indicators ---
    p[10] = u[10] * 0.1  # fwhm_scale
    p[11] = -0.1 + u[11] * (0.1 - (-0.1))  # fwhm_gamma
    p[12] = u[12] * 0.02  # fwhm_jitter

    return p

# -----------Normalization function -----------------
def normalization(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# ============================================================
# ESPRESSO: Multidimension GP log-likelihood
# ============================================================

def log_likelihood(theta_vec, espresso_data, indicators=["rv", "fwhm"]):
    theta = {name: val for name, val in zip(PARAM_NAMES, theta_vec)}

    if (theta["h"]**2 + theta["k"]**2) >= 0.9:
        return -np.inf

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

def compute_mgic(theta_vec, espresso_data, indicators=["rv", "fwhm"]):
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




