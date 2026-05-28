import numpy as np
import batman
import radvel


""" Models: transit and RV assuming an eccentric orbit"""

# -----------------------------
#  Quadratic limb darkening function (Kipping 2013)
# -----------------------------
def limb_darkening(q1, q2):

    sqrt_q1 = np.sqrt(q1)
    u1 = 2 * sqrt_q1 * q2
    u2 = sqrt_q1 * (1 - 2*q2)
    
    return u1, u2


def inverse_limb_darkening(u1,u2, err_u1, err_u2):

    q1 = (u1 + u2)**2
    q2 = 0.5*u1/(u1 + u2)
    
    err_q1 = 2 * (u1 + u2) * np.sqrt(err_u1**2 + err_u2**2)
    err_q2 = q2 * np.sqrt((err_u1/u1)**2 + (err_u1+err_u2)**2/(u1+u2)**2)

    return q1, q2, err_q1, err_q2


# -----------------------------
# BATMAN model function 
# -----------------------------
def transit_model(theta, t, supersample_factor=1, exp_time=0.0):

    t0, per, rp, b, h, k, mstar, rstar, q1, q2 = theta

    # parameterization of eccentricity and omega
    e = h**2 + k**2
    w = np.degrees(np.arctan2(k, h))  # batman expects degrees

    # period in years
    per_yr = per / 365.25

    # a/Rs (dimensionless)
    a = 215.032 * (mstar**(1/3)) * (per_yr**(2/3)) / rstar

    # limb darkening 
    u1, u2 = limb_darkening(q1, q2)

    # parameterization of impact parameter for inclination
    cosi = b / a
    cosi = np.clip(cosi, -1.0, 1.0)
    inc = np.degrees(np.arccos(cosi))

    params = batman.TransitParams()
    params.t0 = 0.0         #time of inferior conjunction (BJD)
    params.per = per        #orbital period
    params.rp = rp          #planet radius (in units of stellar radii Rp/Rs)
    params.a = a            #semi-major axis (in units of stellar radii a/Rs)
    params.inc = inc        #orbital inclination (in degrees)
    params.ecc = e          #eccentricity
    params.w = w            #longitude of periastron (in degrees)
    params.u = [u1,  u2]    #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic" #limb darkening model

    m = batman.TransitModel(params, t-t0, supersample_factor=supersample_factor, exp_time=exp_time) 
    return m.light_curve(params) 



# -----------------------------
# Eccentric planet model function 
# -----------------------------

def planetmodel(theta, t):
    t0, per, h, k, krv, gamma, *_ = theta 
    ecc = h**2 + k**2
    w_pl = np.arctan2(h, k) 

    # radvel utility uses the star's omega to find the star's periastron time
    tp = radvel.orbit.timetrans_to_timeperi(t0, per, ecc, w_pl)
    t_array = np.array(t)
    params = [per, tp, ecc, w_pl, krv]
    rv_model = radvel.kepler.rv_drive(t_array, params) + gamma
    
    return rv_model