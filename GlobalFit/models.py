import numpy as np
import batman
import radvel


# Models: transit and RV

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

    t0, per, rp, b, mstar, rstar, q1, q2 = theta

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
    params.ecc = 0         #eccentricity
    params.w = 90          #longitude of periastron (in degrees)
    params.u = [u1,  u2]    #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic" #limb darkening model

    m = batman.TransitModel(params, t-t0, supersample_factor=supersample_factor, exp_time=exp_time)  
    return m.light_curve(params) 


# -----------------------------
# Sinusoidal planet model function 
# -----------------------------
def planetmodel(theta,t):
    
    t0, per, krv, gamma, *_= theta 
    
    phase = (t-t0)/per
    rvmodel = -1 * krv * np.sin(2*np.pi * phase) + gamma

    return rvmodel