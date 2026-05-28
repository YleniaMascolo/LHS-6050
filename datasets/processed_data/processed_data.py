
import numpy as np
from astropy.io import fits
import pandas as pd
import os
import pickle


"""Script to prepare, clean, and processed multi-facility data for LHS 6050.

This script ingests raw photometric datasets (space-based TESS and ground-based LCO, 
MINERVA, NGTS, TRAPPIST) and spectroscopic datasets (ESPRESSO RVs and activity indicators).
It applies custom instrumental corrections, filters bad quality flags, masks out-of-transit 
data for TESS, normalizes fluxes, and exports a unified, compressed 
pickle (.pkl) file containing all consolidated data structures ready for MCMC fitting.

Input: Separate raw data files per facility/filter.
Output: A compressed unified 'LHS6050_global_data.pkl' file.
"""

data_dir = "/home/ymascolo/Desktop/github/datasets" # path datsets

# ================================
# Load TESS sectors function
# ================================
def tess_sector(filename, t0, P, duration):

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
    
    # crowding correction
    crowdsap = hdr["CROWDSAP"]
    flux_corr = sap_flux / crowdsap

    nanmask = np.isnan(time) + np.isnan(flux_corr) 
    time = time[~nanmask]
    flux_corr = flux_corr[~nanmask]

    # normalization
    norm = np.nanmedian(flux_corr)
    flux = flux_corr / norm

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
    os.path.join(data_dir,"TESS/tess2020294194027-s0031-0000000337217173-0198-s_lc.fits"),
    os.path.join(data_dir, "TESS/tess2023263165758-s0070-0000000337217173-0265-s_lc.fits"),
    os.path.join(data_dir, "TESS/tess2023289093419-s0071-0000000337217173-0266-s_lc.fits")
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
    sigma = np.median(np.abs(np.diff(flux_mask)))
    flux_err = np.full_like(flux, 1.4826 * sigma) / median_flux 

    return time, flux, flux_err, airmass  


# ----------- LCO data ---------------

lco_files = [
    os.path.join(data_dir,"LCO/TIC_337217173-01_20250922_LCO-SS-1.0m_ip_15pix_measurements.tbl"),
    os.path.join(data_dir,"LCO/TIC337217173-11_20240924_LCO-Teid-1m0_ip_measurements.tbl"),
    os.path.join(data_dir,"LCO/TIC337217173-11_20241214_LCO-CTIO-1m0_ip_12px_Measurements.tbl"),
    os.path.join(data_dir,"LCO/TIC337217173-11_20241214_LCO-McD-1m0_ip_9px_Measurements.tbl")
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
    os.path.join(data_dir,"LCO/TIC_337217173-01_20251102_LCO-CTIO-1.0m_gp_12pix_measurements.tbl"),
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
    os.path.join(data_dir,"TRAPPIST/TIC_337217173-01_20251102_TRAPPIST-South-0.6m_Rc_12pix_measurements.tbl"),
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
    os.path.join(data_dir,"MINERVA/minerva_T2.dat"),
    os.path.join(data_dir,"MINERVA/minerva_T4.dat"),
    os.path.join(data_dir,"MINERVA/minerva_T5.dat")
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
    os.path.join(data_dir,"NGTS/NGTS_20250812_412358.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20250812_412361.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20250812_412365.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20250812_412369.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20250812_412372.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20250812_412379.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20251101_419475.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20251101_419478.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20251101_419481.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20251101_419484.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20251101_419487.dat"),
    os.path.join(data_dir,"NGTS/NGTS_20251101_419492.dat")
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
df_all = pd.read_csv(os.path.join(data_dir,"ESPRESSO/RV_data.txt"), sep='\s+')

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


    output_dir = os.path.join(data_dir, "processed_data")
    os.makedirs(output_dir, exist_ok=True)

    save_dict = {
        "tess_data": tess_data, "lco_data": lco_data, "lcog_data": lcog_data,
        "trap_data": trap_data, "minerva_data": minerva_data, "ngts_data": ngts_data,
        "espresso_data": espresso_data
    }

    output_file = os.path.join(output_dir, "LHS6050b_global_data.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
