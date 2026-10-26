
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
pickle (.pkl) file containing all consolidated data structures ready for fitting.

Input: Separate raw data files per facility/filter.
Output: A compressed unified 'LHS6050_global_data.pkl' file.
"""

data_dir = "/home/data" # path datsets

# ================================
# Mask transit function
# ================================
def mask_transits(time, t0, P, duration, factor=3):
   
    phase = (time - t0) % P
    window = factor * duration

    mask = (phase < window) | (phase > P - window)
    return mask

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

    # mask the transit considering a factor 3 of out of transit
    crop_mask = mask_transits(time, t0, P, duration)
    
    time_crop = time[crop_mask]
    flux_crop = flux_corr[crop_mask]

    # mask the in transit using a factor of 1.2 (to exclude ingress and egress)
    transit_width_factor = 0.6
    in_transit_mask = mask_transits(time_crop, t0, P, duration, factor=transit_width_factor)

    # out of transit data used to compute the sigma
    oot_mask = ~in_transit_mask
    time_oot = time_crop[oot_mask]
    flux_oot = flux_crop[oot_mask]
    norm = np.nanmedian(flux_oot)
    
    trend_fit = np.polyfit(time_oot, flux_oot, 2)
    trend = np.polyval(trend_fit, time_oot)
    sigma_oot = np.median(np.abs(np.diff(flux_oot/trend)))

    # flux and error normalised 
    flux_final = flux_crop / norm
    flux_err_final = np.full_like(flux_final,  sigma_oot)

    return time_crop, flux_final, flux_err_final


# ------------- TESS data ------------- 

sector_files = [
    os.path.join(data_dir,"TESS/tess2020294194027-s0031-0000000337217173-0198-s_lc.fits"),
    os.path.join(data_dir, "TESS/tess2023263165758-s0070-0000000337217173-0265-s_lc.fits"),
    os.path.join(data_dir, "TESS/tess2023289093419-s0071-0000000337217173-0266-s_lc.fits")
]

t0 = 2459164.3524 - 2457000  # mid-transit time (BTJD)
per  = 40.3834187              # period (days)
duration = 2.55/24          # transit duration (days)

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
def ground_dataset(filename, t0, P, duration):

    # load datasets
    df = pd.read_csv(filename, sep='\s+') 

    # Define data array 
    time = np.array(df['BJD_TDB']-2457000) # time in BJD - 2457000
    flux_raw = np.array(df['rel_flux_T1'])
    airmass = np.array(df['AIRMASS'])
    fwhm = np.array(df['Width_T1'])
    x_pos = np.array(df['X(IJ)_T1'])	
    y_pos = np.array(df['Y(IJ)_T1'])

    # mask nan data 
    nanmask = np.isnan(time) + np.isnan(flux_raw) + np.isnan(airmass)
    time = time[~nanmask]
    flux_mask = flux_raw[~nanmask]
    airmass = airmass[~nanmask]
    fwhm = fwhm[~nanmask]
    x_pos = x_pos[~nanmask]
    y_pos = y_pos[~nanmask]
    
    # error only on out of transit data
    in_transit_mask = mask_transits(time, t0, P, duration, factor=0.6)
    oot_mask = ~in_transit_mask

    if np.sum(oot_mask) < 5:
        print(f"[WARNING] less than 5 point in {filename}. Use entire dataset")
        flux_oot = flux_mask
        time_oot = time
    else:
        flux_oot = flux_mask[oot_mask]
        time_oot = time[oot_mask]

    median_flux_oot = np.nanmedian(flux_oot) 

    trend_fit = np.polyfit(time_oot, flux_oot, 2)
    trend = np.polyval(trend_fit, time_oot)
    sigma_oot = np.median(np.abs(np.diff(flux_oot/trend)))


    # normalised flux and sigma
    flux = flux_mask / median_flux_oot
    flux_err = np.full_like(flux,  sigma_oot)
    
    return time, flux, flux_err, airmass, fwhm, x_pos, y_pos  


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
    time_lco, flux_lco, flux_err_lco, airmass_lco, fwhm_lco, x_pos_lco, y_pos_lco = ground_dataset(f, t0, per, duration)

    lco_data.append({
    "dataset": labels[i-1],
    "time": time_lco,
    "flux": flux_lco,
    "err": flux_err_lco,
    "airmass": airmass_lco,
    "fwhm": fwhm_lco,
    "x_pos": x_pos_lco, 
    "y_pos": y_pos_lco
    })


# ------------- LCO data g filter ---------------

lcog_files = [
    os.path.join(data_dir,"LCO/TIC_337217173-01_20251102_LCO-CTIO-1.0m_gp_12pix_measurements.tbl"),
]

labels = ['CTIOg']

lcog_data = []
for  i, f in enumerate(lcog_files, start=1):
    time_lcog, flux_lcog, flux_err_lcog, airmass_lcog, fwhm_lcog, x_pos_lcog, y_pos_lcog = ground_dataset(f, t0, per, duration)

    lcog_data.append({
    "dataset": labels[i-1],
    "time": time_lcog,
    "flux": flux_lcog,
    "err": flux_err_lcog,
    "airmass": airmass_lcog,
    "fwhm": fwhm_lcog,
    "x_pos": x_pos_lcog, 
    "y_pos": y_pos_lcog
    })


# --------------- TRAPPIST data ----------------

trap_files = [
    os.path.join(data_dir,"TRAPPIST/TIC_337217173-01_20251102_TRAPPIST-South-0.6m_Rc_12pix_measurements.tbl"),
]

labels = ['TRAPPIST']

trap_data = []
for  i, f in enumerate(trap_files, start=1):
    time_trap, flux_trap, flux_err_trap, airmass_trap, fwhm_trap, x_pos_trap, y_pos_trap = ground_dataset(f, t0, per, duration)

    trap_data.append({
    "dataset": labels[i-1],
    "time": time_trap,
    "flux": flux_trap,
    "err": flux_err_trap,
    "airmass": airmass_trap,
    "fwhm": fwhm_trap,
    "x_pos": x_pos_trap, 
    "y_pos": y_pos_trap
    })



# ================================
# MINERVA: Load data function
# ================================
def minerva_dataset(filename, t0, P, duration):

    data = np.genfromtxt(filename)  
        
    time = data[:,0] - 2457000 # time in BJD - 2457000
    flux_raw = data[:,1]
    #flux_err_raw = data[:,2]
    airmass = data[:,3]
    fwhm = data[:,5]
    x_pos = data[:,6]
    y_pos = data[:,7]

    # mask nan data 
    nanmask = np.isnan(time) + np.isnan(flux_raw) + np.isnan(airmass)  + np.isnan(fwhm) + np.isnan(x_pos) + np.isnan(y_pos)
    time = time[~nanmask]
    flux_mask = flux_raw[~nanmask]
    airmass = airmass[~nanmask]
    fwhm = fwhm[~nanmask]
    x_pos = x_pos[~nanmask]
    y_pos = y_pos[~nanmask]

    # remove data affected by spot
    clean_mask = (time < 3941.205) | (time > 3941.250)
    time = time[clean_mask]
    flux_mask = flux_mask[clean_mask]
    airmass = airmass[clean_mask]
    fwhm = fwhm[clean_mask]
    x_pos = x_pos[clean_mask]
    y_pos = y_pos[clean_mask]

    
    # error only on out of transit data
    in_transit_mask = mask_transits(time, t0, P, duration, factor=0.6)
    oot_mask = ~in_transit_mask


    if np.sum(oot_mask) < 5:
        print(f"[WARNING] less than 5 point in {filename}. Use entire dataset")
        flux_oot = flux_mask
        time_oot = time
    else:
        flux_oot = flux_mask[oot_mask]
        time_oot = time[oot_mask]

    median_flux_oot = np.nanmedian(flux_oot) 

    trend_fit = np.polyfit(time_oot, flux_oot, 2)
    trend = np.polyval(trend_fit, time_oot)
    sigma_oot = np.median(np.abs(np.diff(flux_oot/trend)))


    # normalised flux and sigma
    flux = flux_mask / median_flux_oot
    flux_err = np.full_like(flux,  sigma_oot)

    return time, flux, flux_err, airmass, fwhm, x_pos, y_pos

# ---------------- Minerva data ------------------

files = [
    os.path.join(data_dir,"MINERVA/minerva_T2.dat"),
    os.path.join(data_dir,"MINERVA/minerva_T4.dat"),
    os.path.join(data_dir,"MINERVA/minerva_T5.dat")
]

labels = ['T2','T4','T5']

minerva_data = []
for  i, f in enumerate(files, start=1):
    time_min, flux_min, err_min, airmass_min, fwhm_min, x_pos_min, y_pos_min = minerva_dataset(f, t0, per, duration)

    minerva_data.append({
        "dataset": labels[i-1],
        "time": time_min,
        "flux": flux_min,
        "err": err_min,
        "airmass": airmass_min,
        "fwhm": fwhm_min,
        "x_pos": x_pos_min, 
        "y_pos": y_pos_min
    })


# ================================
# NGTS: Load data function
# ================================
def ground2_dataset(filename, t0, P, duration):
    
    data = np.genfromtxt(filename)  
        
    time = data[:,0] - 2457000 # time in BJD - 2457000
    flux_raw = data[:,1]
    airmass = data[:,3]
    fwhm = data[:,4]
    bg = data[:,5]

    # mask nan data 
    nanmask = np.isnan(time) + np.isnan(flux_raw) + np.isnan(airmass) + np.isnan(fwhm) + np.isnan(bg)
    time = time[~nanmask]
    flux_mask = flux_raw[~nanmask]
    airmass = airmass[~nanmask]
    fwhm = fwhm[~nanmask]
    bg = bg[~nanmask]
    
    # error only on out of transit data
    in_transit_mask = mask_transits(time, t0, P, duration, factor=0.6)
    oot_mask = ~in_transit_mask


    if np.sum(oot_mask) < 5:
        print(f"[WARNING] less than 5 point in {filename}. Use entire dataset")
        flux_oot = flux_mask
        time_oot = time
    else:
        flux_oot = flux_mask[oot_mask]
        time_oot = time[oot_mask]

    median_flux_oot = np.nanmedian(flux_oot) 

    trend_fit = np.polyfit(time_oot, flux_oot, 2)
    trend = np.polyval(trend_fit, time_oot)
    sigma_oot = np.median(np.abs(np.diff(flux_oot/trend)))


    # normalised flux and sigma
    flux = flux_mask / median_flux_oot
    flux_err = np.full_like(flux,  sigma_oot)


    return time, flux, flux_err, airmass, fwhm, bg




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
    time_ngts, flux_ngts, err_ngts, airmass_ngts, fwhm_ngts, bg_ngts = ground2_dataset(f, t0, per, duration)

    ngts_data.append({
        "dataset": labels[i-1],
        "time": time_ngts,
        "flux": flux_ngts,
        "err": err_ngts,
        "airmass": airmass_ngts, 
        "fwhm": fwhm_ngts,
        "bg": bg_ngts
    })



# =================================
# ESPRESSO dataset 
# =================================
df = pd.read_csv(os.path.join(data_dir,"ESPRESSO/RV_data.txt"), sep='\s+')

# df = df_all.dropna(subset=['rv_bart', 'rv_bart_err', 'rv_lbl', 'rv_lbl_err', 'dlw', 'dlw_err', 'fwhm_lbl', 'fwhm_lbl_err'])

t = df['bjd'] - 2457000

rv = df['rv_bart'] - np.mean(df['rv_bart'])
rv_err = df['rv_bart_err']

rv_gz = df['rv_gz'] - np.mean(df['rv_gz'])
rv_gz_err = df['rv_gz_err']

rv_esp = df['rv_esp'] - np.mean(df['rv_esp'])
rv_esp_err = df['rv_esp_err']

rv_lbl = df['rv_lbl'] - np.mean(df['rv_lbl'])
rv_lbl_err = df['rv_lbl_err']

fwhm = df['fwhm_esp'] - np.mean(df['fwhm_esp'])
fwhm_err = df['fwhm_esp_err']

fwhm_gz = df['fwhm_gz'] - np.mean(df['fwhm_gz'])
fwhm_gz_err = df['fwhm_gz_err']

fwhm_lbl = df['fwhm_lbl'] - np.mean(df['fwhm_lbl'])
fwhm_lbl_err = df['fwhm_lbl_err']

bis = df['bis'] - np.mean(df['bis'])
bis_err = df['bis_err']

asym = df['asym'] - np.mean(df['asym'])
asym_err = df['asym_err']

ha = df['ha'] - np.mean(df['ha'])
ha_err = df['ha_err']

na = df['na'] - np.mean(df['na'])
na_err = df['na_err']

ca = df['ca'] - np.mean(df['ca'])
ca_err = df['ca_err']

s = df['s'] - np.mean(df['s'])
s_err = df['s_err']

contrast = df['contrast'] - np.mean(df['contrast'])
contrast_err = df['contrast_err']

dlw = df['dlw'] - np.mean(df['dlw'])
dlw_err = df['dlw_err']


espresso_data = ({
        "time": t,
        "rv": rv,
        "rv_err": rv_err,
        "rv_gz": rv_gz,
        "rv_gz_err": rv_gz_err,
        "rv_esp": rv_esp,
        "rv_esp_err": rv_esp_err,
        "rv_lbl": rv_lbl,
        "rv_lbl_err": rv_lbl_err,
        "fwhm": fwhm,
        "fwhm_err": fwhm_err,
        "fwhm_gz": fwhm_gz,
        "fwhm_gz_err": fwhm_gz_err,
        "fwhm_lbl": fwhm_lbl,
        "fwhm_lbl_err": fwhm_lbl_err,
        "bis": bis, 
        "bis_err": bis_err,
        "asym": asym, 
        "asym_err": asym_err,
        "ha": ha, 
        "ha_err": ha_err,
        "ca": ca, 
        "ca_err": ca_err,
        "na": na, 
        "na_err": na_err,
        "s": s, 
        "s_err": s_err,
        "contrast": contrast, 
        "contrast_err": contrast_err,
        "dlw": dlw, 
        "dlw_err": dlw_err
    })


if __name__ == "__main__":

    save_dict = {
        "tess_data": tess_data, "lco_data": lco_data, "lcog_data": lcog_data,
        "trap_data": trap_data, "minerva_data": minerva_data, "ngts_data": ngts_data,
        "espresso_data": espresso_data
    }

    output_file = os.path.join(data_dir, "LHS6050b_complete_dataset.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
