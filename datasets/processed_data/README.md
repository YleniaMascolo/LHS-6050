## Pickle File Structure (`LHS6050b_complete_dataset.pkl`)

The compiled data file is saved as a Python dictionary containing preprocessed photometric light curves and spectroscopic time series.

> **Note on Normalization & Epochs:**
> * All timestamps (`time`) across all instruments are shifted to **$\text{BJD} - 2457000$**.
> * Photometric fluxes are normalized by dividing by the out-of-transit median flux ($F / F_{\text{OOT}}$).
> * Spectroscopic RVs and activity indicators in `espresso_data` are **mean-subtracted** ($X - \bar{X}$).

---

### Key-Value Map

| Top-Level Key | Type | Description | Internal Structure / Dictionary Keys |
| :--- | :--- | :--- | :--- |
| **`tess_data`** | `list` of `dict` | Space-based photometer light curves split by sector (Sectors 31, 70, 71). | `sector` *(int)*, `time`, `flux`, `err` |
| **`lco_data`** | `list` of `dict` | Ground-based photometry from LCO (i-band; facilities: SS, Teid, CTIO, McD). | `dataset` *(str)*, `time`, `flux`, `err`, `airmass`, `fwhm`, `x_pos`, `y_pos` |
| **`lcog_data`** | `list` of `dict` | Ground-based photometry from LCO (g-band; CTIO). | `dataset` *(str)*, `time`, `flux`, `err`, `airmass`, `fwhm`, `x_pos`, `y_pos` |
| **`trap_data`** | `list` of `dict` | Ground-based photometry from TRAPPIST-South ($R_c$-band). | `dataset` *(str)*, `time`, `flux`, `err`, `airmass`, `fwhm`, `x_pos`, `y_pos` |
| **`minerva_data`**| `list` of `dict` | Ground-based photometry from MINERVA (Telescopes T2, T4, T5). | `dataset` *(str)*, `time`, `flux`, `err`, `airmass`, `fwhm`, `x_pos`, `y_pos` |
| **`ngts_data`** | `list` of `dict` | Ground-based photometry from NGTS (12 sub-datasets). | `dataset` *(str)*, `time`, `flux`, `err`, `airmass`, `fwhm`, `bg` |
| **`espresso_data`**| `dict` | Spectroscopic radial velocities and stellar activity indicators from ESPRESSO. | `time`, `rv` *(default: BART)*, `rv_err`, `rv_gz`, `rv_gz_err`, `rv_esp`, `rv_esp_err`, `rv_lbl`, `rv_lbl_err`, `fwhm`, `fwhm_err`, `fwhm_gz`, `fwhm_gz_err`, `fwhm_lbl`, `fwhm_lbl_err`, `bis`, `bis_err`, `asym`, `asym_err`, `ha`, `ha_err`, `ca`, `ca_err`, `na`, `na_err`, `s`, `s_err`, `contrast`, `contrast_err`, `dlw`, `dlw_err` |

---

### Python Quickstart

```python
import pickle

# Load dataset
with open("LHS6050b_complete_dataset.pkl", "rb") as f:
    data = pickle.load(f)

# Inspect TESS Sectors
for sector in data["tess_data"]:
    print(f"TESS Sector {sector['sector']}: {len(sector['time'])} points")

# Access ESPRESSO Radial Velocities (mean-subtracted)
time_rv = data["espresso_data"]["time"]
rv_bart = data["espresso_data"]["rv"]
rv_err = data["espresso_data"]["rv_err"]
