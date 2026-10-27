# Radial Velocity & Stellar Activity Indicators Dataset (`RV_data.txt`)

High-resolution spectroscopic observations containing Radial Velocity (RV) measurements and stellar activity indicators.
All primary measurements are paired with their corresponding uncertainty or error column (`*_err`).

| Column | Description | Unit |
| :--- | :--- | :--- |
| `bjd` | Barycentric Julian Date | Days ($d$) |
| `rv_gz` ± `rv_gz_err` | Radial Velocity (self reduction pipeline) | $\text{km/s}$ |
| `rv_esp` ± `rv_esp_err` | Radial Velocity (ESPRESSO pipeline) | $\text{km/s}$ |
| `rv_bart` ± `rv_bart_err` | Radial Velocity (BART pipeline) | $\text{km/s}$ |
| `rv_lbl` ± `rv_lbl_err` | Radial Velocity (Line-By-Line method) | $\text{km/s}$ |
| `fwhm_gz` ± `fwhm_gz_err` | Full Width at Half Maximum (self reduction) | $\text{km/s}$ |
| `asym` ± `asym_err` | Spectral Line Asymmetry Index | Dimensionless |
| `bis` ± `bis_err` | Bisector Inverse Span (BIS) | $\text{km/s}$ |
| `fwhm_esp` ± `fwhm_esp_err` | Full Width at Half Maximum (ESPRESSO) | $\text{km/s}$ |
| `contrast` ± `contrast_err` | Cross-Correlation Function (CCF) Contrast | $\%$ |
| `s` ± `s_err` | Mount Wilson $S$-index ($S_{\text{MW}}$) | Dimensionless |
| `ha` ± `ha_err` | $\text{H}\alpha$ activity index | Dimensionless |
| `na` ± `na_err` | Sodium doublet ($\text{Na I D}_1\text{D}_2$) index | Dimensionless |
| `ca` ± `ca_err` | Calcium ($\text{Ca II}$) index | Dimensionless |
| `dlw` ± `dlw_err` | Differential Line Width | $\text{km/s}$ |
| `fwhm_lbl` ± `fwhm_lbl_err` | Full Width at Half Maximum (Line-By-Line) | $\text{km/s}$ |

---
