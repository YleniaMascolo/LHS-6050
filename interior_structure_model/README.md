# ExoMDN interior structure — LHS 6050 b

Two notebooks to reproduce the interior-structure posterior and ridgeplots for LHS 6050 b.

## Requirements

A working ExoMDN install (`pip install -e .` in an ExoMDN clone, plus its
`requirements.txt`: tensorflow, tensorflow-probability, numpy, pandas, scipy,
matplotlib, seaborn). Both notebooks should be self-contained and you can drop them anywhere inside the
ExoMDN clone (e.g. next to `introduction.ipynb`).

## How to run

1. `run_exomdn_lhs6050b.ipynb` — loads the `mass_radius_Teq` model, runs the inference from
   M, R, T_eq (edit the inputs cell for a different planet), and writes
   `lhs6050b_samples.parquet`.
2. `plot_ridge_lhs6050b.ipynb` — loads that parquet and writes, for both mass and radius
   fractions, a ridgeplot and the standard ExoMDN cornerplot:
   `ridge_mass_fractions_lhs6050b.{pdf,png}`, `ridge_radius_fractions_lhs6050b.{pdf,png}`,
   `corner_mass_fractions_lhs6050b.{pdf,png}`, `corner_radius_fractions_lhs6050b.{pdf,png}`.

