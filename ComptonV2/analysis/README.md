# Compton scattering analysis (¹³⁷Cs, 662 keV, steel-tube target)

This code turns the raw `.mca` spectra into every number, table and figure of the report in `../paper/`.

## Running it

```
python run_all.py          # full statistics, ~6 min
python run_all.py --fast   # quick check, ~4 min
```

A run writes:

| Location                                      | Contents                                                                         |
|-----------------------------------------------|----------------------------------------------------------------------------------|
| `output/figures/`                             | All figures as PDF and PNG (also copied to `../paper/figures/`)                  |
| `output/tables/results_per_angle.csv`         | Per-angle results: energies, yields, efficiencies, cross sections, uncertainties |
| `output/tables/cross_section_systematics.csv` | Signed systematic shifts per angle and per source                                |
| `output/tables/energy_calibration.json`       | Calibration coefficients, covariance and lines                                   |
| `output/tables/results_summary.json`          | Every number quoted in the report                                                |
| `../paper/generated/`                         | LaTeX macros and tables used by the report                                       |
| `output/cache/`                               | Monte Carlo caches and `results.pkl` (safe to delete)                            |
| `output/logs/run_all.log`                     | Log of a run (`python run_all.py > output/logs/run_all.log`)                     |

To regenerate only the figures, or only the tables and macros, from the last run:

```
python figures.py
python report.py
```

## Pipeline

| Module                     | Role                                                                                          |
|----------------------------|-----------------------------------------------------------------------------------------------|
| `config.py`                | **All inputs and their uncertainties** (data files, distances, detector, target, systematics) |
| `mca_io.py`                | Reader for Amptek `.mca` files                                                                |
| `physics.py`               | Compton kinematics, Klein–Nishina and Thomson cross sections, attenuation data                |
| `peakfit.py`               | Weighted least-squares fitting with covariance and priors                                     |
| `calibration.py`           | Energy calibration (joint ¹³³Ba multiplet fit), resolution, gain-stability check              |
| `beam.py`                  | Beam profile and intensity from the target-out runs, pile-up, 0° beam removal                 |
| `nai_mc.py`, `response.py` | Photon-transport Monte Carlo of the NaI(Tl) crystal: efficiency and response                  |
| `geometry.py`, `rod_mc.py` | Beam, tube and detector geometry; single- and multiple-scattering Monte Carlo                 |
| `template_fit.py`          | Fit of the net (in − out) spectra with Klein–Nishina templates (α, δ)                         |
| `cross_section.py`         | Full chain, systematic variations (shift method), hypothesis test                             |
| `kinematics.py`            | Electron rest energy from E′(θ) and its uncertainty budget                                    |
| `scatter_fit.py`           | Model-independent Gaussian-centroid cross-check                                               |
| `figures.py`, `report.py`  | Figures; LaTeX tables, macros and CSV/JSON output                                             |
| `earlier_calibration.py`   | Re-implementation of the earlier calibration scripts (see `docs/`)                            |
| `tools/make_xsec_table.py` | One-off generator of `data/xsec_elam.csv` (Elam photon cross sections)                        |

## Key results

| Quantity                        | Result                                                        |
|---------------------------------|---------------------------------------------------------------|
| Energy calibration              | E = −12.34 + 0.24749·ch keV; FWHM 6.9% at 662 keV             |
| Electron rest energy            | m_e c² = 522 ± 7 (stat) ± 16 (syst) keV (accepted: 511.0 keV) |
| No-shift (classical) hypothesis | χ² = 4252 for 6 degrees of freedom                            |
| Angular shape, Klein–Nishina    | χ²/ν = 7.5/5 (p = 0.18)                                       |
| Angular shape, Thomson          | χ²/ν = 29.8/5 (p = 1.7×10⁻⁵, 4.3σ)                            |
| Absolute scale ⟨data/KN⟩        | 0.78 ± 0.05 (stat) ± 0.11 (syst), 1.9σ below Klein–Nishina    |

## Inputs that were not recorded (edit `config.py` and rerun)

- **Target.** A hollow steel tube with ID/OD = 0.40 (from the photos).
  - The outer diameter is inferred from the 0° beam removal of the Oct 6 target-in/out pair: 0.423 ± 0.009, giving OD = 2.14 cm.
  - The photographic estimate, 1.3 cm, is the alternative normalisation.
  - **A caliper measurement would remove the largest uncertainty of the absolute scale.**
- **Tube lateral offset:** −4 ± 2 mm, constrained by the 10° target-in/out runs.
- **NaI(Tl) crystal:** 2″ diameter (2″ photomultiplier); length 2″ nominal, with 1.5″ and 3″ as a systematic.
- **Detector aperture:** radius 2.54 ± 0.30 cm.
- **Crystal-face position:** ±1.5 cm.
- **Scattering-angle zero:** ±0.6°.
- **Oct 6 runs.** The Oct 6 0° pair and the 21–29° runs are in `../data/cs137_oct6/`, separate from `../data/cs137/` because one of them has the same file name as a different (Sept 24) run.

## Excluded runs (shown in tables and figures, not fitted)

- **20°:** electronic-noise burst in the target-in run.
- **50°:** Pb shielding changed between the target-in and target-out runs.
- **10°:** the scattered line cannot be resolved from the direct beam; this run is used only for the beam profile and the tube offset.
- **21–29° (Oct 6):** target-in only, with no target-out partner.

## Further documentation

`docs/calibration_crosscheck.md` compares this calibration with the earlier calibration scripts of the group repository and explains the rate-dependent gain shift.
