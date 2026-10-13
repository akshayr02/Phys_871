# Cross-check of the earlier calibration scripts (group repository, `Compton/`)

**What those scripts do.** `background_subtraction.py` subtracts a long background run from the calibration-source spectra. `calibration.py` then fits a linear channel-to-energy calibration to Cs-137, Ba-133 and Na-22 peaks. The scripts do not cover the scattering runs: there is no target-in minus target-out subtraction, no peak fits versus angle, and no cross sections.

**How this check was done.** Every number below was reproduced from the raw `.mca` files with the same input files, fit windows, fit function and weighting as the original scripts. The re-implementation is `earlier_calibration.py` (run `python earlier_calibration.py`). The figure `output/figures/energy_calibration_curve.pdf` overlays its residuals on the calibration used in the report.

## Issues that change the physics results

### 1. The Cs-137 point comes from a gain-shifted run (main issue)
The script uses `calibration/cs137_withgain.mca` (Sept 29). That run was taken at 14.5 kcps, five times the rate of any other run, and its 662 keV peak sits at channel 2773. In the Sept 24 calibration run and in the 0° direct-beam run, the same line sits at channels 2713–2720.

| Run | Rate | 662 keV centroid |
|---|---|---|
| `cs137_source.mca` (Sept 24) | 2.9 kcps | 2720 |
| `cs137_source_angle_0_no_target.mca` (Sept 24) | 3.0 kcps | 2713 |
| `cs137_withgain.mca` (Sept 29), used by the earlier calibration | 14.5 kcps | **2773 (+2.0%)** |

Ba-133 and Na-22 barely moved between Sept 24 and Sept 29 (+0.004% and +0.17%). Only the high-rate Cs-137 run shifted, which points to a rate-dependent photomultiplier gain.

Consequences:
- The fit reproduces the earlier plots: Na-22 511 keV lands +6.0 keV off the line (4-point fit) or +6.7 keV (5-point fit). The calibration is internally inconsistent.
- The slope comes out at 0.2409 keV/ch instead of 0.2475 keV/ch, a 2.7% scale error.
- The unscattered 662 keV line of the scattering runs is calibrated to 649–651 keV.
- Scattered energies come out 7.6 keV low at 30° and 3.4 keV low at 60°. This biases the Compton-shift test and the extracted m_e c².

**Fix:** calibrate with the low-rate Sept 24 runs (`ba133_sig`, `na22_sig_sep24`, `cs137_source`), which match the conditions of the scattering runs.

### 2. The subtracted background was recorded at a different gain
In `background_data_with_source_brick.mca` (Sept 25–29) the low-energy hump sits at channel ≈247; in all 850 V runs it sits at ≈402. A channel-scale fit gives a factor of about 0.72, and rescaling the axis cuts the shape mismatch from 988 to 159. This matches the 800 V / 850 V mix-up in the logbook: the 850 V background was only restarted on Sept 29 at 11:34.

Subtracting this file channel by channel from 850 V spectra removes the wrong shape. It shows in `calibrated_subtracted_spectra_4p.png`, where the Na-22 spectrum drops to exactly zero between ≈40 and 70 keV.

**Fix:** the calibration needs no background subtraction, because each peak fit already has its own local background term. Never subtract a spectrum taken at a different gain.

### 3. Negative channels are set to zero after subtraction
`if new_val < 0: new_val = 0` biases every net spectrum upward wherever the true net is close to zero. The output format `%.0f` also rounds every channel.

If the same subtraction were applied to target-in minus target-out spectra:

| Angle | Window | True net | With clipping |
|---|---|---|---|
| 90° | 250–330 keV (peak) | 707 | 1099 (+55%) |
| 60° | 360–450 keV (peak) | 834 | 1104 (+32%) |

**Fix:** keep the signed difference and carry its variance, T + s²N, alongside it.

### 4. Blended lines are assigned single energies
- **The "30.973 keV" peak** is the unresolved Cs K x-ray group (Kα 30.6/31.0 keV and Kβ ≈35 keV), fitted in a ±20-channel window (about ±1.5σ). Its centroid is not 30.973 keV. It also sits where NaI(Tl) is most non-proportional, yet it carries a large lever arm in a linear fit meant for 290–660 keV.
- **The "356 keV" window** (1500 ± 150 ch) also contains the 383.8 keV line, which has 14% of the 356 keV intensity. A single Gaussian is pulled 2–3 keV high, consistent with the +1.5 to +2.1 keV residual in the earlier fit.

**Fix:** fit the 276, 303, 356 and 384 keV lines together with fixed energy spacings, and leave the low-energy x-ray point out of the fit.

## Minor issues
- **Channel numbering.** Channels are numbered from 1, while the MCA and the DppMCA ROI centroids are 0-based. This gives a 1-channel (0.24 keV) offset whenever the calibration is applied to 0-based channel numbers.
- **Robustness.** `load_spectrum` fails (a `NameError` or a division by zero) on software-generated spectra, which have `LIVE_TIME - 0.000000`.
- **Portability.** Absolute macOS paths are hard-coded.
- **Uncertainties.** No uncertainty is propagated from the calibration to the scattered-peak energies.

**A note on peak-position uncertainties.** The FWHM is not the uncertainty of a peak position: it describes the line width, i.e. the detector resolution. The statistical uncertainty of a centroid is about σ/√N, to which calibration and line-shape systematics are added. Quoting the FWHM instead overstates the uncertainty by roughly √N, which is more than 100 for the calibration peaks. The 1/√12-channel digitization term is 0.07 keV and negligible.

## Relation to the analysis in this folder
`calibration.py` replaces the earlier calibration:
- It uses the low-rate Sept 24 runs.
- It fits the ¹³³Ba multiplet jointly.
- It adds an estimator systematic.
- It checks the gain against Sept 29 and propagates the full calibration covariance.

Subtracting an ambient background is not needed for the scattering analysis. The target-in minus target-out difference, taken at the same angle and live time, already removes room background and source leakage.

The group repository also contains the Oct 6 runs, which are in `data/cs137_oct6/`:
- **Oct 6, 0°, target in and target out** (300 s live each): `cs137_source_angle_0_target.mca` and a newer `cs137_source_angle_0_no_target.mca`. They give the 0° beam removal, 0.423 ± 0.003, which sets the target thickness in the nominal analysis. `zero_def.mca` is essentially their difference.
- **Oct 6, target in at 21–29°** (`cs137_source_angle_21…29_target.mca`). These have no target-out partners, and their gain differs by about 0.9% from Sept 24, so they cannot be background-subtracted reliably. Target-out runs at the same angles would make them usable.
