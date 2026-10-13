# Raw spectra

Amptek MCA8000D `.mca` files (4096 channels), NaI(Tl) detector at +850 V. The file names are the ones used during data taking.

| Files                                                                       | Date            | Use in the analysis                                              |
|-----------------------------------------------------------------------------|-----------------|------------------------------------------------------------------|
| `calibration/ba133_sig.mca`, `na22_sig_sep24.mca`, `cs137_source.mca`       | Sept 24         | Energy calibration and resolution                                |
| `calibration/ba133_withgain.mca`, `na22_withgain.mca`, `cs137_withgain.mca` | Sept 29         | Gain-stability check (the ¹³⁷Cs run is at 14.5 kcps)             |
| `calibration/bkg_withgain_850.mca`                                          | Sept 29 – Oct 1 | Background for the calibration-spectrum figure                   |
| `calibration/bkg_withgain.mca`, `scaled_background.mca`                     | Sept 29         | Not used                                                         |
| `background_data_with_source_brick.mca`                                     | Sept 25 – 29    | Background of the earlier calibration (`earlier_calibration.py`) |
| `cs137/cs137_source_angle_0_no_target.mca`                                  | Sept 24         | Direct beam at 0°: beam profile, intensity, pile-up              |
| `cs137/cs137_source_angle_{10,20,30,40}_{target,no_target}.mca`             | Sept 24         | Scattering runs, session A                                       |
| `cs137/cs137_source_angle_{50,...,90}_{target,no_target}.mca`               | Oct 1           | Scattering runs, session B                                       |
| `cs137/cs137_source_angle_50_target_pbshielding.mca`                        | Sept 29         | Not used                                                         |
| `cs137/zero_def.mca`                                                        | Oct 6           | 0° difference spectrum (target out − in), cross-check            |
| `cs137_oct6/cs137_source_angle_0_{target,no_target}.mca`                    | Oct 6           | 0° beam removal by the target (tube diameter)                    |
| `cs137_oct6/cs137_source_angle_{21,...,29}_target.mca`                      | Oct 6           | Target-in only, no target-out partner; not used                  |

All source and scattering runs have 300 s live time; the two long background runs have 1.6 × 10⁵ s and 3.3 × 10⁵ s. The Oct 6 runs are kept in a separate folder because `cs137_source_angle_0_no_target.mca` exists for both Sept 24 and Oct 6.
