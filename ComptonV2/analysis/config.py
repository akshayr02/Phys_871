"""Inputs of the analysis: (value, 1-sigma uncertainty), lengths in cm. NOT RECORDED marks estimated inputs."""
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent
COMPTON_DIR = ANALYSIS_DIR.parent
DATA_DIR = COMPTON_DIR / "data"
OCT6_DIR = DATA_DIR / "cs137_oct6"

OUT_DIR = ANALYSIS_DIR / "output"
FIG_DIR = OUT_DIR / "figures"
TAB_DIR = OUT_DIR / "tables"
CACHE_DIR = OUT_DIR / "cache"
LOG_DIR = OUT_DIR / "logs"
PAPER_DIR = COMPTON_DIR / "paper"
PAPER_FIG_DIR = PAPER_DIR / "figures"

LIVE_TIME_DEFAULT = 300.0

CALIB_RUNS = {
    "A": {"ba133": "calibration/ba133_sig.mca",
          "na22": "calibration/na22_sig_sep24.mca",
          "cs137": "calibration/cs137_source.mca"},
    "check": {"ba133": "calibration/ba133_withgain.mca",
              "na22": "calibration/na22_withgain.mca",
              "cs137": "calibration/cs137_withgain.mca"},
}

ANGLE_RUNS = {
    10: ("cs137/cs137_source_angle_10_target.mca", "cs137/cs137_source_angle_10_no_target.mca", "A"),
    20: ("cs137/cs137_source_angle_20_target.mca", "cs137/cs137_source_angle_20_no_target.mca", "A"),
    30: ("cs137/cs137_source_angle_30_target.mca", "cs137/cs137_source_angle_30_no_target.mca", "A"),
    40: ("cs137/cs137_source_angle_40_target.mca", "cs137/cs137_source_angle_40_no_target.mca", "A"),
    50: ("cs137/cs137_source_angle_50_target.mca", "cs137/cs137_source_angle_50_no_target.mca", "B"),
    60: ("cs137/cs137_source_angle_60_target.mca", "cs137/cs137_source_angle_60_no_target.mca", "B"),
    70: ("cs137/cs137_source_angle_70_target.mca", "cs137/cs137_source_angle_70_no_target.mca", "B"),
    80: ("cs137/cs137_source_angle_80_target.mca", "cs137/cs137_source_angle_80_no_target.mca", "B"),
    90: ("cs137/cs137_source_angle_90_target.mca", "cs137/cs137_source_angle_90_no_target.mca", "B"),
}
SESSION_LABELS = {"A": "Sept 24", "B": "Oct 1"}
ZERO_NO_TARGET = "cs137/cs137_source_angle_0_no_target.mca"
ZERO_DIFFERENCE = "cs137/zero_def.mca"
ZERO_PAIR_OCT6 = (OCT6_DIR / "cs137_source_angle_0_target.mca",
                  OCT6_DIR / "cs137_source_angle_0_no_target.mca")
UNPAIRED_RUNS = [OCT6_DIR / f"cs137_source_angle_{a}_target.mca" for a in range(21, 30)]

XS_ANGLES = [20, 30, 40, 50, 60, 70, 80, 90]
ENERGY_ANGLES = XS_ANGLES

FLAGGED = {
    20: "electronic-noise burst in the target-in run (16x the usual counts below "
        "channel 70) and a 6.5% direct-beam photopeak deficit not explained by geometry",
    50: "Pb shielding changed between the target-in and target-out runs (logbook "
        "Oct 1, 9:32-9:51); the net spectrum shows a continuum up to 690 keV",
}
FIT_ANGLES = [a for a in XS_ANGLES if a not in FLAGGED]

SOURCE_TO_TARGET = (28.5, 0.5)
SOURCE_TO_DETECTOR = (48.5, 0.5)
ANGLE_UNCERTAINTY = 0.6
FACE_OFFSET = (0.0, 1.5)  # NOT RECORDED

CRYSTAL = {"diameter": 5.08, "length": 5.08}  # length NOT RECORDED
CRYSTAL_ALTERNATIVES = [{"diameter": 5.08, "length": 3.81},
                        {"diameter": 5.08, "length": 7.62}]
APERTURE_RADIUS = (2.54, 0.30)  # NOT RECORDED
WINDOW_AL_CM = (0.05, 0.05)  # NOT RECORDED

TARGET_MATERIAL = "steel"
TARGET_MATERIAL_ALTERNATIVES = ["Al", "Cu"]
STEEL_DENSITY = (7.85, 0.05)
ROD_OD = (1.30, 0.30)  # NOT RECORDED: photographic estimate
ROD_ID_OVER_OD = (0.40, 0.05)
ROD_Y0 = (-0.40, 0.20)
ROD_DIAMETER_PRIOR = ROD_OD
NORMALISATION_MODE = "transmission"
REMOVAL_SYST_REL = 0.02
FORWARD_SCATTER_ADDBACK = (0.02, 0.02)

REL_EFF_SYST = 0.05
PILEUP_REL_UNC = 0.5
BEAM_STABILITY = 0.007
GAIN_DRIFT = {"A": 0.003, "B": 0.010}

MC_HISTORIES = 300_000
