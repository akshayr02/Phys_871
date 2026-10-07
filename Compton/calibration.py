"""
Channel -> energy calibration from three source spectra.

Each spectrum file is a text file with 4096 channels, either:
  - one column  : counts per channel (channel = line number, starting at 0), or
  - two columns : channel, counts

Edit the CONFIG section below with your file names and the peaks you want to
use (channel centroid + known energy). All peaks from all three sources are
combined into one calibration fit:

    E(ch) = a0 + a1*ch               (FIT_ORDER = 1)
    E(ch) = a0 + a1*ch + a2*ch^2     (FIT_ORDER = 2)

Usage:
    python calibration.py
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =============================== CONFIG ====================================

N_CHANNELS = 4096
FIT_ORDER = 1             # 1 = linear, 2 = quadratic
REFINE_CENTROIDS = True  # True: Gaussian + linear-background fit around each
                          # user centroid (also gives centroid uncertainties)
REFINE_HALF_WIDTH = 50    # channels either side of the centroid for the refit
LOG_Y = False              # log scale for the spectra plots
SAVE_CALIBRATED = True    # write <name>_calibrated.txt (energy, counts)

output_path='/Users/akshayr/Library/CloudStorage/OneDrive-Personal/Documents/School_Work/Phys 871/Grad26/analysis/calibrated/'

# Each source: label, file, and a list of (centroid_channel, known_energy_keV).
# The values below are EXAMPLES -- replace with your own files and centroids.
SOURCES = [
    {
        "name": "Cs-137",
        "file": "Grad26/analysis/bkg_sub/cs137_withgain_subtracted.txt",
        "peaks": [(2711.87, 661.657, 200)],
    },
    {
        "name": "Ba-133",
        "file": "Grad26/analysis/bkg_sub/ba133_withgain_subtracted.txt", #middle peak is optional
        "peaks": [(150.67, 30.973, 20), (378, 81, 50), (1500, 356.01, 150)], 
    },
    {
        "name": "Na-22",
        "file": "Grad26/analysis/bkg_sub/na22_withgain_subtracted.txt",
        "peaks": [(2119, 511.0, 150)],
    },
]

# ===========================================================================


def load_spectrum(path, n_channels=N_CHANNELS):
    channel, counts = np.loadtxt(path, unpack=True)
    return channel, counts


def gauss_lin(x, amp, mu, sigma, m, c):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + m * x + c


def refine_centroid(channels, counts, guess, half_width):
    """Fit a Gaussian + linear background near `guess`.
    Returns (centroid, uncertainty); falls back to (guess, nan) on failure."""
    lo = max(int(guess - half_width), 0)
    hi = min(int(guess + half_width) + 1, len(counts))
    x = channels[lo:hi]
    y = counts[lo:hi]
    if len(x) < 6:
        return guess, np.nan

    p0 = [y.max() - y.min(), guess, half_width / 3, 0.0, y.min()]
    sigma_y = np.sqrt(np.maximum(y, 1.0))  # Poisson errors
    try:
        popt, pcov = curve_fit(gauss_lin, x, y, p0=p0, sigma=sigma_y,
                               absolute_sigma=True, maxfev=10000)
        mu_err = np.sqrt(pcov[1, 1])
        if not np.isfinite(mu_err) or abs(popt[1] - guess) > half_width:
            raise RuntimeError("unreliable fit")
        
        resid = (y - gauss_lin(x, *popt)) / sigma_y
        red_chi2 = np.sum(resid**2) / (len(x) - 5)
        mu_err_scaled = mu_err * np.sqrt(max(red_chi2, 1.0))
        print(red_chi2, mu_err, mu_err_scaled)

        return popt[1], mu_err_scaled
    except (RuntimeError, ValueError):
        print(f"  Warning: Gaussian refit failed near channel {guess}; using given centroid.")
        return guess, np.nan

# def fwhm_error():

# def angular_error():

# def digitization_error():

def main():
    spectra = {}
    cal_ch, cal_E, cal_err, cal_label = [], [], [], []

    # ---- load spectra and gather calibration points ----
    for src in SOURCES:
        print(f"Loading {src['name']} from {src['file']}")
        ch, counts = load_spectrum(src["file"])
        spectra[src["name"]] = (ch, counts)

        for centroid, energy, hw in src["peaks"]:
            err = np.nan
            if REFINE_CENTROIDS:
                centroid, err = refine_centroid(ch, counts, centroid, hw)
            cal_ch.append(centroid)
            cal_E.append(energy)
            cal_err.append(err)
            cal_label.append(src["name"])

    cal_ch = np.array(cal_ch, dtype=float)
    cal_E = np.array(cal_E, dtype=float)
    cal_err = np.array(cal_err, dtype=float)

    n_pts = len(cal_ch)
    if n_pts < FIT_ORDER + 1:
        raise ValueError(f"Need at least {FIT_ORDER + 1} peaks for an order-{FIT_ORDER} fit, got {n_pts}.")

    # ---- fit E(ch) ----
    # Weighted fit only if every centroid has a valid uncertainty.
    # (Channel errors are propagated to energy using the local slope.)
    weighted = REFINE_CENTROIDS and np.all(np.isfinite(cal_err)) and np.all(cal_err > 0)
    if weighted:
        slope0 = np.polyfit(cal_ch, cal_E, FIT_ORDER)
        dEdch = np.polyval(np.polyder(slope0), cal_ch)
        w = 1.0 / np.abs(dEdch * cal_err)
        coeffs, cov = np.polyfit(cal_ch, cal_E, FIT_ORDER, w=w, cov="unscaled")
    elif n_pts > FIT_ORDER + 1:
        coeffs, cov = np.polyfit(cal_ch, cal_E, FIT_ORDER, cov=True)
    else:
        coeffs, cov = np.polyfit(cal_ch, cal_E, FIT_ORDER), None

    calib = np.poly1d(coeffs)
    residuals = cal_E - calib(cal_ch)

    # ---- report ----
    print("\n=========== Calibration result ===========")
    names = [f"a{i}" for i in range(FIT_ORDER, -1, -1)]
    for i, (n, c) in enumerate(zip(names, coeffs)):
        unc = f" +/- {np.sqrt(cov[i, i]):.3e}" if cov is not None else ""
        power = FIT_ORDER - i
        print(f"  {n} (ch^{power}) = {c:.6e}{unc}")
    terms = " + ".join(f"{c:.6g}*ch^{FIT_ORDER - i}" for i, c in enumerate(coeffs))
    print(f"  E(keV) = {terms}")
    print(f"  RMS residual = {np.sqrt(np.mean(residuals**2)):.3f} keV\n")

    print(f"  {'Source':<8}{'Centroid':>10}{'E_known':>10}{'E_fit':>10}{'Resid':>9}")
    for lab, c, e, r in zip(cal_label, cal_ch, cal_E, residuals):
        print(f"  {lab:<8}{c:>10.2f}{e:>10.3f}{e - r:>10.3f}{r:>9.3f}")

    # ---- plot 1: calibration curve + residuals ----
    fig1, (ax, axr) = plt.subplots(
        2, 1, figsize=(7, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    xx = np.linspace(0, N_CHANNELS - 1, 500)
    ax.plot(xx, calib(xx), "k-", lw=1, label=f"order-{FIT_ORDER} fit")
    colors = {s["name"]: f"C{i}" for i, s in enumerate(SOURCES)}
    for name in colors:
        m = np.array([l == name for l in cal_label])
        ax.errorbar(cal_ch[m], cal_E[m], xerr=np.where(np.isfinite(cal_err[m]), cal_err[m], 0),
                    fmt="o", color=colors[name], label=name)
        axr.plot(cal_ch[m], residuals[m], "o", color=colors[name])
    axr.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("Energy (keV)")
    axr.set_ylabel("Residual (keV)")
    axr.set_xlabel("Channel")
    ax.legend()
    ax.grid(alpha=0.3)
    axr.grid(alpha=0.3)
    fig1.suptitle("Energy calibration")
    fig1.tight_layout()

    # ---- plot 2: calibrated spectra ----
    fig2, axes = plt.subplots(len(SOURCES), 1, figsize=(9, 3 * len(SOURCES)), sharex=True)
    axes = np.atleast_1d(axes)
    for a, src in zip(axes, SOURCES):
        ch, counts = spectra[src["name"]]
        energy = calib(ch)
        a.step(energy, counts, where="mid", lw=0.8, color=colors[src["name"]])
        for _, e_known, hw in src["peaks"]:
            a.axvline(e_known, color="gray", ls="--", lw=0.8)
            a.annotate(f"{e_known:g}", (e_known, 1), xycoords=("data", "axes fraction"),
                       xytext=(2, -10), textcoords="offset points", fontsize=8, color="gray")
        if LOG_Y:
            a.set_yscale("log")
        a.set_ylabel("Counts")
        a.set_title(src["name"], loc="left", fontsize=10)
        a.grid(alpha=0.3)
    axes[-1].set_xlabel("Energy (keV)")
    fig2.tight_layout()

    # ---- save calibrated spectra ----
    if SAVE_CALIBRATED:
        for src in SOURCES:
            ch, counts = spectra[src["name"]]
            out = f"{output_path}{src['name']}_subtracted_calibrated.txt"
            np.savetxt(out, np.column_stack([calib(ch), counts]),
                       header="energy_keV counts", fmt=["%.5f", "%.0f"])
            print(f"Saved {out}")

    plt.show()


if __name__ == "__main__":
    main()