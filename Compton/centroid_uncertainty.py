#!/usr/bin/env python3
"""
Uncertainty helpers for the channel -> energy calibration.

  mca_peak_stats()                    DppMCA-style peak calculations (moment centroid,
                                      variance FWHM, net area, net-area uncertainty)
                                      plus a propagated centroid uncertainty.
  fit_centroid()                      Poisson-weighted Gaussian + linear-background fit
                                      (same model as refine_centroid), with optional
                                      reduced-chi2 rescaling of the centroid error.
  gaussian_limit_error()              Best-case centroid error  FWHM / (2.3548 * sqrt(S)).
  BIN_ERR_CH                          1/sqrt(12) channel quantisation term.
  angular_sigma(), compton_*()        Energy error from finite angular acceptance.
  calibration_energy_error()          Error on E(ch) from the calibration-fit covariance.
  measured_energy_error()             Calibration + centroid + binning for a measured peak.
  quad_sum()                          Add independent errors in quadrature.

NOTE: mca_peak_stats() must be given RAW (not background-subtracted) counts, because it
estimates the background from the channels at the edges of the ROI and propagates the
variance of that estimate.
"""

import numpy as np
from scipy.optimize import curve_fit
import calibration

MC2_KEV = 510.999                 # electron rest energy (keV)
FWHM_PER_SIGMA = 2.3548           # 2*sqrt(2 ln 2); the MCA doc rounds this to 2.35
BIN_ERR_CH = 1.0 / np.sqrt(12.0)  # std. dev. of a uniform distribution 1 channel wide


def quad_sum(*terms):
    """sqrt(sum of squares) of independent uncertainties."""
    return np.sqrt(sum(np.square(np.asarray(t, dtype=float)) for t in terms))


# ---------------------------------------------------------------------------
# 1. MCA-style peak statistics (DppMCA "Peak Calculations")
# ---------------------------------------------------------------------------
def mca_peak_stats(counts, lo, hi, n_bg=3):
    """
    ROI = channels lo..hi inclusive. The first and last n_bg channels of the ROI are
    the background fields (Bl, Br); the N channels between them are the peak (G).

        b   = (Bl + Br) / (2 n)                 background per channel
        S   = G - b N                           net area
        P   = sum(i C_i) / sum(C_i)             centroid, C_i = raw_i - b
        var = sum((i - P)^2 C_i) / sum(C_i);    FWHM = 2.35 sqrt(var)
        sigma_S = sqrt(G + N^2/(4 n^2) (Bl + Br))    (the doc divides by S for a %)

    The centroid uncertainty is not given in the doc. It is obtained here by
    propagating Poisson errors through P:
        dP/dR_i = (i - P)/S,   dP/db = -sum(i - P)/S,   Var(R_i) = R_i,
        Var(b) = (Bl + Br)/(4 n^2)
    """
    counts = np.asarray(counts, dtype=float)
    if hi - lo + 1 < 2 * n_bg + 3:
        raise ValueError("ROI too narrow for the requested number of background channels")

    roi = counts[lo:hi + 1]
    ch = np.arange(lo, hi + 1, dtype=float)

    Bl, Br = roi[:n_bg].sum(), roi[-n_bg:].sum()
    peak_ch, peak_raw = ch[n_bg:-n_bg], roi[n_bg:-n_bg]
    N, G = len(peak_raw), peak_raw.sum()

    roi = counts[lo:hi + 1]
    ch = np.arange(lo, hi + 1, dtype=float)

    ######
    roi.sum()
    P = np.sum(peak_ch * C) / S

    b = (Bl + Br) / (2.0 * n_bg)
    C = peak_raw - b
    S = C.sum()

    var = np.sum((peak_ch - P) ** 2 * C) / S
    fwhm = 2.35 * np.sqrt(var)
    ####

    P = np.sum(peak_ch * C) / S
    var = np.sum((peak_ch - P) ** 2 * C) / S
    fwhm = 2.35 * np.sqrt(var)

    sigma_S = np.sqrt(G + (N ** 2 / (4.0 * n_bg ** 2)) * (Bl + Br))

    var_b = (Bl + Br) / (4.0 * n_bg ** 2)
    dP_db = -np.sum(peak_ch - P) / S
    sigma_P = np.sqrt(np.sum((peak_ch - P) ** 2 * peak_raw) / S ** 2 + dP_db ** 2 * var_b)

    return {
        "centroid": P, "centroid_err": sigma_P, "fwhm": fwhm,
        "net_area": S, "net_area_err": sigma_S, "net_area_err_pct": 100.0 * sigma_S / S,
        "gross": G, "bg_per_channel": b, "N": N,
    }


# ---------------------------------------------------------------------------
# 2. Gaussian + linear background fit
# ---------------------------------------------------------------------------
def gauss_lin(x, amp, mu, sigma, m, c):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + m * x + c


def fit_centroid(channels, counts, guess, half_width, sigma_y=None, rescale=True):
    """
    Returns (centroid, centroid_err, fwhm, reduced_chi2).

    sigma_y : per-channel 1-sigma errors (array, same length as counts). Default is
              sqrt(counts), which is only correct for RAW counts. For background-
              subtracted data pass sqrt(signal + background) instead.
    rescale : multiply the centroid error by sqrt(max(reduced chi2, 1)).
    """
    channels = np.asarray(channels, dtype=float)
    counts = np.asarray(counts, dtype=float)
    lo = max(int(guess - half_width), 0)
    hi = min(int(guess + half_width) + 1, len(counts))
    x, y = channels[lo:hi], counts[lo:hi]
    sy = np.sqrt(np.maximum(y, 1.0)) if sigma_y is None else np.asarray(sigma_y, float)[lo:hi]

    p0 = [y.max() - y.min(), guess, half_width / 2.0, 0.0, y.min()]
    popt, pcov = curve_fit(gauss_lin, x, y, p0=p0, sigma=sy,
                           absolute_sigma=True, maxfev=20000)

    red_chi2 = np.sum(((y - gauss_lin(x, *popt)) / sy) ** 2) / (len(x) - 5)
    err = np.sqrt(pcov[1, 1])
    if rescale:
        err *= np.sqrt(max(red_chi2, 1.0))

    calibration.final_error(counts, lo, hi, err)
    
    # return popt[1], err, FWHM_PER_SIGMA * abs(popt[2]), red_chi2
    return popt[1], err


def gaussian_limit_error(fwhm, net_counts):
    """Best-case centroid error (no background): sigma / sqrt(S) = FWHM / (2.3548 sqrt(S))."""
    return fwhm / (FWHM_PER_SIGMA * np.sqrt(net_counts))


# ---------------------------------------------------------------------------
# 3. Angular acceptance (finite slit / target width)
# ---------------------------------------------------------------------------
def angular_sigma(w, r, w_target=0.0):
    """
    1-sigma angular spread in radians for a uniform acceptance: detector slit width w
    (and optionally target width w_target) at distance r from the target.
    Each uniform width contributes variance width^2 / (12 r^2).
    """
    return np.sqrt(w ** 2 + w_target ** 2) / (r * np.sqrt(12.0))


def compton_energy(E0, theta_deg):
    """Scattered photon energy (same units as E0) for Compton scattering at theta."""
    return E0 / (1.0 + (E0 / MC2_KEV) * (1.0 - np.cos(np.radians(theta_deg))))


def compton_energy_error_from_angle(E0, theta_deg, w, r, w_target=0.0):
    """Returns (E', sigma_E') with sigma_E' = |dE'/dtheta| * sigma_theta.
    dE'/dtheta = -E'^2 sin(theta) / (m c^2)   (Compton scattering assumed)."""
    Ep = compton_energy(E0, theta_deg)
    dE_dtheta = Ep ** 2 * np.sin(np.radians(theta_deg)) / MC2_KEV
    return Ep, dE_dtheta * angular_sigma(w, r, w_target)


# ---------------------------------------------------------------------------
# 4. Calibration-curve and measured-peak energy errors
# ---------------------------------------------------------------------------
def calibration_energy_error(ch, cov):
    """
    Error on the fitted E(ch) from the polyfit covariance matrix
    (np.polyfit(..., cov=True) ordering: highest power first).
    """
    ch = np.atleast_1d(np.asarray(ch, dtype=float))
    A = np.vander(ch, cov.shape[0])
    return np.sqrt(np.einsum("ij,jk,ik->i", A, cov, A))


def measured_energy_error(ch, centroid_err_ch, coeffs, cov):
    """
    Energy uncertainty of a measured peak at channel `ch`:
        calibration-curve error  (+)  (dE/dch) * sqrt(centroid_err^2 + (1/sqrt12)^2)
    Add the angular term separately if you compare against a theoretical energy.
    """
    ch = np.atleast_1d(np.asarray(ch, dtype=float))
    dEdch = np.polyval(np.polyder(coeffs), ch)
    sig_ch = quad_sum(centroid_err_ch, BIN_ERR_CH)
    return quad_sum(calibration_energy_error(ch, cov), dEdch * sig_ch)