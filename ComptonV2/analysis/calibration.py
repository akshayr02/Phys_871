"""Energy calibration and energy resolution from the 133Ba, 22Na and 137Cs lines."""
import json

import numpy as np

import config as C
import peakfit as pf
from mca_io import read_mca

BA_HI = [(276.3989, 7.16), (302.8508, 18.34), (356.0129, 62.05), (383.8485, 8.94)]
BA_81 = (80.9979 * 32.9 + 79.6142 * 2.65) / (32.9 + 2.65)
E_NA = 510.999
E_CS = 661.657
PRIMARY_LINES = ("Ba303", "Ba356", "Na511", "Cs662")
ESTIMATOR_SYST_REL = 1.0e-3


def _ba_model(x, c356, gain, s356, a276, a303, a356, a384, b0, b1, b2):
    y = np.zeros_like(x, float)
    for (e, _), a in zip(BA_HI, (a276, a303, a356, a384)):
        c = c356 + (e - BA_HI[2][0]) / gain
        s = s356 * np.sqrt(e / BA_HI[2][0])
        y += pf.gauss(x, a, c, s)
    u = (x - c356) / 100.0
    return y + b0 + b1 * u + b2 * u ** 2


def fit_ba_high(spec, window=(1060, 1720)):
    x = spec.channels[window[0]:window[1]]
    y = spec.counts[window[0]:window[1]]
    p0 = [1490, 0.246, 38, 8e3, 2e4, 6e4, 9e3, 60, -10, 1]
    names = ["c356", "gain", "s356", "a276", "a303", "a356", "a384", "b0", "b1", "b2"]
    lo = [1400, 0.2, 15, 0, 0, 0, 0, -1e4, -1e4, -1e4]
    hi = [1600, 0.3, 80, 1e7, 1e7, 1e7, 1e7, 1e4, 1e4, 1e4]
    r = pf.fit(_ba_model, x, y, p0, names, bounds=(lo, hi))
    d = (BA_HI[1][0] - BA_HI[2][0])
    c303 = r["c356"] + d / r["gain"]
    J = np.zeros(len(names)); J[0] = 1.0; J[1] = -d / r["gain"] ** 2
    var303 = J @ r.cov @ J
    cov_303_356 = J @ r.cov[:, 0]
    return r, c303, np.sqrt(var303), cov_303_356


def fit_single(spec, window, model="step", nsig=3.0):
    x = spec.channels[window[0]:window[1]]
    y = spec.counts[window[0]:window[1]]
    mu0 = x[np.argmax(np.convolve(y, np.ones(9) / 9, "same"))]
    sig0 = max(0.03 * mu0, 8.0)
    return pf.fit_peak_window(spec.channels, spec.counts, mu0, sig0, nsig=nsig, model=model)


def measure_lines(runs, model="step"):
    out, fits = {}, {}
    ba = read_mca(C.DATA_DIR / runs["ba133"])
    na = read_mca(C.DATA_DIR / runs["na22"])
    cs = read_mca(C.DATA_DIR / runs["cs137"])

    rba, c303, e303, cov = fit_ba_high(ba)
    fits["Ba_high"] = rba
    out["Ba303"] = dict(E=BA_HI[1][0], c=c303, dc=e303,
                        s=rba["s356"] * np.sqrt(BA_HI[1][0] / BA_HI[2][0]), ds=rba.sigma("s356"))
    out["Ba356"] = dict(E=BA_HI[2][0], c=rba["c356"], dc=rba.sigma("c356"),
                        s=rba["s356"], ds=rba.sigma("s356"))
    out["_cov_303_356"] = cov

    r81 = fit_single(ba, (330, 425), model="lin", nsig=2.0)
    fits["Ba81"] = r81
    out["Ba81"] = dict(E=BA_81, c=r81["mu"], dc=r81.sigma("mu"), s=r81["sig"], ds=r81.sigma("sig"))

    rna = fit_single(na, (1950, 2300), model=model)
    fits["Na511"] = rna
    out["Na511"] = dict(E=E_NA, c=rna["mu"], dc=rna.sigma("mu"), s=rna["sig"], ds=rna.sigma("sig"))

    c_guess = 2000 + int(np.argmax(np.convolve(cs.counts[2000:], np.ones(15) / 15, "same")))
    rcs = fit_single(cs, (c_guess - 150, c_guess + 150), model=model)
    fits["Cs662"] = rcs
    out["Cs662"] = dict(E=E_CS, c=rcs["mu"], dc=rcs.sigma("mu"), s=rcs["sig"], ds=rcs.sigma("sig"))
    rates = {"ba133": ba.counts.sum() / ba.live_time, "na22": na.counts.sum() / na.live_time,
             "cs137": cs.counts.sum() / cs.live_time}
    return out, fits, rates


def linear_calibration(lines, use=PRIMARY_LINES, degree=1):
    keys = list(use)
    ch = np.array([lines[k]["c"] for k in keys])
    E = np.array([lines[k]["E"] for k in keys])
    Vc = np.diag([lines[k]["dc"] ** 2 + (ESTIMATOR_SYST_REL * lines[k]["c"]) ** 2
                  for k in keys])
    if "Ba303" in keys and "Ba356" in keys:
        i, j = keys.index("Ba303"), keys.index("Ba356")
        Vc[i, j] = Vc[j, i] = (lines["_cov_303_356"] + ESTIMATOR_SYST_REL ** 2
                               * lines["Ba303"]["c"] * lines["Ba356"]["c"])
    coef = np.polyfit(ch, E, degree)
    for _ in range(5):
        slope = np.polyval(np.polyder(coef), ch)
        V = np.outer(slope, slope) * Vc
        W = np.linalg.inv(V)
        A = np.vander(ch, degree + 1)
        cov = np.linalg.inv(A.T @ W @ A)
        coef = cov @ A.T @ W @ E
    resid = E - np.polyval(coef, ch)
    chi2 = float(resid @ W @ resid)
    ndf = len(keys) - (degree + 1)
    return dict(coef=coef, cov=cov, chi2=chi2, ndf=ndf, keys=keys, ch=ch, E=E,
                resid=resid, sigE=np.sqrt(np.diag(V)))


def energy(cal, ch, dch=0.0):
    coef, cov = cal["coef"], cal["cov"]
    scale = max(1.0, cal["chi2"] / cal["ndf"]) if cal["ndf"] > 0 else 1.0
    deg = len(coef) - 1
    ch = np.atleast_1d(np.asarray(ch, float))
    A = np.vander(ch, deg + 1)
    E = A @ coef
    var_cal = np.einsum("ij,jk,ik->i", A, cov * scale, A)
    slope = np.polyval(np.polyder(coef), ch)
    var = var_cal + (slope * dch) ** 2
    return E, np.sqrt(var), np.sqrt(var_cal)


def resolution_fit(lines, keys=("Ba81", "Ba356", "Na511", "Cs662"), cal=None):
    gain = cal["coef"][-2] if cal is not None else 0.2463
    E = np.array([lines[k]["E"] for k in keys])
    s = np.array([lines[k]["s"] for k in keys]) * gain
    ds = np.array([lines[k]["ds"] for k in keys]) * gain
    A = np.vstack([np.ones_like(E), E, E ** 2]).T
    y, dy = s ** 2, 2 * s * ds
    W = np.diag(1 / dy ** 2)
    cov = np.linalg.inv(A.T @ W @ A)
    p = cov @ A.T @ W @ y
    return dict(p=p, cov=cov, E=E, sigma=s, dsigma=ds)


def sigma_E(res, E):
    p = res["p"]
    return np.sqrt(np.maximum(p[0] + p[1] * E + p[2] * E ** 2, 1e-6))


def run(verbose=True):
    linesA, fitsA, ratesA = measure_lines(C.CALIB_RUNS["A"])
    linesK, fitsK, ratesK = measure_lines(C.CALIB_RUNS["check"])
    cal = linear_calibration(linesA)
    cal_quad = linear_calibration(linesA, use=("Ba81",) + PRIMARY_LINES, degree=2)
    cal_lin81 = linear_calibration(linesA, use=("Ba81",) + PRIMARY_LINES, degree=1)
    res = resolution_fit(linesA, cal=cal)

    shifts = {k: linesK[k]["c"] / linesA[k]["c"] - 1 for k in ("Ba356", "Ba303", "Na511", "Cs662", "Ba81")}
    low_rate = [shifts["Ba356"], shifts["Na511"], shifts["Ba303"]]
    gain_syst = float(np.sqrt(np.mean(np.square(low_rate))))
    dr_cal = ratesK["cs137"] - ratesA["cs137"]
    rate_coeff = shifts["Cs662"] / dr_cal if dr_cal > 0 else 0.0
    rate_syst = abs(rate_coeff * (ratesA["cs137"] - 200.0))
    if verbose:
        print("== Energy calibration (session A, Sept 24) ==")
        for k in ("Ba81",) + PRIMARY_LINES:
            L = linesA[k]
            print(f"  {k:6s} E={L['E']:8.3f}  ch={L['c']:8.2f} +- {L['dc']:.2f}  sigma={L['s']:.1f} ch")
        print("  linear:", cal["coef"], "chi2/ndf", cal["chi2"], cal["ndf"], "resid", np.round(cal["resid"], 2))
        print("  81 keV deviation from linear:", BA_81 - energy(cal, linesA["Ba81"]["c"])[0])
        print("  gain shifts check/A:", {k: f"{v*100:+.3f}%" for k, v in shifts.items()})
        print("  rates A:", ratesA, " check:", ratesK)
        print(f"  gain syst (low-rate lines) {gain_syst*100:.3f}%   rate-gain syst {rate_syst*100:.3f}%")
        print("  resolution p:", res["p"], " FWHM(662)/E =", 2.3548 * sigma_E(res, 661.657) / 661.657)
    out = dict(lines=linesA, lines_check=linesK, cal=cal, cal_quad=cal_quad, cal_lin81=cal_lin81,
               res=res, fits=fitsA, fits_check=fitsK, shifts=shifts, gain_syst=gain_syst,
               rate_syst=rate_syst, rate_coeff=rate_coeff, rates=ratesA, rates_check=ratesK)
    _save(out)
    return out


def _save(out):
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    js = {"linear_coef_keV": list(map(float, out["cal"]["coef"])),
          "linear_cov": out["cal"]["cov"].tolist(),
          "chi2": out["cal"]["chi2"], "ndf": out["cal"]["ndf"],
          "quadratic_coef_keV": list(map(float, out["cal_quad"]["coef"])),
          "resolution_sigma2_coeffs": list(map(float, out["res"]["p"])),
          "gain_shift_check_vs_A": out["shifts"], "gain_syst": out["gain_syst"],
          "rate_gain_syst": out["rate_syst"],
          "lines": {k: v for k, v in out["lines"].items() if not k.startswith("_")}}
    with open(C.TAB_DIR / "energy_calibration.json", "w") as fh:
        json.dump(js, fh, indent=2, default=float)


if __name__ == "__main__":
    run()
