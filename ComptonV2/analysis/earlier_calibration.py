"""Calibration of the Sept 29 runs with the earlier method (clipped background subtraction, Gaussian-plus-line fits), for comparison."""
import numpy as np
from scipy.optimize import curve_fit

import config as C
import physics as ph
from mca_io import read_mca

BACKGROUND = "background_data_with_source_brick.mca"
SOURCES = [("Cs-137", "calibration/cs137_withgain.mca", [(2711.87, 661.657, 200)]),
           ("Ba-133", "calibration/ba133_withgain.mca",
            [(150.67, 30.973, 20), (378, 81, 50), (1500, 356.01, 150)]),
           ("Na-22", "calibration/na22_withgain.mca", [(2119, 511.0, 150)])]


def subtract(path):
    bkg = read_mca(C.DATA_DIR / BACKGROUND)
    s = read_mca(C.DATA_DIR / path)
    sub = np.round(np.maximum(s.counts - (s.live_time / bkg.live_time) * bkg.counts, 0))
    return np.arange(1, s.counts.size + 1, dtype=float), sub


def _gauss_lin(x, amp, mu, sigma, m, c):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + m * x + c


def _refine(ch, y, guess, hw):
    lo, hi = max(int(guess - hw), 0), min(int(guess + hw) + 1, len(y))
    x, yy = ch[lo:hi], y[lo:hi]
    sy = np.sqrt(np.maximum(yy, 1.0))
    popt, pcov = curve_fit(_gauss_lin, x, yy, p0=[yy.max() - yy.min(), guess, hw / 3, 0.0, yy.min()],
                           sigma=sy, absolute_sigma=True, maxfev=10000)
    red = np.sum(((yy - _gauss_lin(x, *popt)) / sy) ** 2) / (len(x) - 5)
    return popt[1], np.sqrt(pcov[1, 1]) * np.sqrt(max(red, 1.0))


def run(include_81=False):
    ch_, dch_, E_, lab_ = [], [], [], []
    for name, f, peaks in SOURCES:
        ch, y = subtract(f)
        for guess, e, hw in peaks:
            if e == 81 and not include_81:
                continue
            c, dc = _refine(ch, y, guess, hw)
            ch_.append(c); dch_.append(dc); E_.append(e); lab_.append(name)
    ch_, dch_, E_ = map(np.array, (ch_, dch_, E_))
    slope = np.polyfit(ch_, E_, 1)[0]
    coef = np.polyfit(ch_, E_, 1, w=1 / np.abs(slope * dch_))
    return dict(ch=ch_, dch=dch_, E=E_, label=lab_, coef=coef, resid=E_ - np.polyval(coef, ch_))


def energy_of_channel0(coef, ch0):
    return np.polyval(coef, np.asarray(ch0) + 1)


if __name__ == "__main__":
    import calibration as cb
    ours = cb.run(verbose=False)["cal"]
    for inc in (False, True):
        r = run(inc)
        print(f"== earlier calibration ({4 + inc} points): E = {r['coef'][1]:.3f} + {r['coef'][0]:.6f} ch")
        for l, c, e, d in zip(r["label"], r["ch"], r["E"], r["resid"]):
            print(f"   {l:7s} ch {c:8.2f}  E {e:7.2f}  resid {d:+6.2f} keV")
        a, b = ours["coef"][1], ours["coef"][0]
        for th in (0, 30, 60, 90):
            E = ph.compton_energy(th)
            print(f"   theta {th:2d}: true {E:6.1f} keV -> earlier calibration {energy_of_channel0(r['coef'], (E - a) / b):6.1f} keV")
