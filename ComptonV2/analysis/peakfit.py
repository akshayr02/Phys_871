"""Weighted least-squares fitting with covariance, priors and peak models."""
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.special import erfc
from scipy.stats import chi2 as chi2_dist

SQ2PI = np.sqrt(2 * np.pi)


def gauss(x, area, mu, sig):
    return area / (SQ2PI * sig) * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def step(x, h, mu, sig):
    return h * 0.5 * erfc((x - mu) / (np.sqrt(2) * sig))


@dataclass
class FitResult:
    names: list
    p: np.ndarray
    cov: np.ndarray
    chi2: float
    ndf: int
    model: callable
    x: np.ndarray
    y: np.ndarray
    var: np.ndarray

    @property
    def err(self):
        return np.sqrt(np.diag(self.cov))

    @property
    def redchi2(self):
        return self.chi2 / max(self.ndf, 1)

    @property
    def pvalue(self):
        return float(chi2_dist.sf(self.chi2, max(self.ndf, 1)))

    def __getitem__(self, name):
        return self.p[self.names.index(name)]

    def sigma(self, name):
        return self.err[self.names.index(name)]

    def covariance(self, a, b):
        return self.cov[self.names.index(a), self.names.index(b)]

    def eval(self, x=None):
        return self.model(self.x if x is None else x, *self.p)

    def summary(self):
        s = [f"chi2/ndf = {self.chi2:.1f}/{self.ndf}  (p = {self.pvalue:.3f})"]
        s += [f"  {n:>8s} = {v:12.5g} +- {e:.3g}" for n, v, e in zip(self.names, self.p, self.err)]
        return "\n".join(s)


def fit(model, x, y, p0, names, var=None, bounds=(-np.inf, np.inf),
        pearson=True, n_iter=3, priors=None):
    priors = priors or {}
    pidx = [(list(names).index(k), m, s) for k, (m, s) in priors.items()]

    def prior_res(q):
        return np.array([(q[i] - m) / s for i, m, s in pidx])

    x = np.asarray(x, float)
    y = np.asarray(y, float)
    fixed_var = var is not None
    v = np.maximum(np.asarray(var, float), 1.0) if fixed_var else np.maximum(y, 1.0)
    p = np.asarray(p0, float)
    for it in range(1 if fixed_var or not pearson else n_iter):
        res = least_squares(lambda q: np.concatenate([(y - model(x, *q)) / np.sqrt(v),
                                                      prior_res(q)]), p,
                            bounds=bounds, x_scale="jac", max_nfev=20000)
        p = res.x
        if not fixed_var and pearson:
            v = np.maximum(model(x, *p), 1.0)
    r = (y - model(x, *p)) / np.sqrt(v)
    J = res.jac
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(J.T @ J)
    chi2 = float(np.sum(r ** 2) + np.sum(prior_res(p) ** 2))
    return FitResult(list(names), p, cov, chi2, x.size + len(pidx) - p.size,
                     model, x, y, v)


def rebin(y, n):
    m = (len(y) // n) * n
    return np.asarray(y[:m], float).reshape(-1, n).sum(axis=1)


def rebin_x(n, nch):
    m = (nch // n) * n
    return np.arange(0, m, n) + (n - 1) / 2.0


def peak_step_lin(x, area, mu, sig, h, c0, c1):
    return gauss(x, area, mu, sig) + step(x, h, mu, sig) + c0 + c1 * (x - mu) / 100.0


def peak_lin(x, area, mu, sig, c0, c1):
    return gauss(x, area, mu, sig) + c0 + c1 * (x - mu) / 100.0


def peak_quad(x, area, mu, sig, c0, c1, c2):
    u = (x - mu) / 100.0
    return gauss(x, area, mu, sig) + c0 + c1 * u + c2 * u ** 2


MODELS = {
    "step": (peak_step_lin, ["area", "mu", "sig", "h", "c0", "c1"]),
    "lin": (peak_lin, ["area", "mu", "sig", "c0", "c1"]),
}


def fit_peak_window(x, y, mu0, sig0, var=None, nsig=3.0, model="step",
                    n_iter=4, sig_bounds=(3, 400), xlim=None):
    f, names = MODELS[model]
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mu, sig = float(mu0), float(sig0)
    r = None
    for _ in range(n_iter):
        lo, hi = mu - nsig * sig, mu + nsig * sig
        if xlim is not None:
            lo, hi = max(lo, xlim[0]), min(hi, xlim[1])
        m = (x >= lo) & (x <= hi)
        xs, ys = x[m], y[m]
        vs = None if var is None else np.asarray(var, float)[m]
        bw = np.median(np.diff(xs)) if xs.size > 1 else 1.0
        area0 = max(np.sum(ys) * 0.6 * bw if var is None else np.sum(ys) * bw, 1.0)
        bg0 = float(np.median(ys[-5:])) if ys.size > 5 else 0.0
        if model == "step":
            p0 = [area0, mu, sig, max(float(np.median(ys[:5])) - bg0, 0.0), bg0, 0.0]
            lo_b = [0, lo, sig_bounds[0], -1e7, -1e7, -1e7]
            hi_b = [1e10, hi, sig_bounds[1], 1e7, 1e7, 1e7]
        else:
            p0 = [area0, mu, sig, bg0, 0.0]
            lo_b = [0, lo, sig_bounds[0], -1e7, -1e7]
            hi_b = [1e10, hi, sig_bounds[1], 1e7, 1e7]
        p0 = np.clip(p0, np.array(lo_b) + 1e-9, np.array(hi_b) - 1e-9)
        r = fit(f, xs, ys, p0, names, var=vs, bounds=(lo_b, hi_b))
        new_mu, new_sig = r["mu"], r["sig"]
        if abs(new_mu - mu) < 0.05 * sig and abs(new_sig - sig) < 0.05 * sig:
            mu, sig = new_mu, new_sig
            break
        mu, sig = new_mu, new_sig
    r.bin_width = bw
    return r
