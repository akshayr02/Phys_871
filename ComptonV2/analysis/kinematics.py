"""Electron rest energy from the scattered-photon energies, 1/E' = 1/E0 + (1 - cos theta)/(m c^2)."""
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as chi2d

import physics as ph


def energy_covariance(nom, systs, angles, cal, gain_rel):
    E = np.array([nom["angles"][t]["E"] for t in angles])
    st = np.array([nom["angles"][t]["dE_stat"] for t in angles])
    es = np.array([nom["angles"][t].get("E_est", 0.0) for t in angles])
    V = np.diag(st ** 2 + es ** 2)
    b, a = cal["coef"][0], cal["coef"][1]
    ch = (E - a) / b
    A = np.vstack([ch, np.ones_like(ch)]).T
    scale = max(1.0, cal["chi2"] / cal["ndf"])
    V_cal = A @ (cal["cov"] * scale) @ A.T
    V_gain = np.zeros((E.size, E.size))
    sess = np.array([nom["angles"][t].get("session", "A") for t in angles])
    for s_, g in gain_rel.items():
        v = np.where(sess == s_, g * E, 0.0)
        V_gain += np.outer(v, v)
    V_geo = np.zeros_like(V)
    for s in systs:
        d = s["d_E"] - s["d_Epk"]
        V_geo += np.outer(d, d)
    return E, V, V_cal, V_gain, V_geo


def fit_mass(nom, angles, V, fix_E0=True):
    E = np.array([nom["angles"][t]["E"] for t in angles])
    x = np.array([nom["angles"][t]["x_eff"] for t in angles])
    Epk = np.array([nom["angles"][t]["E_pk"] for t in angles])
    J = Epk - 1.0 / (1.0 / ph.E_CS137 + x / ph.ME_C2)
    L = np.linalg.cholesky(np.linalg.inv(V))

    def model(p):
        m = p[0]
        e0 = ph.E_CS137 if fix_E0 else p[1]
        return 1.0 / (1.0 / e0 + x / m) + J

    p0 = [511.0] if fix_E0 else [511.0, 661.0]
    r = least_squares(lambda p: L.T @ (E - model(p)), p0)
    Jm = r.jac
    cov = np.linalg.inv(Jm.T @ Jm)
    chi2 = float(np.sum(r.fun ** 2))
    ndf = E.size - len(p0)
    out = dict(m=r.x[0], dm=np.sqrt(cov[0, 0]), chi2=chi2, ndf=ndf,
               p=float(chi2d.sf(chi2, ndf)), x=x, E=E, J=J, model=model(r.x))
    if not fix_E0:
        out.update(E0=r.x[1], dE0=np.sqrt(cov[1, 1]), rho=cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
    return out


def no_shift_test(E, V):
    r = E - ph.E_CS137
    chi2 = float(r @ np.linalg.inv(V) @ r)
    return dict(chi2=chi2, ndf=E.size, p=float(chi2d.sf(chi2, E.size)))


def mass_budget(nom, systs, angles, fit_idx, V_parts):
    E = np.array([nom["angles"][t]["E"] for t in angles])
    x = np.array([nom["angles"][t]["x_eff"] for t in angles])
    Epk = np.array([nom["angles"][t]["E_pk"] for t in angles])
    J = Epk - 1.0 / (1.0 / ph.E_CS137 + x / ph.ME_C2)

    def fit(Ev, V):
        L = np.linalg.cholesky(np.linalg.inv(V))
        r = least_squares(lambda p: L.T @ (Ev - (1.0 / (1.0 / ph.E_CS137 + x / p[0]) + J)), [511.0])
        return r.x[0], np.sqrt(np.linalg.inv(r.jac.T @ r.jac)[0, 0])

    Vs = V_parts["stat"]
    m0, e0 = fit(E, Vs)
    rows = [("stat", "Statistical + signal extraction", e0)]
    for key, lab in (("cal", "Energy calibration"), ("gain", "Gain drift")):
        _, e = fit(E, Vs + V_parts[key])
        rows.append((key, lab, float(np.sqrt(max(e ** 2 - e0 ** 2, 0)))))
    for s in systs:
        d = (s["d_E"] - s["d_Epk"])[fit_idx]
        m1, _ = fit(E + d, Vs)
        rows.append(("geo", s["label"], abs(m1 - m0)))
    total = float(np.sqrt(sum(r[2] ** 2 for r in rows[1:])))
    return rows, total


def linear_fit(x, E, dE):
    y = 1.0 / E
    dy = dE / E ** 2
    W = np.diag(1 / dy ** 2)
    A = np.vstack([np.ones_like(x), x]).T
    cov = np.linalg.inv(A.T @ W @ A)
    c = cov @ A.T @ W @ y
    chi2 = float(np.sum(((y - A @ c) / dy) ** 2))
    return dict(c=c, cov=cov, m=1 / c[1], dm=np.sqrt(cov[1, 1]) / c[1] ** 2,
                E0=1 / c[0], dE0=np.sqrt(cov[0, 0]) / c[0] ** 2, chi2=chi2, ndf=x.size - 2)
