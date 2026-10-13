"""Gaussian-centroid extraction of the scattered line from the net spectra (cross-check)."""
import numpy as np

import calibration as cb
import config as C
import peakfit as pf
import physics as ph
from mca_io import read_mca

REBIN = 8


def load_pair(theta):
    ft, fn, session = C.ANGLE_RUNS[theta]
    T = read_mca(C.DATA_DIR / ft)
    N = read_mca(C.DATA_DIR / fn)
    return T, N, session


def net_spectrum(T, N, rebin=REBIN):
    s = T.live_time / N.live_time
    t = pf.rebin(T.counts, rebin)
    n = pf.rebin(N.counts, rebin)
    x = pf.rebin_x(rebin, T.counts.size)
    return x, t - s * n, t + s ** 2 * n, n * s


def _template_model(base, template):
    def f(x, *p):
        kappa = p[-1]
        return base(x, *p[:-1]) - kappa * np.interp(x, template[0], template[1])
    return f


SIG_PRIOR_REL = 0.15


def fit_scattered(theta, cal, res, sig_kin_keV=0.0, model="step", nsig=3.0,
                  rebin=REBIN, use_template=None, E_seed=None, sig_prior=True):
    T, N, session = load_pair(theta)
    x, d, var, nscaled = net_spectrum(T, N, rebin)
    a, b = cal["coef"][1], cal["coef"][0]
    e_nom = ph.compton_energy(theta) if E_seed is None else E_seed
    mu0 = (e_nom - a) / b
    sig_res = cb.sigma_E(res, e_nom)
    sig0 = np.hypot(sig_res, sig_kin_keV) / b
    priors = {"sig": (sig0, SIG_PRIOR_REL * sig0)} if sig_prior else None
    if use_template is None:
        use_template = theta <= 20
    base, names = pf.MODELS[model]
    xlim = (mu0 - 4 * sig0, mu0 + 4 * sig0)
    mu, sig = mu0, sig0
    tmpl = (x, nscaled)
    f = _template_model(base, tmpl) if use_template else base
    fnames = names + (["kappa"] if use_template else [])
    r = None
    for _ in range(5):
        lo, hi = max(mu - nsig * sig, xlim[0]), min(mu + nsig * sig, xlim[1])
        m = (x >= lo) & (x <= hi)
        xs, ys, vs = x[m], d[m], var[m]
        area0 = max(np.sum(ys) * rebin, 10.0)
        p0 = [area0, mu, sig] + ([0.0] if model == "step" else []) + [0.0, 0.0]
        lo_b = [-1e5, mu0 - 2 * sig0, 0.4 * sig0] + ([-1e5] if model == "step" else []) + [-1e5, -1e5]
        hi_b = [1e8, mu0 + 2 * sig0, 2.5 * sig0] + ([1e5] if model == "step" else []) + [1e5, 1e5]
        if use_template:
            p0 += [0.0]; lo_b += [-1.0]; hi_b += [1.0]
        p0 = list(np.clip(p0, np.array(lo_b) + 1e-9, np.array(hi_b) - 1e-9))
        r = pf.fit(f, xs, ys, p0, fnames, var=vs, bounds=(lo_b, hi_b), priors=priors)
        if abs(r["mu"] - mu) < 0.05 * sig and abs(r["sig"] - sig) < 0.05 * sig:
            break
        mu, sig = r["mu"], r["sig"]
    yld = r["area"] / rebin
    dyld = r.sigma("area") / rebin
    E, dE_tot, dE_cal = cb.energy(cal, r["mu"], r.sigma("mu"))
    m2 = np.abs(x - r["mu"]) < 2 * r["sig"]
    s = T.live_time / N.live_time
    t_win = pf.rebin(T.counts, rebin)[m2].sum()
    n_win = s * pf.rebin(N.counts, rebin)[m2].sum()
    return dict(theta=theta, session=session, fit=r, x=x, net=d, var=var,
                template=nscaled if use_template else None,
                mu=r["mu"], dmu=r.sigma("mu"), sig=r["sig"], dsig=r.sigma("sig"),
                E=float(E[0]), dE_stat=float(np.sqrt(dE_tot[0] ** 2 - dE_cal[0] ** 2)),
                dE_cal=float(dE_cal[0]), sigE=r["sig"] * b, dsigE=r.sigma("sig") * b,
                Y=yld, dY=dyld, chi2=r.chi2, ndf=r.ndf,
                T_win=t_win, N_win=n_win, ratio_NT=n_win / max(t_win, 1),
                live_time=T.live_time)


VARIANTS = [
    dict(model="lin", nsig=3.0, rebin=8),
    dict(model="step", nsig=3.0, rebin=8),
    dict(model="lin", nsig=2.5, rebin=8),
    dict(model="lin", nsig=3.5, rebin=8),
    dict(model="lin", nsig=3.0, rebin=4),
    dict(model="lin", nsig=3.0, rebin=16),
]


def extract_all(cal, res, pred=None, angles=None, verbose=True):
    angles = C.ENERGY_ANGLES if angles is None else angles
    pred = pred or {}
    out = {}
    for th in angles:
        Es, sk = pred.get(th, (None, 0.0))
        nom = fit_scattered(th, cal, res, sk, E_seed=Es, **VARIANTS[0])
        alts = []
        for v in VARIANTS[1:]:
            try:
                alts.append(fit_scattered(th, cal, res, sk, E_seed=Es, **v))
            except Exception as exc:
                print("  variant failed", th, v, exc)
        try:
            alts.append(fit_scattered(th, cal, res, sk, E_seed=Es, sig_prior=False, **VARIANTS[0]))
        except Exception as exc:
            print("  free-width variant failed", th, exc)
        if th <= 20:
            alts.append(fit_scattered(th, cal, res, sk, E_seed=Es, use_template=False, **VARIANTS[0]))
        dY = [a["Y"] - nom["Y"] for a in alts]
        dE = [a["E"] - nom["E"] for a in alts]
        nom["Y_syst"] = float(np.sqrt(np.mean(np.square(dY)))) if dY else 0.0
        nom["E_syst_est"] = float(np.sqrt(np.mean(np.square(dE)))) if dE else 0.0
        nom["variants"] = alts
        out[th] = nom
        if verbose:
            r = nom["fit"]
            kap = f" kappa={r['kappa']:+.4f}+-{r.sigma('kappa'):.4f}" if "kappa" in r.names else ""
            print(f"  {th:3d} deg [{nom['session']}]  E' = {nom['E']:6.1f} +- {nom['dE_stat']:.1f}(st) "
                  f"+- {nom['dE_cal']:.1f}(cal) +- {nom['E_syst_est']:.1f}(est) keV  "
                  f"(KN {ph.compton_energy(th):6.1f})  sigE={nom['sigE']:5.1f}  "
                  f"Y = {nom['Y']:7.0f} +- {nom['dY']:4.0f}(st) +- {nom['Y_syst']:4.0f}(est)  "
                  f"chi2/ndf={nom['chi2']:.1f}/{nom['ndf']}  N/T={nom['ratio_NT']:.2f}{kap}")
    return out


if __name__ == "__main__":
    o = cb.run(verbose=False)
    extract_all(o["cal"], o["res"])
