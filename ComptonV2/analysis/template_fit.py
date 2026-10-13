"""Fit of the net spectra with forward-folded Klein-Nishina templates."""
import numpy as np

import config as C
import physics as ph
import response as R
import rod_mc
import scatter_fit as sf
from peakfit import fit


def make_templates(theta, s, S0, live, resp, res, n_mc=300_000, seed=None):
    orders = rod_mc.transport(theta, s, n=n_mc, seed=seed or (100 + theta))
    E1, w1 = orders[0]
    Em = np.concatenate([o[0] for o in orders[1:]])
    wm = np.concatenate([o[1] for o in orders[1:]])
    norm = S0 * live
    pk = R.smear(R.deposit_spectrum(E1, w1, resp, photopeak_only=True), res) * norm
    ct = R.smear(R.deposit_spectrum(E1, w1, resp, continuum_only=True), res) * norm
    ms = R.smear(R.deposit_spectrum(Em, wm, resp), res) * norm
    fep = np.exp(np.interp(np.log(E1), np.log(resp["grid"]), np.log(resp["fep"])))
    import nai_mc
    wp = w1 * fep * nai_mc.window_transmission(E1)
    E_pk = np.sum(wp * E1) / np.sum(wp)
    x_eff = ph.ME_C2 * np.sum(wp * (1 / E1 - 1 / ph.E_CS137)) / np.sum(wp)
    E_rms = np.sqrt(np.sum(wp * (E1 - E_pk) ** 2) / np.sum(wp))
    return dict(theta=theta, pk=pk, ct=ct, ms=ms, Y_pk=pk.sum(), E_pk=E_pk, E_rms=E_rms,
                x_eff=x_eff, ms_frac=ms.sum() / (pk.sum() + ct.sum()),
                n_order=[o[1].sum() * norm for o in orders])


def _dens(spec):
    return spec / R.DE


def _leak_peak(x, nscaled, cal):
    from peakfit import fit as _fit, gauss
    b, a = cal["coef"][0], cal["coef"][1]
    mu0 = (661.657 - a) / b
    m = np.abs(x - mu0) < 250
    f = lambda xx, A, mu, sg, c0, c1: gauss(xx, A, mu, sg) + c0 + c1 * (xx - mu0) / 100
    r = _fit(f, x[m], nscaled[m], [max(nscaled[m].sum() * 0.3, 1), mu0, 80, 1, 0],
             ["A", "mu", "sg", "c0", "c1"], bounds=([0, mu0 - 60, 40, -1e4, -1e4], [1e8, mu0 + 60, 130, 1e5, 1e5]))
    return r["A"], r["mu"], r["sg"]


def fit_template(theta, tmpl, cal, res, rebin=8, nsig=4.0, bkg="lin", beta=1.0,
                 use_template=None, fix_shift=False, leak=None):
    import calibration as cb
    from peakfit import gauss
    T, N, session = sf.load_pair(theta)
    x, d, var, nscaled = sf.net_spectrum(T, N, rebin)
    b, a = cal["coef"][0], cal["coef"][1]
    E_x = a + b * x
    sig_tot = np.hypot(cb.sigma_E(res, tmpl["E_pk"]), tmpl["E_rms"])
    lo, hi = tmpl["E_pk"] - nsig * sig_tot, tmpl["E_pk"] + nsig * sig_tot
    if use_template is None:
        use_template = theta <= 20
    if leak is None:
        leak = theta > 20 and hi > 661.657 - 2 * cb.sigma_E(res, 661.657)
    if leak:
        A_L, mu_L, sg_L = _leak_peak(x, nscaled, cal)
        hi = min(hi, a + b * (mu_L + 3 * sg_L))
    m = (E_x >= lo) & (E_x <= hi)
    xs, ys, vs, Es, ns = x[m], d[m], var[m], E_x[m], nscaled[m]
    x0 = (tmpl["E_pk"] - a) / b
    dens = _dens(tmpl["pk"] + tmpl["ct"] + beta * tmpl["ms"])
    dens_pk = _dens(tmpl["pk"])

    def shape(E, delta):
        return np.interp(E - delta, R.E_CENT, dens, left=0, right=0) * b * rebin

    names = ["alpha", "delta", "c0", "c1"] + (["c2"] if bkg == "quad" else []) \
        + (["kappa"] if use_template else []) + (["kappaL", "eta"] if leak else [])

    def model(xx, *p):
        alpha, delta = p[0], (0.0 if fix_shift else p[1])
        EE = a + b * xx
        u = (xx - x0) / 100.0
        y = alpha * shape(EE, delta) + p[2] + p[3] * u
        k = 4
        if bkg == "quad":
            y = y + p[k] * u ** 2
            k += 1
        if use_template:
            y = y - p[k] * np.interp(xx, xs, ns)
            k += 1
        if leak:
            g = gauss(xx, A_L, mu_L, sg_L)
            y = y - p[k] * g + p[k + 1] * g * (xx - mu_L) / sg_L ** 2
        return y

    p0 = [1.0, 0.0, 0.0, 0.0] + ([0.0] if bkg == "quad" else []) + ([0.0] if use_template else []) \
        + ([0.0, 0.0] if leak else [])
    lo_b = [-5, -40, -1e5, -1e5] + ([-1e5] if bkg == "quad" else []) + ([-1.0] if use_template else []) \
        + ([-1.0, -40.0] if leak else [])
    hi_b = [10, 40, 1e5, 1e5] + ([1e5] if bkg == "quad" else []) + ([1.0] if use_template else []) \
        + ([1.0, 40.0] if leak else [])
    priors = {"delta": (0.0, 1e-3)} if fix_shift else None
    r = fit(model, xs, ys, p0, names, var=vs, bounds=(lo_b, hi_b), priors=priors)
    E_meas = tmpl["E_pk"] + r["delta"]
    dE_cal = float(cb.energy(cal, (E_meas - a) / b)[2][0])
    w2 = np.abs(E_x - E_meas) < 2 * sig_tot
    s_ratio = T.live_time / N.live_time
    N_win = s_ratio * N.counts[: (T.counts.size // rebin) * rebin].reshape(-1, rebin).sum(1)[w2].sum()
    S_win = r["alpha"] * np.sum(np.interp(E_x[w2] - r["delta"], R.E_CENT, dens_pk) * b * rebin)
    return dict(theta=theta, session=session, fit=r, alpha=r["alpha"], dalpha=r.sigma("alpha"),
                delta=r["delta"], ddelta=r.sigma("delta") if not fix_shift else 0.0,
                E=E_meas, dE_stat=r.sigma("delta") if not fix_shift else 0.0, dE_cal=dE_cal,
                Y=r["alpha"] * tmpl["Y_pk"], dY=r.sigma("alpha") * tmpl["Y_pk"],
                chi2=r.chi2, ndf=r.ndf, x=x, net=d, var=var, window=(lo, hi),
                sig_tot=sig_tot, S_win=S_win, N_win=N_win,
                SB=S_win / max(N_win, 1), model=model)


VARIANTS = [
    dict(),
    dict(nsig=3.0), dict(nsig=5.0),
    dict(rebin=4), dict(rebin=16),
    dict(bkg="quad"),
    dict(beta=0.7), dict(beta=1.3),
]


def fit_all(templates, cal, res, verbose=True):
    out = {}
    for th, tmpl in templates.items():
        nom = fit_template(th, tmpl, cal, res)
        alts = [fit_template(th, tmpl, cal, res, **v) for v in VARIANTS[1:]]
        if th == 30:
            alts.append(fit_template(th, tmpl, cal, res, leak=False))
        da = [v["alpha"] - nom["alpha"] for v in alts]
        dE = [v["E"] - nom["E"] for v in alts]
        nom["alpha_syst"] = float(np.sqrt(np.mean(np.square(da))))
        nom["E_syst_est"] = float(np.sqrt(np.mean(np.square(dE))))
        nom["variants"] = alts
        out[th] = nom
        if verbose:
            r = nom["fit"]
            print(f"  {th:3d} deg  alpha = {nom['alpha']:.3f} +- {nom['dalpha']:.3f}(st) +- "
                  f"{nom['alpha_syst']:.3f}(est)   delta = {nom['delta']:+.1f} +- {nom['ddelta']:.1f} keV"
                  f"  E' = {nom['E']:.1f}  chi2/ndf = {nom['chi2']:.1f}/{nom['ndf']}  "
                  f"Y_KN = {tmpl['Y_pk']:.0f}  MS/single = {tmpl['ms_frac']:.2f}  S/B = {nom['SB']:.2f}")
    return out
