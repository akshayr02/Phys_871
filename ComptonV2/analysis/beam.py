"""Primary beam from the target-out runs: angular profile, intensity, pile-up and 0-degree removal by the target."""
import numpy as np
from scipy.optimize import brentq, least_squares

import config as C
import geometry as G
import peakfit as pf
import physics as ph
from mca_io import read_mca


def photopeak_counts(spec, mu=None, sig=None):
    if mu is None:
        r = pf.fit_peak_window(spec.channels, spec.counts, 2715, 78, nsig=3.0)
        return r["area"], r.sigma("area"), r
    x = spec.channels.astype(float)
    m = np.abs(x - mu) < 3 * sig
    f = lambda xx, A, c0, c1: pf.gauss(xx, A, mu, sig) + c0 + c1 * (xx - mu) / 100
    r = pf.fit(f, x[m], spec.counts[m], [max(spec.counts[m].sum() * 0.2, 1), 1, 0],
               ["area", "c0", "c1"], bounds=([-1e6, -1e4, -1e4], [1e8, 1e5, 1e5]))
    return r["area"], r.sigma("area"), r


def angular_scan(verbose=True):
    angles = [0] + sorted(C.ANGLE_RUNS)
    files = {0: C.ZERO_NO_TARGET}
    files.update({a: C.ANGLE_RUNS[a][1] for a in C.ANGLE_RUNS})
    s0 = read_mca(C.DATA_DIR / files[0])
    A0, dA0, r0 = photopeak_counts(s0)
    mu0, sig0 = r0["mu"], r0["sig"]
    rows = []
    for a in angles:
        sp = read_mca(C.DATA_DIR / files[a])
        if a <= 20:
            A, dA, r = photopeak_counts(sp)
        else:
            A, dA, r = photopeak_counts(sp, mu0, sig0)
        rows.append(dict(theta=a, counts=A, dcounts=dA, live=sp.live_time,
                         rate=A / sp.live_time, drate=dA / sp.live_time,
                         session=C.ANGLE_RUNS.get(a, (None, None, "A"))[2],
                         total_rate=sp.counts.sum() / sp.live_time,
                         real=sp.real_time))
        if verbose:
            print(f"  no-target {a:3d} deg: 662 photopeak = {A:9.0f} +- {dA:6.0f}  ({A/sp.live_time:8.2f} /s)  "
                  f"total rate {sp.counts.sum()/sp.live_time:7.1f} /s")
    return rows, (mu0, sig0)


def pileup_fraction(verbose=True):
    s0 = read_mca(C.DATA_DIR / C.ZERO_NO_TARGET)
    sb = read_mca(C.DATA_DIR / C.ANGLE_RUNS[90][1])
    hi = 3050
    excess = s0.counts[hi:].sum() - sb.counts[hi:].sum() * s0.live_time / sb.live_time
    A0 = photopeak_counts(s0)[0]
    f = excess / A0
    if verbose:
        print(f"  pile-up: {excess:.0f} excess counts above 743 keV -> photopeak loss {100*f:.2f}%")
    return f, 0.5 * f


def fit_profile(rows, s: G.Setup, eps662_direct, halo=True, verbose=True):
    th = np.array([r["theta"] for r in rows], float)
    y = np.array([r["rate"] for r in rows])
    dy0 = np.array([r["drate"] for r in rows])
    dy = np.hypot(dy0, 0.01 * y)

    def setup(p):
        if halo:
            return s.with_(psi0=np.radians(p[1]), w=np.radians(p[2]), halo=p[3], lam=np.radians(p[4]))
        return s.with_(psi0=np.radians(p[1]), w=np.radians(p[2]), halo=0.0)

    def model(p):
        ss = setup(p)
        g = np.array([G.direct_acceptance(t, ss, 3000) for t in th])
        return p[0] * eps662_direct * g + p[-1]

    S0g = y[0] / eps662_direct / G.direct_acceptance(0, s)
    if halo:
        names = ["S0", "psi0_deg", "w_deg", "halo", "lam_deg", "leak"]
        p0, lo, hi = [S0g, 4.5, 0.6, 0.05, 3.0, 0.4], [0, 1, 0.05, 0, 0.5, -5], [1e12, 15, 5, 1, 15, 50]
    else:
        names = ["S0", "psi0_deg", "w_deg", "leak"]
        p0, lo, hi = [S0g, 4.5, 0.8, 0.4], [0, 1, 0.05, -5], [1e12, 15, 5, 50]
    res = least_squares(lambda p: (model(p) - y) / dy, p0, bounds=(lo, hi), x_scale="jac")
    J = res.jac
    cov = np.linalg.pinv(J.T @ J)
    chi2 = float(np.sum(res.fun ** 2))
    ndf = len(y) - len(p0)
    scale = max(1.0, chi2 / max(ndf, 1))
    cov = cov * scale
    out = dict(p=res.x, cov=cov, names=names, chi2=chi2, ndf=ndf, theta=th, rate=y,
               drate=dy, model=model(res.x), setup=setup(res.x), halo=halo,
               model_fn=model)
    if verbose:
        print(f"  beam profile fit ({'core+halo' if halo else 'core only'}) chi2/ndf = {chi2:.1f}/{ndf}")
        for n_, v, e in zip(names, res.x, np.sqrt(np.diag(cov))):
            print(f"    {n_:9s} = {v:12.5g} +- {e:.3g}")
        print("    resid/sigma:", np.round(res.fun, 2))
    return out


def removal_measurement(mu0, sig0, verbose=True):
    T6 = read_mca(C.ZERO_PAIR_OCT6[0])
    N6 = read_mca(C.ZERO_PAIR_OCT6[1])
    At, dAt, rt = photopeak_counts(T6)
    An, dAn, rn = photopeak_counts(N6)
    At, dAt = At / T6.live_time, dAt / T6.live_time
    An, dAn = An / N6.live_time, dAn / N6.live_time
    f = 1 - At / An
    df_stat = (At / An) * np.hypot(dAt / At, dAn / An)
    df_syst = f * C.REMOVAL_SYST_REL
    n0 = read_mca(C.DATA_DIR / C.ZERO_NO_TARGET)
    dz = read_mca(C.DATA_DIR / C.ZERO_DIFFERENCE, live_time_override=C.LIVE_TIME_DEFAULT)
    f_zd = photopeak_counts(dz)[0] / photopeak_counts(n0)[0]
    if verbose:
        print(f"  0-deg removal (Oct 6 pair): f_rem = {f:.4f} +- {df_stat:.4f}(st) +- {df_syst:.4f}(sy); "
              f"photopeak centroids target {rt['mu']:.1f} / no-target {rn['mu']:.1f};"
              f"  zero_def.mca cross-check {f_zd:.4f}")
    return dict(f=f, df_stat=df_stat, df_syst=df_syst, f_zero_def=f_zd,
                rate_target=At, rate_notarget=An, mu_t=rt["mu"], mu_n=rn["mu"])


def solve_rod_diameter(f_removed, s: G.Setup, addback=C.FORWARD_SCATTER_ADDBACK[0]):
    def g(d):
        return (1 - G.transmission_0deg(s.with_(rod_d=d), addback=addback)) - f_removed
    lo, hi = 0.05, 12.0
    if g(hi) < 0:
        return np.nan
    return brentq(g, lo, hi, xtol=1e-4)
