"""Differential cross section: full analysis chain, systematic variations and hypothesis tests."""
from dataclasses import dataclass, replace

import numpy as np

import beam
import config as C
import geometry as G
import nai_mc
import physics as ph
import response as R
import template_fit as tf

LIVE = C.LIVE_TIME_DEFAULT


@dataclass(frozen=True)
class Inputs:
    L1: float = C.SOURCE_TO_TARGET[0]
    L2: float = C.SOURCE_TO_DETECTOR[0]
    face: float = C.FACE_OFFSET[0]
    a: float = C.APERTURE_RADIUS[0]
    dtheta: float = 0.0
    od: float = C.ROD_OD[0]
    hole: float = C.ROD_ID_OVER_OD[0]
    y0: float = C.ROD_Y0[0]
    material: str = C.TARGET_MATERIAL
    density: float = None
    mode: str = C.NORMALISATION_MODE
    f_removal_scale: float = 1.0
    addback: float = C.FORWARD_SCATTER_ADDBACK[0]
    crystal: tuple = (C.CRYSTAL["diameter"], C.CRYSTAL["length"])
    halo: bool = True
    pileup_scale: float = 1.0
    rel_eff: float = 0.0
    S0_scale: float = 1.0

    def with_(self, **kw):
        return replace(self, **kw)


def _crystal_table(tabs, crystal):
    names = [("nominal", C.CRYSTAL)] + [(f"alt{i}", c) for i, c in enumerate(C.CRYSTAL_ALTERNATIVES)]
    for n, c in names:
        if np.isclose(c["diameter"], crystal[0]) and np.isclose(c["length"], crystal[1]):
            return n
    raise KeyError(crystal)


def _tilted_response(resp, k):
    if k == 0:
        return resp
    g = np.log(ph.E_CS137 / resp["grid"]) / np.log(ph.E_CS137 / ph.compton_energy(90))
    r2 = dict(resp)
    r2["fep"] = resp["fep"] * (1 + k * g)
    return r2


class Context:
    def __init__(self, cal_out, verbose=True):
        self.cal = cal_out["cal"]
        self.res = cal_out["res"]
        self.scan, (self.mu662, self.sig662) = beam.angular_scan(verbose=verbose)
        self.pileup, self.dpileup = beam.pileup_fraction(verbose=verbose)
        self.removal = beam.removal_measurement(self.mu662, self.sig662, verbose=verbose)
        self.tabs = nai_mc.run(verbose=False)


def chain(inp: Inputs, ctx: Context, angles, n_mc=C.MC_HISTORIES, verbose=False,
          with_thomson=False, variants=False):
    name = _crystal_table(ctx.tabs, inp.crystal)
    td = ctx.tabs[name + "_direct"]
    eff_dir = float(np.interp(ph.E_CS137, td["E"], td["eps_fep"])) * nai_mc.window_transmission(ph.E_CS137)
    s = G.Setup(L1=inp.L1, L2=inp.L2 + inp.face, a=inp.a, dtheta=inp.dtheta,
                material=inp.material, density=inp.density, hole=inp.hole, rod_d=inp.od,
                y0=inp.y0)
    prof = beam.fit_profile(ctx.scan, s, eff_dir, halo=inp.halo, verbose=verbose)
    sp = prof["setup"]
    S0 = prof["p"][0] * inp.S0_scale / (1 - ctx.pileup * inp.pileup_scale)
    if inp.mode == "transmission":
        od = beam.solve_rod_diameter(ctx.removal["f"] * inp.f_removal_scale, sp, addback=inp.addback)
        sp = sp.with_(rod_d=od)
    resp = _tilted_response(R.build({"diameter": inp.crystal[0], "length": inp.crystal[1]}), inp.rel_eff)
    eff = nai_mc.Efficiency(ctx.tabs[name])
    out = dict(inputs=inp, S0=S0, profile=prof, setup=sp, od=sp.rod_d, eff_dir=eff_dir,
               transmission0=G.transmission_0deg(sp), illum=G.illumination_integral(sp),
               angles={})
    for th in angles:
        tmpl = tf.make_templates(th, sp, S0, LIVE, resp, ctx.res, n_mc=n_mc)
        r = tf.fit_template(th, tmpl, ctx.cal, ctx.res)
        d = dict(session=C.ANGLE_RUNS[th][2],
                 alpha=r["alpha"], dalpha=r["dalpha"], E=r["E"], dE_stat=r["dE_stat"],
                 dE_cal=r["dE_cal"], delta=r["delta"], ddelta=r["ddelta"],
                 Y=r["Y"], dY=r["dY"], Y_KN=tmpl["Y_pk"], E_pk=tmpl["E_pk"], E_rms=tmpl["E_rms"],
                 x_eff=tmpl["x_eff"], ms_frac=tmpl["ms_frac"], chi2=r["chi2"], ndf=r["ndf"],
                 SB=r["SB"], S_win=r["S_win"], N_win=r["N_win"], fit=r, tmpl=tmpl)
        if variants:
            alts = [tf.fit_template(th, tmpl, ctx.cal, ctx.res, **v) for v in tf.VARIANTS[1:]]
            if th == 30:
                alts.append(tf.fit_template(th, tmpl, ctx.cal, ctx.res, leak=False))
            d["alpha_est"] = float(np.sqrt(np.mean([(v["alpha"] - r["alpha"]) ** 2 for v in alts])))
            d["E_est"] = float(np.sqrt(np.mean([(v["E"] - r["E"]) ** 2 for v in alts])))
            d["variant_alphas"] = [v["alpha"] for v in alts]
        mc_kn = G.scatter_mc(th, sp, eff.fep, n=n_mc, seed=500 + th)
        d["acc_KN"] = mc_kn
        if with_thomson:
            mc_th = G.scatter_mc(th, sp, eff.fep, n=n_mc, seed=500 + th,
                                 dsdo=lambda c: ph.thomson_dsdo(np.degrees(np.arccos(np.clip(c, -1, 1)))))
            d["Th_over_KN"] = mc_th["I"] / mc_kn["I"]
        d["dsdo"] = d["alpha"] * ph.klein_nishina_dsdo(th)
        out["angles"][th] = d
        if verbose:
            print(f"  {th:3d}: alpha={d['alpha']:.3f}+-{d['dalpha']:.3f}  E'={d['E']:.1f}  "
                  f"Y_KN={d['Y_KN']:.0f}  chi2={d['chi2']:.0f}/{d['ndf']}")
    return out


def systematic_variations(inp: Inputs):
    v = []
    v.append(("Source-target distance", [inp.with_(L1=inp.L1 + C.SOURCE_TO_TARGET[1]),
                                         inp.with_(L1=inp.L1 - C.SOURCE_TO_TARGET[1])]))
    v.append(("Source-detector distance", [inp.with_(L2=inp.L2 + C.SOURCE_TO_DETECTOR[1]),
                                           inp.with_(L2=inp.L2 - C.SOURCE_TO_DETECTOR[1])]))
    v.append(("Detector-face depth", [inp.with_(face=inp.face + C.FACE_OFFSET[1]),
                                      inp.with_(face=inp.face - C.FACE_OFFSET[1])]))
    v.append(("Aperture radius", [inp.with_(a=inp.a + C.APERTURE_RADIUS[1]),
                                  inp.with_(a=inp.a - C.APERTURE_RADIUS[1])]))
    v.append(("Angle offset", [inp.with_(dtheta=C.ANGLE_UNCERTAINTY),
                               inp.with_(dtheta=-C.ANGLE_UNCERTAINTY)]))
    if inp.mode == "geometric":
        v.append(("Tube outer diameter", [inp.with_(od=inp.od + C.ROD_OD[1]),
                                          inp.with_(od=inp.od - C.ROD_OD[1])]))
    else:
        v.append(("0-deg removal fraction", [inp.with_(f_removal_scale=1 + C.REMOVAL_SYST_REL),
                                             inp.with_(f_removal_scale=1 - C.REMOVAL_SYST_REL)]))
        v.append(("Forward-scatter add-back", [inp.with_(addback=inp.addback + C.FORWARD_SCATTER_ADDBACK[1]),
                                               inp.with_(addback=max(inp.addback - C.FORWARD_SCATTER_ADDBACK[1], 0))]))
    v.append(("Tube lateral offset", [inp.with_(y0=inp.y0 + C.ROD_Y0[1]),
                                      inp.with_(y0=inp.y0 - C.ROD_Y0[1])]))
    v.append(("Tube ID/OD", [inp.with_(hole=inp.hole + C.ROD_ID_OVER_OD[1]),
                             inp.with_(hole=inp.hole - C.ROD_ID_OVER_OD[1])]))
    if inp.material == "steel":
        v.append(("Steel density", [inp.with_(density=C.STEEL_DENSITY[0] + C.STEEL_DENSITY[1]),
                                    inp.with_(density=C.STEEL_DENSITY[0] - C.STEEL_DENSITY[1])]))
    alts = [(c["diameter"], c["length"]) for c in C.CRYSTAL_ALTERNATIVES]
    v.append(("NaI crystal size", [inp.with_(crystal=alts[1]), inp.with_(crystal=alts[0])]))
    v.append(("Relative FEP efficiency", [inp.with_(rel_eff=C.REL_EFF_SYST),
                                          inp.with_(rel_eff=-C.REL_EFF_SYST)]))
    v.append(("Beam-profile model", [inp.with_(halo=False), inp]))
    v.append(("Beam-intensity stability", [inp.with_(S0_scale=1 + C.BEAM_STABILITY),
                                           inp.with_(S0_scale=1 - C.BEAM_STABILITY)]))
    v.append(("Pile-up (0 deg)", [inp.with_(pileup_scale=1 + C.PILEUP_REL_UNC),
                                  inp.with_(pileup_scale=1 - C.PILEUP_REL_UNC)]))
    return v


def run_systematics(nominal, ctx, angles, n_mc=C.MC_HISTORIES, verbose=True):
    inp = nominal["inputs"]
    ref = chain(inp, ctx, angles, n_mc)
    a0 = np.array([nominal["angles"][t]["alpha"] for t in angles])
    E0 = np.array([nominal["angles"][t]["E"] for t in angles])
    x0 = np.array([nominal["angles"][t]["x_eff"] for t in angles])
    Epk0 = np.array([nominal["angles"][t]["E_pk"] for t in angles])
    res = []
    p, cov = nominal["profile"]["p"], nominal["profile"]["cov"]
    dS = np.sqrt(cov[0, 0]) / p[0]
    res.append(dict(label="Beam intensity S0 (fit)", d_alpha=-a0 * dS,
                    d_E=np.zeros_like(E0), d_x=np.zeros_like(x0), d_Epk=np.zeros_like(E0),
                    od_plus=nominal["od"], od_minus=nominal["od"]))
    for label, (vp, vm) in systematic_variations(inp):
        rp = chain(vp, ctx, angles, n_mc)
        rm = chain(vm, ctx, angles, n_mc) if vm is not inp else ref
        ap = np.array([rp["angles"][t]["alpha"] for t in angles])
        am = np.array([rm["angles"][t]["alpha"] for t in angles])
        if vm is inp:
            da = ap - am
            dE = np.array([rp["angles"][t]["E"] for t in angles]) - np.array([rm["angles"][t]["E"] for t in angles])
            dx = np.array([rp["angles"][t]["x_eff"] for t in angles]) - np.array([rm["angles"][t]["x_eff"] for t in angles])
            dpk = np.array([rp["angles"][t]["E_pk"] for t in angles]) - np.array([rm["angles"][t]["E_pk"] for t in angles])
        else:
            da = 0.5 * (ap - am)
            dE = 0.5 * (np.array([rp["angles"][t]["E"] for t in angles])
                        - np.array([rm["angles"][t]["E"] for t in angles]))
            dx = 0.5 * (np.array([rp["angles"][t]["x_eff"] for t in angles])
                        - np.array([rm["angles"][t]["x_eff"] for t in angles]))
            dpk = 0.5 * (np.array([rp["angles"][t]["E_pk"] for t in angles])
                         - np.array([rm["angles"][t]["E_pk"] for t in angles]))
        res.append(dict(label=label, d_alpha=da, d_E=dE, d_x=dx, d_Epk=dpk,
                        od_plus=rp["od"], od_minus=rm["od"]))
        if verbose:
            print(f"  {label:28s} d(alpha)/alpha [%]: " + " ".join(f"{100*x:+6.1f}" for x in da / a0)
                  + f"   dE' [keV]: " + " ".join(f"{x:+5.1f}" for x in dE))
    return res


def covariance(nominal, systs, angles):
    st = np.array([nominal["angles"][t]["dalpha"] for t in angles])
    es = np.array([nominal["angles"][t].get("alpha_est", 0.0) for t in angles])
    V_unc = np.diag(st ** 2 + es ** 2)
    D = np.array([s["d_alpha"] for s in systs])
    V_cor = D.T @ D
    return V_unc, V_cor


def gls_mean(y, V):
    Vi = np.linalg.inv(V)
    one = np.ones_like(y)
    var = 1.0 / (one @ Vi @ one)
    m = var * (one @ Vi @ y)
    chi2 = float((y - m) @ Vi @ (y - m))
    return m, np.sqrt(var), chi2


def mean_with_shifts(alpha, V_unc, D):
    m, dm, chi2 = gls_mean(alpha, V_unc)
    shifts = np.array([gls_mean(alpha + d, V_unc)[0] - m for d in D])
    return dict(mean=m, stat=dm, syst=float(np.sqrt(np.sum(shifts ** 2))), shifts=shifts, chi2=chi2)


def _theory_cov(V_unc, R, t, c=1.0, shape=False):
    V = V_unc.copy()
    for r in R:
        rr = r - r.mean() if shape else r
        v = c * t * rr
        V += np.outer(v, v)
    return V


def absolute_test(alpha, V_unc, R, t):
    from scipy.stats import chi2 as chi2d
    V = _theory_cov(V_unc, R, t)
    r = alpha - t
    chi2 = float(r @ np.linalg.inv(V) @ r)
    return dict(chi2=chi2, ndf=alpha.size, p=float(chi2d.sf(chi2, alpha.size)))


def shape_test(alpha, V_unc, R, t, with_syst=True):
    from scipy.stats import chi2 as chi2d
    c = 1.0
    for _ in range(6):
        V = _theory_cov(V_unc, R, t, c, shape=True) if with_syst else V_unc
        Vi = np.linalg.inv(V)
        c = (t @ Vi @ alpha) / (t @ Vi @ t)
    dc = np.sqrt(1.0 / (t @ Vi @ t))
    r = alpha - c * t
    chi2 = float(r @ Vi @ r)
    ndf = alpha.size - 1
    return dict(c=c, dc=dc, chi2=chi2, ndf=ndf, p=float(chi2d.sf(chi2, ndf)))
