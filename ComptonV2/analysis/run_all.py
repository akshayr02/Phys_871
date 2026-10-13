"""Run the complete analysis and regenerate every number, table and figure: python run_all.py [--fast]"""
import json
import pickle
import sys
import time

import numpy as np

import beam
import calibration as cb
import config as C
import cross_section as xs
import geometry as G
import kinematics as kin
import physics as ph
import rod_mc
import scatter_fit as sf

FAST = "--fast" in sys.argv
N_MC = 120_000 if FAST else C.MC_HISTORIES
N_SYST = 80_000 if FAST else 150_000


def tenth_degree_shadow(nom, ctx, y0s=(0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6, -0.8),
                        dths=(-1, 0, 1), ods=(1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2)):
    import nai_mc
    import response as R
    resp = R.build()
    fep = lambda E: np.exp(np.interp(np.log(E), np.log(resp["grid"]), np.log(resp["fep"]))) \
        * nai_mc.window_transmission(E)
    N10 = [r for r in ctx.scan if r["theta"] == 10][0]["counts"]
    T, N, _ = sf.load_pair(10)
    aT, daT, _ = beam.photopeak_counts(T)
    aN, daN, _ = beam.photopeak_counts(N)
    meas = ((aT - aN) / aN, np.hypot(daT, daN) / aN)
    sp = nom["setup"]
    sig662 = cb.sigma_E(ctx.res, ph.E_CS137)
    pred = {}
    for dth in dths:
        row = []
        for y0 in y0s:
            s = sp.with_(y0=y0, dtheta=dth)
            sh = 1 - G.transmission_0deg(s, theta_deg=10)
            o = rod_mc.transport(10, s, n=60_000)
            add = sum((w * fep(E))[E > ph.E_CS137 - 3 * sig662].sum() for E, w in o) * nom["S0"] * C.LIVE_TIME_DEFAULT
            row.append(add / N10 - sh)
        pred[dth] = np.array(row)
    rem = np.array([1 - G.transmission_0deg(sp.with_(rod_d=od)) for od in ods])
    return dict(y0s=np.array(y0s), ods=np.array(ods), pred=pred, meas=meas, removal0=rem,
                od_nominal=sp.rod_d,
                removal_meas=(ctx.removal["f"], np.hypot(ctx.removal["df_stat"], ctx.removal["df_syst"])))


def od_scan(ctx, angles, V_unc, ods=(0.9, 1.1, 1.3, 1.5, 1.7, 1.9, 2.1, 2.3)):
    out = []
    for od in ods:
        r = xs.chain(xs.Inputs(mode="geometric", od=od), ctx, angles, N_SYST)
        a = np.array([r["angles"][t]["alpha"] for t in angles])
        da = np.array([r["angles"][t]["dalpha"] for t in angles])
        m, dm, chi2 = xs.gls_mean(a, np.diag(da ** 2))
        out.append((od, m, dm))
        print(f"  OD = {od:.2f} cm -> <alpha> = {m:.3f} +- {dm:.3f}")
    return np.array(out)


def main():
    t0 = time.time()
    for d in (C.TAB_DIR, C.FIG_DIR, C.CACHE_DIR, C.LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    print("== 1. Energy calibration")
    calo = cb.run()
    print("== 2. Beam characterisation (target-out scan, pile-up, 0-deg removal)")
    ctx = xs.Context(calo)
    angles = C.XS_ANGLES
    fit_idx = [angles.index(a) for a in C.FIT_ANGLES]

    modes = [C.NORMALISATION_MODE] + [m for m in ("transmission", "geometric") if m != C.NORMALISATION_MODE]
    runs, systs_by = {}, {}
    for i, mode in enumerate(modes):
        print(f"== {3 + i}. {'Nominal' if i == 0 else 'Alternative'} chain ({mode} normalisation)")
        runs[mode] = xs.chain(xs.Inputs(mode=mode), ctx, angles, N_MC, verbose=(i == 0),
                              with_thomson=True, variants=True)
        print(f"   tube OD = {runs[mode]['od']:.3f} cm;  systematic variations ...")
        systs_by[mode] = xs.run_systematics(runs[mode], ctx, angles, N_SYST, verbose=(i == 0))
    nom = runs[modes[0]]
    systs = systs_by[modes[0]]

    print("== 5. Material hypotheses")
    materials = {}
    for mat in ["steel"] + C.TARGET_MATERIAL_ALTERNATIVES:
        for mode in ("transmission", "geometric"):
            r = runs[mode] if mat == "steel" else xs.chain(xs.Inputs(mode=mode, material=mat), ctx, angles, N_SYST)
            a = np.array([r["angles"][t]["alpha"] for t in angles])[fit_idx]
            da = np.array([r["angles"][t]["dalpha"] for t in angles])[fit_idx]
            m, dm, chi2 = xs.gls_mean(a, np.diag(da ** 2))
            materials[(mat, mode)] = dict(od=r["od"], alpha=m, dalpha=dm, chi2=chi2,
                                          T0=r["transmission0"], alphas=a)
            print(f"  {mat:6s} {mode:12s}: OD = {r['od']:.3f} cm  T0 = {r['transmission0']:.3f}  "
                  f"<alpha> = {m:.3f} +- {dm:.3f}")

    print("== 6. Target-size consistency (10-deg shadow, 0-deg removal, OD scan)")
    shadow = tenth_degree_shadow(nom, ctx)
    print("  10-deg measured (T-N)/N =", shadow["meas"])
    scan_od = od_scan(ctx, C.FIT_ANGLES, None)

    print("== 7. Statistical combination and hypothesis tests")
    results = {}
    for tag in modes:
        n_, s_ = runs[tag], systs_by[tag]
        V_unc, V_cor = xs.covariance(n_, s_, angles)
        alpha = np.array([n_["angles"][t]["alpha"] for t in angles])
        thkn = np.array([n_["angles"][t]["Th_over_KN"] for t in angles])
        D = np.array([sy["d_alpha"] for sy in s_])
        sel = np.ix_(fit_idx, fit_idx)
        a_f, Vu, Df = alpha[fit_idx], V_unc[sel], D[:, fit_idx]
        Rf = Df / a_f
        one, tf_ = np.ones_like(a_f), thkn[fit_idx]
        mws = xs.mean_with_shifts(a_f, Vu, Df)
        res = dict(alpha=alpha, V_unc=V_unc, V_cor=V_cor, D=D, Th_over_KN=thkn, mean=mws,
                   kn_abs=xs.absolute_test(a_f, Vu, Rf, one),
                   th_abs=xs.absolute_test(a_f, Vu, Rf, tf_),
                   kn_shape=xs.shape_test(a_f, Vu, Rf, one, with_syst=False),
                   th_shape=xs.shape_test(a_f, Vu, Rf, tf_, with_syst=False),
                   kn_shape_tot=xs.shape_test(a_f, Vu, Rf, one, with_syst=True),
                   th_shape_tot=xs.shape_test(a_f, Vu, Rf, tf_, with_syst=True))
        m = mws["mean"]
        res["re"] = (ph.R_E * np.sqrt(m), ph.R_E * 0.5 / np.sqrt(m) * mws["stat"],
                     ph.R_E * 0.5 / np.sqrt(m) * mws["syst"])
        results[tag] = res
        print(f"  [{tag}] <alpha> = {m:.3f} +- {mws['stat']:.3f}(stat+est) +- {mws['syst']:.3f}(syst)"
              f"  chi2(const)={mws['chi2']:.1f}/{len(a_f)-1}"
              f"   r_e = {res['re'][0]*1e13:.3f} +- {res['re'][1]*1e13:.3f} +- {res['re'][2]*1e13:.3f} fm")
        for k in ("kn_abs", "th_abs", "kn_shape", "th_shape", "kn_shape_tot", "th_shape_tot"):
            print(f"     {k:13s}: " + ", ".join(f"{kk}={vv:.4g}" for kk, vv in res[k].items()))

    print("== 8. Kinematics (electron rest energy)")
    gain_rel = dict(C.GAIN_DRIFT)
    kres = {}
    for tag, n_, s_ in (("template", nom, systs),):
        sel_angles = C.FIT_ANGLES
        Ever, V, V_cal, V_gain, V_geo = kin.energy_covariance(n_, [dict(d_E=s["d_E"][fit_idx], d_Epk=s["d_Epk"][fit_idx]) for s in s_],
                                                               sel_angles, calo["cal"], gain_rel)
        Vtot = V + V_cal + V_gain + V_geo
        f1 = kin.fit_mass(n_, sel_angles, Vtot, fix_E0=True)
        f1s = kin.fit_mass(n_, sel_angles, V, fix_E0=True)
        f2 = kin.fit_mass(n_, sel_angles, Vtot, fix_E0=False)
        f1["dm_stat"] = f1s["dm"]
        f1["budget"], f1["dm_syst"] = kin.mass_budget(
            n_, s_, sel_angles, fit_idx, dict(stat=V, cal=V_cal, gain=V_gain))
        f1["m"] = f1s["m"]
        r_ = f1s["E"] - f1s["model"]
        f1["chi2"] = float(r_ @ np.linalg.inv(Vtot) @ r_)
        from scipy.stats import chi2 as _chi2d
        f1["p"] = float(_chi2d.sf(f1["chi2"], f1["ndf"]))
        ns = kin.no_shift_test(Ever, Vtot)
        Eall, Va, Vca, Vga, Vgo = kin.energy_covariance(n_, [dict(d_E=s["d_E"], d_Epk=s["d_Epk"]) for s in s_],
                                                         angles, calo["cal"], gain_rel)
        fall = kin.fit_mass(n_, angles, Va + Vca + Vga + Vgo, fix_E0=True)
        kres[tag] = dict(fix=f1, free=f2, noshift=ns, all=fall, V=V, Vtot=Vtot, angles=sel_angles,
                         V_parts=dict(stat=V, cal=V_cal, gain=V_gain, geo=V_geo))
        print(f"  m c^2 (E0 fixed) = {f1['m']:.1f} +- {f1['dm_stat']:.1f}(stat) +- {f1['dm_syst']:.1f}(syst) keV"
              f"   chi2/ndf = {f1['chi2']:.2f}/{f1['ndf']}")
        print(f"  m c^2, E0 free   = {f2['m']:.1f} +- {f2['dm']:.1f} keV, E0 = {f2['E0']:.1f} +- {f2['dE0']:.1f}"
              f"  (rho = {f2['rho']:.2f})   chi2/ndf = {f2['chi2']:.2f}/{f2['ndf']}")
        print(f"  all angles: m c^2 = {fall['m']:.1f} +- {fall['dm']:.1f}  chi2/ndf = {fall['chi2']:.1f}/{fall['ndf']}")
        print(f"  no-shift (classical) chi2/ndf = {ns['chi2']:.0f}/{ns['ndf']}")

    pred = {t: (nom["angles"][t]["E_pk"], nom["angles"][t]["E_rms"]) for t in angles}
    gauss = sf.extract_all(calo["cal"], calo["res"], pred, angles=angles, verbose=True)
    Eg = np.array([gauss[t]["E"] for t in C.FIT_ANGLES])
    dEg = np.array([np.sqrt(gauss[t]["dE_stat"] ** 2 + gauss[t]["E_syst_est"] ** 2 + gauss[t]["dE_cal"] ** 2)
                    for t in C.FIT_ANGLES])
    xg = np.array([nom["angles"][t]["x_eff"] for t in C.FIT_ANGLES])
    lin_g = kin.linear_fit(xg, Eg, dEg)
    Et = np.array([nom["angles"][t]["E"] for t in C.FIT_ANGLES])
    dEt = np.sqrt(np.diag(kres["template"]["Vtot"]))
    lin_t = kin.linear_fit(xg, Et, dEt)
    print(f"  textbook linear fit (template E'): m = {lin_t['m']:.1f} +- {lin_t['dm']:.1f}, E0 = {lin_t['E0']:.1f} +- {lin_t['dE0']:.1f}")
    print(f"  textbook linear fit (Gaussian E'): m = {lin_g['m']:.1f} +- {lin_g['dm']:.1f}, E0 = {lin_g['E0']:.1f} +- {lin_g['dE0']:.1f}")

    bundle = dict(calo={k: v for k, v in calo.items() if k not in ("fits", "fits_check")},
                  calo_fits=calo["fits"], ctx_scan=ctx.scan, pileup=(ctx.pileup, ctx.dpileup),
                  removal=ctx.removal, angles=angles, fit_angles=C.FIT_ANGLES,
                  nom=_strip(nom), alt=_strip(runs[modes[1]]), systs=systs, systs_alt=systs_by[modes[1]],
                  nominal_mode=modes[0], alt_mode=modes[1],
                  od_trans=runs["transmission"]["od"], od_geom=runs["geometric"]["od"],
                  materials=materials, shadow=shadow, scan_od=scan_od, results=results,
                  kin=kres, gauss={t: _strip_g(g) for t, g in gauss.items()},
                  lin_g=lin_g, lin_t=lin_t, gain_rel=gain_rel, fast=FAST,
                  tabs=ctx.tabs)
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(C.CACHE_DIR / "results.pkl", "wb") as fh:
        pickle.dump(bundle, fh)
    import report
    report.write_all(bundle)
    import figures
    figures.make_all(bundle)
    print(f"done in {time.time() - t0:.0f} s")


def _strip(r):
    out = {k: v for k, v in r.items() if k != "angles"}
    out["profile"] = {k: v for k, v in r["profile"].items() if k != "model_fn"}
    out["angles"] = {}
    for t, d in r["angles"].items():
        dd = {k: v for k, v in d.items() if k not in ("fit",)}
        f = d["fit"]
        dd["fit_p"] = f["fit"].p
        dd["fit_names"] = f["fit"].names
        dd["x"], dd["net"], dd["var"], dd["window"] = f["x"], f["net"], f["var"], f["window"]
        out["angles"][t] = dd
    return out


def _strip_g(g):
    keep = ("theta", "E", "dE_stat", "dE_cal", "E_syst_est", "sigE", "dsigE", "Y", "dY",
            "Y_syst", "chi2", "ndf", "mu", "dmu", "ratio_NT", "T_win", "N_win")
    return {k: g[k] for k in keep if k in g}


if __name__ == "__main__":
    main()
