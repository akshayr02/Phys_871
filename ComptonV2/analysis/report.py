"""LaTeX macros and tables for the report, CSV and JSON result tables."""
import csv
import json
import pickle

import numpy as np
from scipy.optimize import least_squares

import calibration as cb
import config as C
import nai_mc
import physics as ph

GEN = C.PAPER_DIR / "generated"


def _mass_fit(E, x, Epk, V):
    J = Epk - 1.0 / (1.0 / ph.E_CS137 + x / ph.ME_C2)
    L = np.linalg.cholesky(np.linalg.inv(V))
    r = least_squares(lambda p: L.T @ (E - (1.0 / (1.0 / ph.E_CS137 + x / p[0]) + J)), [511.0])
    cov = np.linalg.inv(r.jac.T @ r.jac)
    return r.x[0], np.sqrt(cov[0, 0])


def _gls_err(V):
    one = np.ones(V.shape[0])
    return np.sqrt(1.0 / (one @ np.linalg.inv(V) @ one))


def pm(v, e, nd=None):
    if nd is None:
        nd = max(0, int(-np.floor(np.log10(abs(e)))) + 1) if e > 0 else 2
        nd = min(nd, 4)
    return f"{v:.{nd}f} \\pm {e:.{nd}f}"


SYST_LATEX = {
    "Beam intensity S0 (fit)": ("Primary beam", "Beam intensity $S_0$ (profile fit)"),
    "Beam-profile model": ("Primary beam", "Beam-profile model"),
    "Beam-intensity stability": ("Primary beam", "Beam-intensity stability"),
    "Pile-up (0 deg)": ("Primary beam", "Pile-up at $0^\\circ$"),
    "Source-target distance": ("Geometry", "Source--target distance $L_1$"),
    "Source-detector distance": ("Geometry", "Source--detector distance $L_2$"),
    "Detector-face depth": ("Geometry", "Crystal-face position"),
    "Aperture radius": ("Geometry", "Detector aperture radius"),
    "Angle offset": ("Geometry", "Scattering-angle offset"),
    "0-deg removal fraction": ("Target", "$0^\\circ$ beam-removal fraction"),
    "Forward-scatter add-back": ("Target", "Forward-scatter add-back"),
    "Tube outer diameter": ("Target", "Tube outer diameter"),
    "Tube lateral offset": ("Target", "Tube lateral offset"),
    "Tube ID/OD": ("Target", "Tube ID/OD"),
    "Steel density": ("Target", "Steel density"),
    "NaI crystal size": ("Detector", "NaI(Tl) crystal length"),
    "Relative FEP efficiency": ("Detector", "Relative FEP efficiency"),
    "Energy calibration": ("Energy scale", "Energy calibration"),
    "Gain drift": ("Energy scale", "Gain drift (per session)"),
}
GROUP_ORDER = ["Primary beam", "Geometry", "Target", "Detector", "Energy scale"]


def budgets(b):
    angles, fa = b["angles"], b["fit_angles"]
    idx = [angles.index(a) for a in fa]
    out = {}
    for tag, systs in ((b["nominal_mode"], b["systs"]), (b["alt_mode"], b["systs_alt"])):
        res = b["results"][tag]
        mw = res["mean"]
        a = res["alpha"][idx]
        rows = [("Statistical + signal extraction", mw["stat"] / mw["mean"])]
        for s_, sh in zip(systs, mw["shifts"]):
            rows.append((s_["label"], abs(sh) / mw["mean"], 100 * s_["d_alpha"][idx] / a))
        out[tag] = rows
    out["mass"] = [(lab, v) for _, lab, v in b["kin"]["template"]["fix"]["budget"]]
    return out


def write_all(b):
    C.TAB_DIR.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    angles, fa = b["angles"], b["fit_angles"]
    nom, res = b["nom"], b["results"][b["nominal_mode"]]
    resT = b["results"][b["alt_mode"]]
    tabs = b["tabs"]
    eff = nai_mc.Efficiency(tabs["nominal"])
    V_unc, V_cor = res["V_unc"], res["V_cor"]
    bud = budgets(b)

    rows = []
    for i, th in enumerate(angles):
        d = nom["angles"][th]
        g = b["gauss"][th]
        kn = ph.klein_nishina_dsdo(th)
        acc = d["acc_KN"]
        rows.append(dict(
            theta_deg=th, flagged=int(th in C.FLAGGED),
            E_KN_nominal_keV=ph.compton_energy(th), E_pred_acc_keV=d["E_pk"],
            E_meas_keV=d["E"], dE_stat_keV=d["dE_stat"], dE_est_keV=d.get("E_est", 0.0),
            dE_cal_keV=d["dE_cal"], E_gauss_keV=g["E"], dE_gauss_keV=np.hypot(g["dE_stat"], g["E_syst_est"]),
            x_eff=d["x_eff"], theta_mean_deg=acc["theta_mean"], theta_rms_deg=acc["theta_rms"],
            sigma_line_keV=np.hypot(cb.sigma_E(b["calo"]["res"], d["E_pk"]), d["E_rms"]),
            Y_meas=d["Y"], dY_stat=d["dY"], Y_KN=d["Y_KN"], Y_Th=d["Y_KN"] * d["Th_over_KN"],
            eps_FEP=float(eff.fep(ph.compton_energy(th))), PT=float(eff.pt(ph.compton_energy(th))),
            eps_tot=float(eff.tot(ph.compton_energy(th))), self_abs=acc["A_mean"],
            ms_over_single=d["ms_frac"], SB=d["SB"], N_over_T=d["N_win"] / max(d["N_win"] + d["S_win"], 1),
            alpha=res["alpha"][i], dalpha_stat=d["dalpha"], dalpha_est=d.get("alpha_est", 0.0),
            dalpha_corr=np.sqrt(V_cor[i, i]),
            dsdo_1e27=res["alpha"][i] * kn / 1e-27,
            ddsdo_unc_1e27=np.sqrt(V_unc[i, i]) * kn / 1e-27,
            ddsdo_corr_1e27=np.sqrt(V_cor[i, i]) * kn / 1e-27,
            KN_1e27=kn / 1e-27, Thomson_1e27=ph.thomson_dsdo(th) / 1e-27,
            alpha_alt_norm=resT["alpha"][i], dalpha_alt_norm=np.sqrt(resT["V_unc"][i, i]),
            chi2=d["chi2"], ndf=d["ndf"]))
    with open(C.TAB_DIR / "results_per_angle.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})

    with open(C.TAB_DIR / "cross_section_systematics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source"] + [f"dalpha/alpha_{t}deg_percent" for t in fa] + ["on_mean_percent"])
        for r in bud[b["nominal_mode"]][1:]:
            w.writerow([r[0]] + [f"{v:.3g}" for v in r[2]] + [f"{100*r[1]:.3g}"])

    k = b["kin"]["template"]
    cal = b["calo"]["cal"]
    m, dmu, dms = res["mean"]["mean"], res["mean"]["stat"], res["mean"]["syst"]
    mT, dmuT, dmsT = resT["mean"]["mean"], resT["mean"]["stat"], resT["mean"]["syst"]
    re_ = res["re"]
    prof = nom["profile"]
    pnames = dict(zip(prof["names"], zip(prof["p"], np.sqrt(np.diag(prof["cov"])))))
    mats = b["materials"]
    macros = {
        "calA": f"{cal['coef'][1]:.2f}", "calB": f"{cal['coef'][0]:.5f}",
        "calChi": f"{cal['chi2']:.1f}", "calNdf": f"{cal['ndf']}",
        "resFWHMsix": f"{100*2.3548*cb.sigma_E(b['calo']['res'], 661.657)/661.657:.1f}",
        "resFWHMthree": f"{100*2.3548*cb.sigma_E(b['calo']['res'], 300.)/300.:.1f}",
        "gainShiftBa": f"{100*b['calo']['shifts']['Ba356']:.3f}",
        "gainShiftNa": f"{100*b['calo']['shifts']['Na511']:.2f}",
        "gainShiftCs": f"{100*b['calo']['shifts']['Cs662']:.1f}",
        "gainDriftA": f"{100*b['gain_rel']['A']:.1f}", "gainDriftB": f"{100*b['gain_rel']['B']:.1f}",
        "beamStability": f"{100*C.BEAM_STABILITY:.1f}",
        "devEightyOne": f"{cb.BA_81 - cb.energy(cal, b['calo']['lines']['Ba81']['c'])[0][0]:+.1f}",
        "Szero": f"{prof['p'][0]/1e5:.2f}", "dSzero": f"{np.sqrt(prof['cov'][0,0])/1e5:.2f}",
        "psiZero": pm(*pnames["psi0_deg"], 2), "wEdge": pm(*pnames["w_deg"], 2),
        "haloAmp": pm(*pnames["halo"], 3), "haloLam": pm(*pnames["lam_deg"], 2),
        "leakRate": pm(*pnames["leak"], 2),
        "beamChi": f"{prof['chi2']:.1f}", "beamNdf": f"{prof['ndf']}",
        "pileup": f"{100*b['pileup'][0]:.2f}", "dpileup": f"{100*b['pileup'][1]:.2f}",
        "rateZero": f"{b['ctx_scan'][0]['rate']:.1f}",
        "removal": f"{b['removal']['f']:.3f}",
        "removalErr": f"{np.hypot(b['removal']['df_stat'], b['removal']['df_syst']):.3f}",
        "odTrans": f"{b['od_trans']:.2f}", "odPhoto": f"{C.ROD_OD[0]:.2f}", "dodPhoto": f"{C.ROD_OD[1]:.2f}",
        "yOffset": f"{C.ROD_Y0[0]:.1f}", "dyOffset": f"{C.ROD_Y0[1]:.1f}",
        "removalZeroDef": f"{b['removal']['f_zero_def']:.3f}",
        "idod": f"{C.ROD_ID_OVER_OD[0]:.2f}", "didod": f"{C.ROD_ID_OVER_OD[1]:.2f}",
        "shadowMeas": f"{100*b['shadow']['meas'][0]:+.2f}", "shadowErr": f"{100*b['shadow']['meas'][1]:.2f}",
        "shadowPredCentred": f"{100*b['shadow']['pred'][0][0]:+.1f}",
        "shadowPredNominal": f"{100*np.interp(C.ROD_Y0[0], b['shadow']['y0s'][::-1], b['shadow']['pred'][0][::-1]):+.1f}",
        "alphaMean": f"{m:.2f}", "alphaStat": f"{dmu:.2f}", "alphaSyst": f"{dms:.2f}",
        "alphaTot": f"{np.hypot(dmu, dms):.2f}", "alphaConstChi": f"{res['mean']['chi2']:.1f}",
        "alphaMeanAlt": f"{mT:.2f}", "alphaStatAlt": f"{dmuT:.2f}", "alphaSystAlt": f"{dmsT:.2f}",
        "knShapeTotChi": f"{res['kn_shape_tot']['chi2']:.1f}", "knShapeTotP": f"{res['kn_shape_tot']['p']:.2f}",
        "thShapeTotChi": f"{res['th_shape_tot']['chi2']:.1f}", "thShapeTotP": _pfmt(res['th_shape_tot']['p']),
        "reVal": f"{re_[0]*1e13:.2f}", "reStat": f"{re_[1]*1e13:.2f}", "reSyst": f"{re_[2]*1e13:.2f}",
        "knShapeChi": f"{res['kn_shape']['chi2']:.1f}", "knShapeNdf": f"{res['kn_shape']['ndf']}",
        "knShapeP": f"{res['kn_shape']['p']:.2f}",
        "thShapeChi": f"{res['th_shape']['chi2']:.1f}", "thShapeNdf": f"{res['th_shape']['ndf']}",
        "thShapeP": _pfmt(res['th_shape']['p']),
        "knShapeC": f"{res['kn_shape']['c']:.2f}", "knShapeDC": f"{res['kn_shape']['dc']:.2f}",
        "thShapeC": f"{res['th_shape']['c']:.2f}", "thShapeDC": f"{res['th_shape']['dc']:.2f}",
        "knAbsChi": f"{res['kn_abs']['chi2']:.1f}", "knAbsNdf": f"{res['kn_abs']['ndf']}",
        "knAbsP": f"{res['kn_abs']['p']:.2f}",
        "thAbsChi": f"{res['th_abs']['chi2']:.1f}", "thAbsNdf": f"{res['th_abs']['ndf']}",
        "thAbsP": _pfmt(res['th_abs']['p']),
        "thShapeChiAlt": f"{resT['th_shape']['chi2']:.1f}", "knShapeChiAlt": f"{resT['kn_shape']['chi2']:.1f}",
        "mFix": f"{k['fix']['m']:.0f}", "mFixStat": f"{k['fix']['dm_stat']:.0f}",
        "mFixSyst": f"{k['fix']['dm_syst']:.0f}", "mFixChi": f"{k['fix']['chi2']:.1f}",
        "mFixNdf": f"{k['fix']['ndf']}",
        "mFree": f"{k['free']['m']:.0f}", "dmFree": f"{k['free']['dm']:.0f}",
        "EzeroFree": f"{k['free']['E0']:.0f}", "dEzeroFree": f"{k['free']['dE0']:.0f}",
        "mAll": f"{k['all']['m']:.0f}", "dmAll": f"{k['all']['dm']:.0f}",
        "noShiftChi": f"{k['noshift']['chi2']:.0f}", "noShiftNdf": f"{k['noshift']['ndf']}",
        "mLinT": f"{b['lin_t']['m']:.0f}", "dmLinT": f"{b['lin_t']['dm']:.0f}",
        "mLinG": f"{b['lin_g']['m']:.0f}", "dmLinG": f"{b['lin_g']['dm']:.0f}",
        "EzeroLinT": f"{b['lin_t']['E0']:.0f}", "dEzeroLinT": f"{b['lin_t']['dE0']:.0f}",
        "nFit": f"{len(fa)}",
        "selfAbsMin": f"{min(nom['angles'][t]['acc_KN']['A_mean'] for t in fa):.2f}",
        "selfAbsMax": f"{max(nom['angles'][t]['acc_KN']['A_mean'] for t in fa):.2f}",
        "msMin": f"{min(nom['angles'][t]['ms_frac'] for t in fa):.2f}",
        "msMax": f"{max(nom['angles'][t]['ms_frac'] for t in fa):.2f}",
        "thetaRmsMin": f"{min(nom['angles'][t]['acc_KN']['theta_rms'] for t in angles):.1f}",
        "thetaRmsMax": f"{max(nom['angles'][t]['acc_KN']['theta_rms'] for t in angles):.1f}",
        "SBmin": f"{min(nom['angles'][t]['SB'] for t in fa):.2f}",
        "SBmax": f"{max(nom['angles'][t]['SB'] for t in fa):.2f}",
        "epsFEPsix": f"{float(eff.fep(661.657)):.3f}", "epsFEPnin": f"{float(eff.fep(ph.compton_energy(90))):.3f}",
        "PTsix": f"{float(eff.pt(661.657)):.2f}", "PTnin": f"{float(eff.pt(ph.compton_energy(90))):.2f}",
        "epsDirect": f"{nom['eff_dir']:.3f}",
        "odKN": f"{np.interp(1.0, b['scan_od'][::-1, 1], b['scan_od'][::-1, 0]):.2f}",
        "matSteelT": f"{mats[('steel','transmission')]['alpha']:.2f}",
        "matAlT": f"{mats[('Al','transmission')]['alpha']:.2f}",
        "matCuT": f"{mats[('Cu','transmission')]['alpha']:.2f}",
        "matAlG": f"{mats[('Al','geometric')]['alpha']:.2f}",
        "odAlT": f"{mats[('Al','transmission')]['od']:.2f}",
        "odCuT": f"{mats[('Cu','transmission')]['od']:.2f}",
        "KNtotal": f"{ph.klein_nishina_total()/1e-24:.4f}",
        "Lone": f"{C.SOURCE_TO_TARGET[0]:.1f}", "dLone": f"{C.SOURCE_TO_TARGET[1]:.1f}",
        "Ltwo": f"{C.SOURCE_TO_DETECTOR[0]:.1f}", "dLtwo": f"{C.SOURCE_TO_DETECTOR[1]:.1f}",
        "rdist": f"{C.SOURCE_TO_DETECTOR[0] - C.SOURCE_TO_TARGET[0]:.1f}",
        "aperture": f"{C.APERTURE_RADIUS[0]:.2f}", "daperture": f"{C.APERTURE_RADIUS[1]:.2f}",
        "dOmega": f"{np.pi * C.APERTURE_RADIUS[0]**2 / (C.SOURCE_TO_DETECTOR[0] - C.SOURCE_TO_TARGET[0])**2:.3f}",
        "angleUnc": f"{C.ANGLE_UNCERTAINTY:.1f}", "faceUnc": f"{C.FACE_OFFSET[1]:.1f}",
        "relEffSyst": f"{100*C.REL_EFF_SYST:.0f}", "removalSystRel": f"{100*C.REMOVAL_SYST_REL:.0f}",
        "histories": f"{C.MC_HISTORIES:,}".replace(",", "{,}"),
        "mFixRel": f"{100*np.hypot(k['fix']['dm_stat'], k['fix']['dm_syst'])/k['fix']['m']:.1f}",
        "comptonWavelength": f"{2*np.pi*197.3269804/ph.ME_C2:.4f}",
        "reCODATA": f"{ph.R_E*1e13:.3f}",
        "yKNmin": f"{min(nom['angles'][t]['Y_KN'] for t in fa):.0f}",
        "yKNmax": f"{max(nom['angles'][t]['Y_KN'] for t in fa):.0f}",
        "nGoodAngles": f"{len(fa)}",
        "rmsCalResid": f"{np.sqrt(np.mean(np.square(cal['resid']))):.2f}",
        "alphaSigmaKN": f"{abs(1 - m) / np.hypot(dmu, dms):.1f}",
        "thShapeZ": f"{_zscore(res['th_shape_tot']['p']):.1f}",
        "knShapeTotNdf": f"{res['kn_shape_tot']['ndf']}",
        "fitChiMin": f"{min(nom['angles'][t]['chi2'] / nom['angles'][t]['ndf'] for t in angles):.1f}",
        "fitChiMax": f"{max(nom['angles'][t]['chi2'] / nom['angles'][t]['ndf'] for t in angles):.1f}",
        "alphaMinFit": f"{min(res['alpha'][[angles.index(t) for t in fa]]):.2f}",
        "alphaMaxFit": f"{max(res['alpha'][[angles.index(t) for t in fa]]):.2f}",
        "alphaNinety": f"{res['alpha'][angles.index(90)]:.2f}",
        "yNinety": f"{nom['angles'][90]['Y']:.0f}", "dyNinety": f"{nom['angles'][90]['dY']:.0f}",
        "bkgNinety": f"{nom['angles'][90]['N_win']:.0f}",
        "NTmin": f"{min(1 / (1 + nom['angles'][t]['SB']) for t in fa):.2f}",
        "NTmax": f"{max(1 / (1 + nom['angles'][t]['SB']) for t in fa):.2f}",
    }
    with open(GEN / "macros.tex", "w") as fh:
        fh.write("% generated by analysis/report.py -- do not edit\n")
        for kk, v in macros.items():
            fh.write(f"\\newcommand{{\\{kk}}}{{{v}}}\n")

    lines = b["calo"]["lines"]
    with open(GEN / "tab_calibration.tex", "w") as fh:
        fh.write("\\begin{tabular}{lcccc}\n\\hline\\hline\n")
        fh.write("Source & $E_\\gamma$ (keV) & Centroid (ch) & $E_\\gamma-E_{\\rm fit}$ (keV) & FWHM (keV)\\\\\n\\hline\n")
        names = {"Ba81": "$^{133}$Ba$^{a}$", "Ba303": "$^{133}$Ba$^{b}$", "Ba356": "$^{133}$Ba$^{b}$",
                 "Na511": "$^{22}$Na", "Cs662": "$^{137}$Cs"}
        for kk in ("Ba81",) + cb.PRIMARY_LINES:
            L = lines[kk]
            Efit = cb.energy(cal, L["c"])[0][0]
            fw = 2.3548 * L["s"] * cal["coef"][0]
            fh.write(f"{names[kk]} & {L['E']:.2f} & ${L['c']:.1f}\\pm{np.hypot(L['dc'], cb.ESTIMATOR_SYST_REL*L['c']):.1f}$"
                     f" & ${L['E']-Efit:+.2f}$ & {fw:.1f}\\\\\n")
        fh.write("\\hline\\hline\n\\end{tabular}\n")

    with open(GEN / "tab_angles.tex", "w") as fh:
        fh.write("\\begin{tabular}{rcccccccc}\n\\hline\\hline\n")
        fh.write("$\\theta$ & $E'_{\\rm KN}$ & $E'_{\\rm meas}$ & $\\varepsilon_{\\rm FEP}$ & $A_{\\rm self}$ & "
                 "$Y_{\\rm meas}$ & $Y_{\\rm KN}$ & S/B & $d\\sigma/d\\Omega$ \\\\\n")
        fh.write("(deg) & (keV) & (keV) & & & (counts) & (counts) & & ($10^{-27}$\\,cm$^2$/sr)\\\\\n\\hline\n")
        for r in rows:
            th = r["theta_deg"]
            flag = "$^{*}$" if r["flagged"] else ""
            dEt = np.sqrt(r["dE_stat_keV"] ** 2 + r["dE_est_keV"] ** 2)
            fh.write(f"{th}{flag} & {r['E_pred_acc_keV']:.1f} & ${r['E_meas_keV']:.1f}\\pm{dEt:.1f}$ & "
                     f"{r['eps_FEP']:.3f} & {r['self_abs']:.2f} & ${r['Y_meas']:.0f}\\pm{r['dY_stat']:.0f}$ & "
                     f"{r['Y_KN']:.0f} & {r['SB']:.2f} & "
                     f"${r['dsdo_1e27']:.1f}\\pm{r['ddsdo_unc_1e27']:.1f}\\pm{r['ddsdo_corr_1e27']:.1f}$\\\\\n")
        fh.write("\\hline\\hline\n\\end{tabular}\n")

    with open(GEN / "tab_syst.tex", "w") as fh:
        fh.write("\\begin{tabular}{lcc}\n\\hline\\hline\n")
        fh.write("Source & $\\delta\\langle\\alpha\\rangle/\\langle\\alpha\\rangle$ (\\%) & "
                 "$\\delta(m_ec^2)$ (keV)\\\\\n\\hline\n")
        xs_rows = {r[0]: r[1] for r in bud[b["nominal_mode"]]}
        m_rows = dict(bud["mass"])
        fmt = lambda v, f: "---" if v is None else f.format(v)
        fh.write(f"Statistical + signal extraction & {100*xs_rows['Statistical + signal extraction']:.1f} & "
                 f"{m_rows['Statistical + signal extraction']:.1f}\\\\\n")
        for grp in GROUP_ORDER:
            labs = [l for l, (g, _) in SYST_LATEX.items() if g == grp and (l in xs_rows or l in m_rows)]
            if not labs:
                continue
            fh.write(f"\\multicolumn{{3}}{{l}}{{\\textit{{{grp}}}}}\\\\\n")
            for l in labs:
                xv = xs_rows.get(l)
                mv = m_rows.get(l)
                fh.write(f"\\quad {SYST_LATEX[l][1]} & {fmt(None if xv is None else 100*xv, '{:.1f}')} & "
                         f"{fmt(mv, '{:.1f}')}\\\\\n")
        tot_x = np.sqrt(sum(v ** 2 for k_, v in xs_rows.items() if k_ != "Statistical + signal extraction"))
        tot_m = b["kin"]["template"]["fix"]["dm_syst"]
        fh.write("\\hline\n")
        fh.write(f"Total systematic & {100*tot_x:.1f} & {tot_m:.1f}\\\\\n")
        fh.write("\\hline\\hline\n\\end{tabular}\n")

    with open(GEN / "tab_norm.tex", "w") as fh:
        fh.write("\\begin{tabular}{llccc}\n\\hline\\hline\n")
        fh.write("Material & Tube OD from & OD (cm) & $T_0$ & $\\langle\\alpha\\rangle$ (stat)\\\\\n\\hline\n")
        for mat in ["steel"] + C.TARGET_MATERIAL_ALTERNATIVES:
            for mode, lab in (("transmission", "$0^\\circ$ removal"), ("geometric", "photograph")):
                r = mats[(mat, mode)]
                fh.write(f"{mat if mat!='Al' else 'Al'} & {lab} & {r['od']:.2f} & {r['T0']:.2f} & "
                         f"${r['alpha']:.2f}\\pm{r['dalpha']:.2f}$\\\\\n")
        fh.write("\\hline\\hline\n\\end{tabular}\n")

    with open(GEN / "tab_expected.tex", "w") as fh:
        fh.write("\\begin{tabular}{rcccccccc}\n\\hline\\hline\n")
        fh.write("$\\theta$ & $E'$ & $T_e$ & $(d\\sigma/d\\Omega)_{\\rm KN}$ & $(d\\sigma/d\\Omega)_{\\rm Th}$ & KN/Th "
                 "& $\\varepsilon_{\\rm FEP}$ & $Y_{\\rm KN}$ & $Y_{\\rm Th}$\\\\\n")
        fh.write("(deg) & (keV) & (keV) & \\multicolumn{2}{c}{($10^{-27}$\\,cm$^2$/sr)} & & & (counts) & (counts)\\\\\n\\hline\n")
        for th in angles:
            d = nom["angles"][th]
            E = ph.compton_energy(th)
            kn, thm = ph.klein_nishina_dsdo(th) / 1e-27, ph.thomson_dsdo(th) / 1e-27
            fh.write(f"{th} & {E:.1f} & {ph.E_CS137 - E:.1f} & {kn:.1f} & {thm:.1f} & {kn / thm:.2f} & "
                     f"{float(eff.fep(E)):.3f} & {d['Y_KN']:.0f} & {d['Y_KN'] * d['Th_over_KN']:.0f}\\\\\n")
        fh.write("\\hline\\hline\n\\end{tabular}\n")

    from mca_io import read_mca
    with open(GEN / "tab_runs.tex", "w") as fh:
        fh.write("\\begin{tabular}{rlrrrl}\n\\hline\\hline\n")
        fh.write("$\\theta$ (deg) & Date & $N_{\\rm in}$ & $N_{\\rm out}$ & $R_{\\rm out}$ (s$^{-1}$) & Status\\\\\n\\hline\n")
        for th, (ft, fn, sess) in sorted(C.ANGLE_RUNS.items()):
            T = read_mca(C.DATA_DIR / ft)
            N = read_mca(C.DATA_DIR / fn)
            if th in C.FLAGGED:
                status = "flagged$^{a}$" if th == 20 else "flagged$^{b}$"
            elif th == 10:
                status = "beam profile only"
            else:
                status = "used"
            fh.write(f"{th} & {C.SESSION_LABELS[sess]} & {T.counts.sum():.0f} & {N.counts.sum():.0f} & "
                     f"{N.counts.sum() / N.live_time:.0f} & {status}\\\\\n")
        T6, N6 = read_mca(C.ZERO_PAIR_OCT6[0]), read_mca(C.ZERO_PAIR_OCT6[1])
        N24 = read_mca(C.DATA_DIR / C.ZERO_NO_TARGET)
        fh.write(f"0 & Sept 24 & --- & {N24.counts.sum():.0f} & {N24.counts.sum() / N24.live_time:.0f} & beam profile, $S_0$\\\\\n")
        fh.write(f"0 & Oct 6 & {T6.counts.sum():.0f} & {N6.counts.sum():.0f} & {N6.counts.sum() / N6.live_time:.0f} & target thickness\\\\\n")
        fh.write("\\hline\\hline\n\\end{tabular}\n")

    summary = {k_: v for k_, v in macros.items()}
    summary["per_angle"] = rows
    with open(C.TAB_DIR / "results_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    print("  wrote tables, macros and summary")


def _zscore(p):
    from scipy.stats import norm
    return float(norm.isf(p / 2))


def _pfmt(p):
    if p < 1e-3:
        e = int(np.floor(np.log10(max(p, 1e-300))))
        return f"{p / 10 ** e:.1f}\\times10^{{{e}}}"
    return f"{p:.4f}"


if __name__ == "__main__":
    with open(C.CACHE_DIR / "results.pkl", "rb") as fh:
        write_all(pickle.load(fh))
