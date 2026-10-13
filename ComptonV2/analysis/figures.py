"""Figures of the report (Physical Review column widths)."""
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

import beam
import calibration as cb
import config as C
import geometry as G
import nai_mc
import physics as ph
import response as R

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, MUTED = "#8a8985", "#0b0b0b", "#52514e"
SC, DC = 3.4, 7.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 7.2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "axes.linewidth": 0.7, "lines.linewidth": 1.2, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "axes.edgecolor": INK, "text.color": INK, "axes.labelcolor": INK,
})


def _save(fig, name):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    C.PAPER_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(C.FIG_DIR / f"{name}.{ext}")
    fig.savefig(C.PAPER_FIG_DIR / f"{name}.pdf")
    plt.close(fig)


def _label(ax, s, x=0.03, y=0.95):
    ax.text(x, y, s, transform=ax.transAxes, va="top", ha="left", fontsize=9)


def fig_setup(b):
    nom = b["nom"]
    sp = nom["setup"]
    fig, ax = plt.subplots(figsize=(SC, 2.35))
    L1, rd = sp.L1, sp.rd
    ax.add_patch(Rectangle((-9, -6), 9, 12, fc="#d9d8d4", ec=INK, lw=0.7))
    ax.add_patch(Rectangle((-3.2, -0.5), 3.2, 1.0, fc="white", ec=INK, lw=0.5))
    ax.add_patch(Circle((-3.2, 0), 0.45, fc=ORANGE, ec=INK, lw=0.5, zorder=5))
    ax.text(-4.5, -7.6, r"$^{137}$Cs + Pb collimator", ha="center", fontsize=7)
    psi0 = sp.psi0
    xe = L1 + rd + 4
    ax.add_patch(Polygon([[0, 0.5], [xe, xe * np.tan(psi0)], [xe, -xe * np.tan(psi0)], [0, -0.5]],
                         fc=BLUE, alpha=0.10, ec="none"))
    for sgn in (1, -1):
        ax.plot([0, xe], [sgn * 0.5, sgn * xe * np.tan(psi0)], color=BLUE, lw=0.6)
    ax.plot([-3.2, xe + 2], [0, 0], color=MUTED, lw=0.5, ls=(0, (4, 3)))
    R_, Ri = sp.rod_d / 2, sp.hole * sp.rod_d / 2
    ax.add_patch(Circle((L1, 0), R_ * 1.6, fc="#7d7c78", ec=INK, lw=0.6, zorder=6))
    ax.add_patch(Circle((L1, 0), Ri * 1.6, fc="white", ec=INK, lw=0.4, zorder=7))
    ax.text(L1, -4.6, "steel tube", ha="center", fontsize=7)
    th = np.radians(np.linspace(-15, 100, 200))
    ax.plot(L1 + rd * np.cos(th), rd * np.sin(th), color=GREY, lw=0.5, ls=(0, (1, 2)))
    for tdeg, alpha_ in ((60, 1.0), (0, 0.35)):
        t = np.radians(tdeg)
        cx, cy = L1 + (rd + 3.0) * np.cos(t), (rd + 3.0) * np.sin(t)
        w, h = 6.0, 2 * sp.a + 1.2
        rect = Rectangle((cx - w / 2, cy - h / 2), w, h, angle=tdeg, rotation_point="center",
                         fc="#c9c8c3", ec=INK, lw=0.6, alpha=alpha_)
        ax.add_patch(rect)
    t = np.radians(60)
    ax.annotate("", xy=(L1 + rd * np.cos(t), rd * np.sin(t)), xytext=(L1, 0),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.7, shrinkA=0, shrinkB=0))
    ax.add_patch(Wedge((L1, 0), 6.0, 0, 60, fill=False, ec=INK, lw=0.6))
    ax.text(L1 + 6.8 * np.cos(np.radians(30)), 6.8 * np.sin(np.radians(30)), r"$\theta$", fontsize=9)
    ax.text(L1 + 0.5 * rd * np.cos(t) - 3.2, 0.5 * rd * np.sin(t) + 0.6, rf"$r={rd:.0f}$ cm", fontsize=7,
            rotation=60)
    ax.text(L1 + (rd + 3.0) * np.cos(t) + 6.0, (rd + 3.0) * np.sin(t) - 1.0, "NaI(Tl)\n+ Pb sleeve",
            fontsize=7, ha="left", va="center")
    ax.annotate("", xy=(L1, -9.0), xytext=(0, -9.0),
                arrowprops=dict(arrowstyle="<->", color=INK, lw=0.6))
    ax.text(L1 / 2, -10.6, rf"$L_1={L1:.1f}$ cm", ha="center", fontsize=7)
    ax.text(L1 + 9.0, (L1 + 9.0) * np.tan(psi0) + 0.9, rf"$\psi_0={np.degrees(psi0):.1f}^\circ$", fontsize=7,
            color=BLUE, rotation=np.degrees(psi0))
    ax.set_xlim(-10, L1 + rd + 10)
    ax.set_ylim(-12, 27)
    ax.set_aspect("equal")
    ax.axis("off")
    _save(fig, "setup_geometry")


def fig_calibration(b):
    cal = b["calo"]["cal"]
    lines = b["calo"]["lines"]
    res = b["calo"]["res"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(SC, 3.6), sharex=True,
                                 gridspec_kw=dict(hspace=0.08, height_ratios=[1, 1]))
    Eg = np.linspace(60, 700, 300)
    chg = (Eg - cal["coef"][1]) / cal["coef"][0]
    _, dEg, _ = cb.energy(cal, chg)
    a1.fill_between(Eg, -dEg, dEg, color=BLUE, alpha=0.18, lw=0, label=r"calibration $\pm1\sigma$")
    a1.axhline(0, color=BLUE, lw=0.8)
    labels = {"Ba81": r"$^{133}$Ba", "Ba303": r"$^{133}$Ba", "Ba356": r"$^{133}$Ba",
              "Na511": r"$^{22}$Na", "Cs662": r"$^{137}$Cs"}
    for k in ("Ba81",) + cb.PRIMARY_LINES:
        L = lines[k]
        Efit, _, _ = cb.energy(cal, L["c"])
        dE = np.hypot(L["dc"], cb.ESTIMATOR_SYST_REL * L["c"]) * cal["coef"][0]
        used = k in cb.PRIMARY_LINES
        a1.errorbar(L["E"], L["E"] - Efit[0], yerr=dE, fmt="o" if used else "s", ms=4,
                    mfc=INK if used else "white", mec=INK, ecolor=INK, elinewidth=0.8, capsize=0)
        off = {"Ba303": (-22, 8), "Ba356": (5, 8)}.get(k, (4, 4))
        a1.annotate(labels[k], (L["E"], L["E"] - Efit[0]), xytext=off, textcoords="offset points",
                    fontsize=6.5, color=MUTED)
    a1.set_ylabel(r"$E_\gamma - E_{\rm fit}$ (keV)")
    a1.set_ylim(-3.2, 3.2)
    a1.legend(loc="lower right")
    _label(a1, "(a)")
    E = np.array([lines[k]["E"] for k in ("Ba81", "Ba356", "Na511", "Cs662")])
    s = np.array([lines[k]["s"] for k in ("Ba81", "Ba356", "Na511", "Cs662")]) * cal["coef"][0]
    ds = np.array([lines[k]["ds"] for k in ("Ba81", "Ba356", "Na511", "Cs662")]) * cal["coef"][0]
    a2.errorbar(E, 2.3548 * s / E * 100, yerr=2.3548 * ds / E * 100, fmt="o", ms=4, color=INK,
                elinewidth=0.8)
    a2.plot(Eg, 2.3548 * cb.sigma_E(res, Eg) / Eg * 100, color=BLUE, lw=1.0,
            label=r"$\sigma_E^2=a+bE+cE^2$")
    a2.set_ylabel("FWHM / $E$ (%)")
    a2.set_xlabel(r"$E_\gamma$ (keV)")
    a2.set_ylim(5, 16)
    a2.set_xlim(50, 700)
    a2.legend(loc="upper right")
    _label(a2, "(b)")
    _save(fig, "energy_calibration_residuals")


def _spec_E(spec, cal, rebin):
    import peakfit as pf
    x = pf.rebin_x(rebin, spec.counts.size)
    y = pf.rebin(spec.counts, rebin)
    return cal["coef"][1] + cal["coef"][0] * x, y


def fig_spectra(b):
    from mca_io import read_mca
    cal = b["calo"]["cal"]
    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.35), gridspec_kw=dict(wspace=0.32))
    s0 = read_mca(C.DATA_DIR / C.ZERO_NO_TARGET)
    E, y = _spec_E(s0, cal, 8)
    a = axs[0]
    a.step(E, y / s0.live_time, where="mid", color=INK, lw=0.7)
    a.set_yscale("log")
    a.set_xlim(0, 760)
    a.set_ylim(0.03, 300)
    feats = [(661.7, "photopeak\n661.7 keV", (-7, 2), "right"), (477.3, "Compton\nedge", (0, 16), "center"),
             (184.3, "backscatter", (0, 14), "center"), (75.0, "Pb K x rays", (6, 34), "left"),
             (32.2, "Ba K x rays", (2, -26), "left")]
    for e, t, off, ha in feats:
        i = np.argmin(np.abs(E - e))
        a.annotate(t, (e, y[i] / s0.live_time), xytext=off, textcoords="offset points",
                   ha=ha, fontsize=6, color=MUTED,
                   arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
    a.set_xlabel("Energy (keV)")
    a.set_ylabel(r"Rate (s$^{-1}$ per 2 keV)")
    _label(a, r"(a) $0^\circ$, direct beam", y=0.97)
    from scatter_fit import load_pair
    T, N, _ = load_pair(60)
    Et, yt = _spec_E(T, cal, 16)
    _, yn = _spec_E(N, cal, 16)
    a = axs[1]
    a.step(Et, yt, where="mid", color=INK, lw=0.7, label="target in")
    a.step(Et, yn, where="mid", color=AQUA, lw=0.7, ls=(0, (3, 1.5)), label="target out")
    a.set_yscale("log")
    a.set_xlim(0, 760)
    a.set_ylim(5, 3e3)
    a.axvline(ph.compton_energy(60), color=BLUE, lw=0.6, ls=":")
    a.set_xlabel("Energy (keV)")
    a.set_ylabel("Counts / 4 keV / 300 s")
    a.legend(loc="upper right")
    _label(a, r"(b) $\theta=60^\circ$", y=0.97)
    a = axs[2]
    d = b["nom"]["angles"][60]
    _net_panel(a, d, cal, rebin_disp=16, show_comp=True)
    a.set_xlim(150, 700)
    a.set_xlabel("Energy (keV)")
    a.set_ylabel("Net counts / 4 keV")
    _label(a, r"(c) $\theta=60^\circ$, in $-$ out", y=0.97)
    a.legend(loc="upper right", fontsize=6.0, bbox_to_anchor=(1.0, 0.88))
    _save(fig, "spectra_examples")


def _net_panel(a, d, cal, rebin_disp=16, show_comp=True, xmin=120, xmax=720):
    bfac = rebin_disp // 8
    x, net, var = d["x"], d["net"], d["var"]
    m = (len(x) // bfac) * bfac
    xd = x[:m].reshape(-1, bfac).mean(1)
    nd = net[:m].reshape(-1, bfac).sum(1)
    vd = var[:m].reshape(-1, bfac).sum(1)
    Ed = cal["coef"][1] + cal["coef"][0] * xd
    sel = (Ed > xmin) & (Ed < xmax)
    a.errorbar(Ed[sel], nd[sel], np.sqrt(vd[sel]), fmt="o", ms=1.8, color=INK, elinewidth=0.5, lw=0)
    p = dict(zip(d["fit_names"], d["fit_p"]))
    tm = d["tmpl"]
    b_, a_ = cal["coef"][0], cal["coef"][1]
    Ef = np.linspace(d["window"][0], d["window"][1], 300)
    per_bin = b_ * rebin_disp
    tot = np.interp(Ef - p["delta"], R.E_CENT, (tm["pk"] + tm["ct"] + tm["ms"]) / R.DE) * per_bin
    xf = (Ef - a_) / b_
    x0 = (tm["E_pk"] - a_) / b_
    bg = (p["c0"] + p["c1"] * (xf - x0) / 100.0 + (p.get("c2", 0) * ((xf - x0) / 100.0) ** 2)) * bfac
    a.plot(Ef, p["alpha"] * tot + bg, color=BLUE, lw=1.1, label="fit (KN template)")
    if show_comp:
        pk = np.interp(Ef - p["delta"], R.E_CENT, tm["pk"] / R.DE) * per_bin
        ms = np.interp(Ef - p["delta"], R.E_CENT, (tm["ms"] + tm["ct"]) / R.DE) * per_bin
        a.plot(Ef, p["alpha"] * pk, color=AQUA, lw=0.9, ls=(0, (4, 1.5)), label="single scatter")
        a.plot(Ef, p["alpha"] * ms + bg, color=GREY, lw=0.9, ls=(0, (1, 1)), label="multiple + bkg")
    a.axhline(0, color=GREY, lw=0.4)


def fig_netfits(b):
    cal = b["calo"]["cal"]
    angles = b["angles"]
    fig, axs = plt.subplots(2, 4, figsize=(DC, 3.4), gridspec_kw=dict(wspace=0.28, hspace=0.32))
    for a, th in zip(axs.ravel(), angles):
        d = b["nom"]["angles"][th]
        _net_panel(a, d, cal, rebin_disp=16, show_comp=True)
        lo = max(120, d["E_pk"] - 260)
        a.set_xlim(lo, min(720, d["E_pk"] + 220))
        flag = th in C.FLAGGED
        _label(a, rf"${th}^\circ$" + (r"$^{*}$" if flag else ""), x=0.04, y=0.95)
        a.text(0.97, 0.95, rf"$\alpha={d['alpha']:.2f}\pm{d['dalpha']:.2f}$", transform=a.transAxes,
               ha="right", va="top", fontsize=6.0,
               bbox=dict(fc="white", ec="none", pad=0.6, alpha=0.85))
        ylo, yhi = a.get_ylim()
        a.set_ylim(ylo, yhi + 0.25 * (yhi - ylo))
        a.tick_params(labelsize=7)
    for a in axs[1]:
        a.set_xlabel("Energy (keV)", fontsize=8)
    for a in axs[:, 0]:
        a.set_ylabel("Net counts / 4 keV", fontsize=8)
    h, l = axs[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, 1.02))
    _save(fig, "net_spectra_fits")


def fig_beam(b):
    nom = b["nom"]
    prof = nom["profile"]
    rows = b["ctx_scan"]
    th = np.array([r["theta"] for r in rows])
    y = np.array([r["rate"] for r in rows])
    dy = np.array([r["drate"] for r in rows])
    sp = prof["setup"]
    eff = nom["eff_dir"]
    tg = np.linspace(0, 92, 93)
    S0, leak = prof["p"][0], prof["p"][-1]
    g = np.array([G.direct_acceptance(t, sp, 2000) for t in tg])
    p_core = beam.fit_profile(rows, sp.with_(halo=0.0), eff, halo=False, verbose=False)
    gc = np.array([G.direct_acceptance(t, p_core["setup"], 2000) for t in tg])
    fig, a = plt.subplots(figsize=(SC, 2.6))
    a.errorbar(th, y, dy, fmt="o", ms=3.5, color=INK, elinewidth=0.8, label="target out, 662-keV peak")
    a.plot(tg, S0 * eff * g + leak, color=BLUE, label=rf"core + halo ($\chi^2$/ndf = {prof['chi2']:.1f}/{prof['ndf']})")
    a.plot(tg, p_core["p"][0] * eff * gc + p_core["p"][-1], color=ORANGE, ls=(0, (4, 2)),
           label=rf"core only ($\chi^2$/ndf = {p_core['chi2']:.0f}/{p_core['ndf']})")
    a.set_yscale("log")
    a.set_ylim(0.05, 3e3)
    a.set_xlim(-3, 93)
    a.set_xlabel(r"Detector angle $\theta$ (deg)")
    a.set_ylabel(r"Photopeak rate (s$^{-1}$)")
    a.legend(loc="upper right", fontsize=6.4)
    ins = a.inset_axes([0.10, 0.08, 0.30, 0.30])
    psi = np.radians(np.linspace(0, 14, 200))
    ins.plot(np.degrees(psi), G.beam_f(psi, sp), color=BLUE, lw=1.0)
    ins.plot(np.degrees(psi), G.beam_f(psi, p_core["setup"]), color=ORANGE, lw=0.9, ls=(0, (4, 2)))
    ins.set_xlabel(r"$\psi$ (deg)", fontsize=6.5, labelpad=1)
    ins.set_ylabel(r"$f(\psi)$", fontsize=6.5, labelpad=1)
    ins.tick_params(labelsize=6)
    _save(fig, "beam_profile")


def fig_efficiency(b):
    tabs = b["tabs"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(SC, 3.8), gridspec_kw=dict(hspace=0.35))
    t = tabs["nominal"]
    a1.plot(t["E"], t["eps_tot"], color=INK, lw=1.1, label=r"$\varepsilon_{\rm tot}$")
    a1.plot(t["E"], t["eps_fep"], color=BLUE, lw=1.1, label=r"$\varepsilon_{\rm FEP}$")
    a1.plot(t["E"], t["PT"], color=AQUA, lw=1.1, ls=(0, (5, 1.5, 1, 1.5)), label="P/T")
    for k, ls in (("alt0", (0, (2, 1.5))), ("alt1", (0, (1, 1)))):
        tt = tabs[k]
        a1.plot(tt["E"], tt["eps_fep"], color=BLUE, lw=0.6, ls=ls)
    tt0, tt1 = tabs["alt0"], tabs["alt1"]
    a1.text(585, np.interp(585, tt1["E"], tt1["eps_fep"]) + 0.02, r"$3''$ long", color=BLUE, fontsize=6.0)
    a1.text(585, np.interp(585, tt0["E"], tt0["eps_fep"]) - 0.055, r"$1.5''$ long", color=BLUE, fontsize=6.0)
    for th in C.XS_ANGLES:
        a1.axvline(ph.compton_energy(th), color=GREY, lw=0.3, ls=":")
    a1.set_xlim(150, 700)
    a1.set_ylim(0.15, 1.0)
    a1.set_xlabel("Photon energy (keV)")
    a1.set_ylabel("Intrinsic efficiency")
    a1.legend(loc="upper right", ncol=3, fontsize=6.5)
    _label(a1, r"(a) $2''\times2''$ NaI(Tl), $r=20$ cm", y=0.12)
    v = tabs["val3x3"]
    bic = nai_mc.BICRON_3X3
    e_b = bic[:, 0]
    fep_b = bic[:, 1] * bic[:, 2]
    fep_mc = np.interp(e_b, v["E"], v["eps_fep"])
    i0 = np.argmax(e_b)
    a2.plot(e_b, fep_b / fep_b[i0], "s", ms=4.5, mfc="white", mec=INK, label="Bicron (M&N Table 9.1)")
    a2.plot(e_b, fep_mc / fep_mc[i0], "o", ms=3.2, color=BLUE, label="this MC")
    a2.set_xlabel("Photon energy (keV)")
    a2.set_ylabel(r"$\varepsilon_{\rm FEP}(E)/\varepsilon_{\rm FEP}(614)$")
    a2.legend(loc="upper right", fontsize=6.5)
    _label(a2, r"(b) $3''\times3''$ validation", y=0.14)
    _save(fig, "detector_efficiency")


def fig_target(b):
    sh = b["shadow"]
    sc = b["scan_od"]
    fig, axs = plt.subplots(1, 3, figsize=(DC, 2.2), gridspec_kw=dict(wspace=0.38))
    ods = sh["ods"]
    a = axs[0]
    f, df = sh["removal_meas"]
    a.axhspan(f - df, f + df, color=ORANGE, alpha=0.18, lw=0)
    a.axhline(f, color=ORANGE, lw=0.8, ls=(0, (4, 2)), label="measured (Oct 6 pair)")
    a.plot(ods, sh["removal0"], color=INK, lw=1.1, label="prediction")
    a.set_xlabel("Tube OD (cm)")
    a.set_ylabel(r"$0^\circ$ beam removal")
    a.axvline(b["od_trans"], color=ORANGE, lw=0.6, ls=":")
    a.legend(loc="upper left", fontsize=6.3)
    _label(a, "(a)", x=0.86, y=0.15)
    a = axs[1]
    m, dm = sh["meas"]
    a.axhspan(100 * (m - dm), 100 * (m + dm), color=AQUA, alpha=0.25, lw=0)
    a.axhline(100 * m, color=AQUA, lw=0.9, label="measured")
    y0s = 10 * sh["y0s"]
    for dth, ls in ((-1, (0, (1, 1))), (0, "-"), (1, (0, (4, 2)))):
        a.plot(y0s, 100 * sh["pred"][dth], color=INK, lw=0.9, ls=ls,
               label=rf"$\Delta\theta={dth:+d}^\circ$" if dth else r"$\Delta\theta=0$")
    a.axvspan(10 * (C.ROD_Y0[0] - C.ROD_Y0[1]), 10 * (C.ROD_Y0[0] + C.ROD_Y0[1]), color=GREY, alpha=0.2, lw=0)
    a.set_xlabel(rf"Tube offset $y_0$ (mm), OD = {sh['od_nominal']:.2f} cm")
    a.set_ylabel(r"$10^\circ$: (in$-$out)/out (%)")
    a.set_ylim(-14, 3)
    a.legend(loc="lower right", fontsize=6.0)
    _label(a, "(b)", x=0.05, y=0.62)
    a = axs[2]
    a.errorbar(sc[:, 0], sc[:, 1], sc[:, 2], fmt="o-", ms=3, color=INK, lw=0.8, elinewidth=0.7)
    a.axhline(1.0, color=BLUE, lw=0.9, label="Klein-Nishina")
    od0, dod = C.ROD_OD
    a.axvspan(od0 - dod, od0 + dod, color=GREY, alpha=0.2, lw=0, label="photo estimate")
    a.axvline(b["od_trans"], color=ORANGE, lw=0.8, ls=(0, (4, 2)), label=r"from $0^\circ$ removal")
    a.set_xlim(0.8, max(2.35, b["od_trans"] + 0.1))
    a.set_xlabel("Tube OD (cm)")
    a.set_ylabel(r"$\langle\alpha\rangle$ = data / KN")
    a.legend(loc="upper right", fontsize=6.0)
    _label(a, "(c)", x=0.05, y=0.15)
    _save(fig, "target_thickness")


def fig_kinematics(b):
    nom = b["nom"]
    angles = b["angles"]
    kres = b["kin"]["template"]
    fit_angles = b["fit_angles"]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(SC, 3.9), sharex=True,
                                 gridspec_kw=dict(hspace=0.06, height_ratios=[2.0, 1]))
    tg = np.linspace(0, 100, 300)
    a1.plot(tg, ph.compton_energy(tg), color=BLUE, label="Compton, Eq. (1)")
    a1.axhline(ph.E_CS137, color=ORANGE, ls=(0, (4, 2)), label="classical (no shift)")
    Vt = kres["Vtot"]
    dE_fit = dict(zip(fit_angles, np.sqrt(np.diag(Vt))))
    for th in angles:
        d = nom["angles"][th]
        flag = th in C.FLAGGED
        err = dE_fit.get(th, np.hypot(d["dE_stat"], d.get("E_est", 0)))
        kw = dict(fmt="o", ms=3.8, elinewidth=0.8, capsize=0)
        if flag:
            kw.update(mfc="white", mec=GREY, ecolor=GREY)
        else:
            kw.update(color=INK)
        a1.errorbar(th, d["E"], err, **kw)
        a2.errorbar(th, d["E"] - d["E_pk"], err, **kw)
        g = b["gauss"][th]
        a2.errorbar(th + 1.6, g["E"] - d["E_pk"], np.hypot(g["dE_stat"], g["E_syst_est"]), fmt="D", ms=2.8,
                    mfc="white", mec=AQUA, ecolor=AQUA, elinewidth=0.6)
    a1.plot([], [], "o", color=INK, ms=3.8, label="template fit")
    a1.plot([], [], "o", mfc="white", mec=GREY, ms=3.8, label="flagged run")
    a1.set_ylabel(r"$E'$ (keV)")
    a1.set_ylim(250, 700)
    a1.legend(loc="lower left", fontsize=6.6)
    f1 = kres["fix"]
    a1.text(0.97, 0.84, rf"$m_ec^2 = {f1['m']:.0f}\pm{f1['dm_stat']:.0f}\pm{f1['dm_syst']:.0f}$ keV",
            transform=a1.transAxes, ha="right", va="top", fontsize=7.5)
    _label(a1, "(a)", x=0.03, y=0.84)
    a2.axhline(0, color=BLUE, lw=0.9)
    a2.plot([], [], "D", mfc="white", mec=AQUA, ms=2.8, label="Gaussian centroid")
    a2.legend(loc="lower right", fontsize=6.3)
    a2.set_ylabel(r"$E' - E'_{\rm KN}$ (keV)")
    a2.set_xlabel(r"Scattering angle $\theta$ (deg)")
    a2.set_ylim(-45, 30)
    a2.set_xlim(0, 100)
    _label(a2, "(b)", x=0.03, y=0.93)
    _save(fig, "compton_kinematics")

    fig, a = plt.subplots(figsize=(SC, 2.5))
    x = kres["fix"]["x"]
    E = kres["fix"]["E"]
    dE = np.sqrt(np.diag(Vt))
    lin = b["lin_t"]
    a.errorbar(x, 1e3 / E, 1e3 * dE / E ** 2, fmt="o", ms=3.8, color=INK, elinewidth=0.8,
               label="data (template fit)")
    xx = np.linspace(0, 1.1, 50)
    a.plot(xx, 1e3 * (lin["c"][0] + lin["c"][1] * xx), color=BLUE,
           label=rf"fit: $m_ec^2={lin['m']:.0f}\pm{lin['dm']:.0f}$ keV")
    a.plot(xx, 1e3 * (1 / ph.E_CS137 + xx / ph.ME_C2), color=AQUA, ls=(0, (4, 2)), lw=0.9,
           label=r"$m_ec^2 = 511$ keV")
    a.set_xlabel(r"$\langle 1-\cos\theta\rangle$")
    a.set_ylabel(r"$1/E'$ (MeV$^{-1}$)")
    a.legend(loc="upper left", fontsize=6.6)
    _save(fig, "inverse_energy_fit")


def fig_xsec(b):
    nom = b["nom"]
    res, resT = b["results"][b["nominal_mode"]], b["results"][b["alt_mode"]]
    angles = b["angles"]
    fit_idx = [angles.index(a) for a in b["fit_angles"]]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(SC, 4.3), sharex=True,
                                 gridspec_kw=dict(hspace=0.06, height_ratios=[1.7, 1]))
    tg = np.linspace(0, 180, 361)
    kn = ph.klein_nishina_dsdo(tg) / 1e-27
    thom = ph.thomson_dsdo(tg) / 1e-27
    a1.plot(tg, kn, color=BLUE, label="Klein-Nishina")
    a1.plot(tg, thom, color=ORANGE, ls=(0, (4, 2)), label="Thomson")
    alpha = res["alpha"]
    V_unc, V_cor = res["V_unc"], res["V_cor"]
    e_unc = np.sqrt(np.diag(V_unc))
    e_tot = np.sqrt(np.diag(V_unc + V_cor))
    knp = ph.klein_nishina_dsdo(np.array(angles)) / 1e-27
    for i, th in enumerate(angles):
        flag = th in C.FLAGGED
        kw = dict(ms=4, elinewidth=0.9, capsize=0)
        if flag:
            kw.update(fmt="o", mfc="white", mec=GREY, ecolor=GREY)
        else:
            kw.update(fmt="o", color=INK)
        a1.errorbar(th, alpha[i] * knp[i], e_tot[i] * knp[i], alpha=0.45 if not flag else 0.35,
                    **{**kw, "elinewidth": 0.6})
        a1.errorbar(th, alpha[i] * knp[i], e_unc[i] * knp[i], **kw)
        a2.errorbar(th, alpha[i], e_tot[i], alpha=0.45 if not flag else 0.35, **{**kw, "elinewidth": 0.6})
        a2.errorbar(th, alpha[i], e_unc[i], **kw)
    aT = resT["alpha"]
    eT = np.sqrt(np.diag(resT["V_unc"]))
    a2.errorbar(np.array(angles) + 2.0, aT, eT, fmt="s", ms=3, mfc="white", mec=AQUA, ecolor=AQUA,
                elinewidth=0.6, label=rf"tube OD {b['alt']['od']:.2f} cm "
                + ("(photograph)" if b["alt_mode"] == "geometric" else r"($0^\circ$ removal)"))
    a1.plot([], [], "o", color=INK, ms=4, label="this work")
    a1.plot([], [], "o", mfc="white", mec=GREY, ms=4, label="flagged run")
    a1.set_ylabel(r"$d\sigma/d\Omega$ ($10^{-27}$ cm$^2$/sr)")
    a1.set_ylim(0, 95)
    a1.set_xlim(0, 180)
    a1.legend(loc="upper right", fontsize=6.5)
    _label(a1, "(a)", y=0.12)
    thk = np.interp(tg, angles, res["Th_over_KN"])
    a2.plot(tg[(tg >= 20) & (tg <= 90)], thk[(tg >= 20) & (tg <= 90)], color=ORANGE, ls=(0, (4, 2)),
            label="Thomson / KN (folded)")
    a2.axhline(1, color=BLUE)
    m, dm = res["mean"]["mean"], np.hypot(res["mean"]["stat"], res["mean"]["syst"])
    a2.axhspan(m - dm, m + dm, color=GREY, alpha=0.2, lw=0, label=r"$\langle\alpha\rangle\pm1\sigma$")
    a2.set_ylabel("Data / KN")
    a2.set_xlabel(r"Scattering angle $\theta$ (deg)")
    a2.set_ylim(0, 3.4)
    a2.legend(loc="upper right", fontsize=6.2, ncol=1)
    _label(a2, "(b)", y=0.93)
    _save(fig, "differential_cross_section")


CAL_SOURCES = [("Cs-137", "cs137", ["Cs662"], BLUE, "o"),
               ("Ba-133", "ba133", ["Ba81", "Ba303", "Ba356"], ORANGE, "s"),
               ("Na-22", "na22", ["Na511"], AQUA, "D")]


def fig_calibration_curve(b):
    import earlier_calibration as cc
    cal = b["calo"]["cal"]
    lines = b["calo"]["lines"]
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(SC, 3.7), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2.2, 1], hspace=0.06))
    xx = np.linspace(0, 4095, 200)
    ax.plot(xx, cal["coef"][1] + cal["coef"][0] * xx, color=INK, lw=0.9,
            label=rf"$E = {cal['coef'][1]:.2f} + {cal['coef'][0]:.5f}\,{{\rm ch}}$")
    for name, _, keys, col, mk in CAL_SOURCES:
        for i, k in enumerate(keys):
            L = lines[k]
            used = k in cb.PRIMARY_LINES
            dch = np.hypot(L["dc"], cb.ESTIMATOR_SYST_REL * L["c"])
            Efit = cb.energy(cal, L["c"])[0][0]
            kw = dict(marker=mk, ms=4.5, ls="none", mec=col, mfc=col if used else "white", color=col)
            ax.plot(L["c"], L["E"], label=name if i == len(keys) - 1 else None, **kw)
            axr.errorbar(L["c"], L["E"] - Efit, yerr=dch * cal["coef"][0], elinewidth=0.8, capsize=0, **kw)
            ax.annotate(f"{L['E']:.0f}", (L["c"], L["E"]), xytext=(4, -9), textcoords="offset points",
                        fontsize=6.3, color=MUTED)
    ax.plot([], [], "o", mfc="white", mec=GREY, ms=4.5, label="not in fit")
    r = cc.run(include_81=False)
    axr.plot(r["ch"] - 1, r["resid"], "x", color=GREY, ms=5, mew=1.0,
             label="earlier calibration (Sept 29 runs, incl. 14.5 kcps Cs-137)")
    axr.axhline(0, color=INK, lw=0.7)
    ax.set_ylabel("Energy (keV)")
    ax.set_xlim(0, 3000)
    ax.set_ylim(0, 720)
    ax.legend(loc="upper left", fontsize=6.6)
    axr.set_ylabel(r"$E_\gamma-E_{\rm fit}$ (keV)")
    axr.set_xlabel("Channel")
    axr.set_ylim(-3, 8)
    axr.legend(loc="upper left", fontsize=6.0)
    _label(ax, "(a)", x=0.92, y=0.12)
    _label(axr, "(b)", x=0.92, y=0.95)
    _save(fig, "energy_calibration_curve")


def fig_calibration_spectra(b):
    from mca_io import read_mca
    cal = b["calo"]["cal"]
    bkg = read_mca(C.DATA_DIR / "calibration/bkg_withgain_850.mca")
    marks = {"Cs-137": [(32.2, "Ba K x rays"), (184.3, "backscatter"), (477.3, "Compton edge"),
                        (661.657, "661.7")],
             "Ba-133": [(31.6, "Cs K x rays"), (80.9, "81.0"), (302.9, "302.9"), (356.0, "356.0"),
                        (383.8, "383.8")],
             "Na-22": [(511.0, "511.0"), (340.7, "Compton edge")]}
    fig, axs = plt.subplots(3, 1, figsize=(SC, 5.6), sharex=True, gridspec_kw=dict(hspace=0.08))
    E = cal["coef"][1] + cal["coef"][0] * np.arange(4096)
    reb = 4
    for a, (name, key, _, col, _) in zip(axs, CAL_SOURCES):
        sp = read_mca(C.DATA_DIR / C.CALIB_RUNS["A"][key])
        gross = sp.counts
        net = gross - sp.live_time / bkg.live_time * bkg.counts
        Eb = E[: 4096 // reb * reb].reshape(-1, reb).mean(1)
        g = gross[: 4096 // reb * reb].reshape(-1, reb).sum(1)
        n = net[: 4096 // reb * reb].reshape(-1, reb).sum(1)
        a.step(Eb, g / 1e3, where="mid", color=GREY, lw=0.5, label="raw")
        a.step(Eb, n / 1e3, where="mid", color=col, lw=0.8, label="background subtracted")
        a.axhline(0, color=GREY, lw=0.4)
        ymax = 1.45 * np.max(n[(Eb > 20)]) / 1e3
        for e, t in marks[name]:
            a.axvline(e, color=MUTED, lw=0.5, ls=(0, (3, 2)))
            a.text(e + 4, 0.97 * ymax, t, fontsize=5.8, color=MUTED, rotation=90, va="top")
        a.set_ylim(-0.04 * ymax, ymax)
        a.set_ylabel(r"Counts / 1 keV ($10^3$)")
        tx = {"Cs-137": 0.33, "Ba-133": 0.70, "Na-22": 0.76}[name]
        a.text(tx, 0.95, name, transform=a.transAxes, ha="left", va="top", fontsize=8.5)
        a.text(tx, 0.83, f"{sp.start_time[:10]}\n{sp.live_time:.0f} s live", transform=a.transAxes,
               ha="left", va="top", fontsize=6.3, color=MUTED)
    axs[0].legend(loc="center left", bbox_to_anchor=(0.30, 0.45), fontsize=6.3)
    axs[1].text(0.60, 0.50, "low-energy peaks offset:\nNaI non-proportionality;\ncalibration valid 280-662 keV",
                transform=axs[1].transAxes, fontsize=5.8, color=MUTED, va="top")
    axs[-1].set_xlabel("Energy (keV)")
    axs[-1].set_xlim(0, 760)
    _save(fig, "calibration_source_spectra")


def fig_theory(b):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(DC, 2.5), gridspec_kw=dict(wspace=0.28))
    th = np.linspace(0, 180, 361)
    Ep = ph.compton_energy(th)
    a1.plot(th, Ep, color=BLUE, label=r"scattered photon $E'$")
    a1.plot(th, ph.E_CS137 - Ep, color=AQUA, ls=(0, (5, 1.5, 1, 1.5)), label=r"recoil electron $T_e=E-E'$")
    a1.axhline(ph.E_CS137, color=ORANGE, ls=(0, (4, 2)), label="classical (no shift)")
    for t in C.XS_ANGLES:
        a1.plot(t, ph.compton_energy(t), "o", ms=3, color=BLUE)
    a1.annotate(r"backscatter $E'_{\min}=184$ keV", (180, ph.compton_energy(180)), xytext=(-6, -16),
                textcoords="offset points", ha="right", fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
    a1.annotate(r"Compton edge $T_e^{\max}=477$ keV", (180, ph.E_CS137 - ph.compton_energy(180)),
                xytext=(-6, -18), textcoords="offset points", ha="right", fontsize=6.5, color=MUTED,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
    a1.set_xlim(0, 180)
    a1.set_ylim(0, 780)
    a1.set_xlabel(r"Scattering angle $\theta$ (deg)")
    a1.set_ylabel("Energy (keV)")
    a1.legend(loc="upper right", fontsize=6.4, bbox_to_anchor=(1.0, 0.82))
    _label(a1, "(a)", y=0.96, x=0.85)
    r2 = ph.R_E ** 2 / 1e-27
    a2.plot(th, ph.thomson_dsdo(th) / 1e-27, color=ORANGE, ls=(0, (4, 2)), label="Thomson (classical)")
    a2.plot(th, ph.klein_nishina_dsdo(th, 30.0) / 1e-27, color=GREY, ls=(0, (1, 1)), lw=1.0,
            label="Klein-Nishina, 30 keV")
    a2.plot(th, ph.klein_nishina_dsdo(th) / 1e-27, color=BLUE, label="Klein-Nishina, 662 keV")
    a2.plot(th, ph.klein_nishina_dsdo(th, 2000.0) / 1e-27, color=AQUA, ls=(0, (5, 1.5, 1, 1.5)),
            label="Klein-Nishina, 2 MeV")
    a2.axvspan(min(C.FIT_ANGLES), max(C.FIT_ANGLES), color=GREY, alpha=0.12, lw=0)
    a2.text(0.5 * (min(C.FIT_ANGLES) + max(C.FIT_ANGLES)), 3, "measured", ha="center", fontsize=6.5,
            color=MUTED)
    a2.set_xlim(0, 180)
    a2.set_ylim(0, 1.05 * r2)
    a2.set_xlabel(r"Scattering angle $\theta$ (deg)")
    a2.set_ylabel(r"$d\sigma/d\Omega$ ($10^{-27}$ cm$^2$/sr)")
    a2.legend(loc="upper right", fontsize=6.2)
    _label(a2, "(b)", y=0.10, x=0.03)
    _save(fig, "theory_predictions")


def make_all(b):
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    for f in (fig_theory, fig_setup, fig_calibration, fig_spectra, fig_netfits, fig_beam, fig_efficiency,
              fig_target, fig_kinematics, fig_xsec, fig_calibration_curve, fig_calibration_spectra):
        try:
            f(b)
            print("  figure:", f.__name__)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print("  FAILED:", f.__name__, exc)


if __name__ == "__main__":
    with open(C.CACHE_DIR / "results.pkl", "rb") as fh:
        make_all(pickle.load(fh))
