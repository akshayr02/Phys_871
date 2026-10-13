"""Source, beam, target and detector geometry; beam transmission and single-scattering Monte Carlo."""
from dataclasses import dataclass, replace

import numpy as np

import config as C
import physics as ph


@dataclass(frozen=True)
class Setup:
    L1: float = C.SOURCE_TO_TARGET[0]
    L2: float = C.SOURCE_TO_DETECTOR[0]
    a: float = C.APERTURE_RADIUS[0]
    rod_d: float = C.ROD_DIAMETER_PRIOR[0]
    hole: float = C.ROD_ID_OVER_OD[0]
    y0: float = 0.0
    material: str = C.TARGET_MATERIAL
    density: float = None
    psi0: float = np.radians(5.0)
    w: float = np.radians(0.8)
    halo: float = 0.0
    lam: float = np.radians(3.0)
    dtheta: float = 0.0

    @property
    def rd(self):
        return self.L2 - self.L1

    @property
    def rho(self):
        return self.density or ph.TARGET_MATERIALS[self.material]["density"]

    @property
    def n_e(self):
        return ph.electron_density(self.material, self.rho)

    @property
    def area(self):
        return np.pi / 4 * self.rod_d ** 2 * (1 - self.hole ** 2)

    def mu(self, E, kind="total"):
        return ph.mu_linear(ph.TARGET_MATERIALS[self.material]["xsec"], E, self.rho, kind)

    def with_(self, **kw):
        return replace(self, **kw)


def beam_f(psi, s: Setup):
    core = 1.0 / (1.0 + np.exp((psi - s.psi0) / s.w))
    if s.halo == 0.0:
        return core
    return core + s.halo / (1.0 + np.exp((psi - s.psi0) / s.lam))


def aperture_points(theta_deg, s: Setup, n=4000, rng=None):
    th = np.radians(theta_deg + s.dtheta)
    if rng is None:
        k = np.arange(n) + 0.5
        r = s.a * np.sqrt(k / n)
        phi = k * np.pi * (3 - np.sqrt(5))
    else:
        r = s.a * np.sqrt(rng.random(n))
        phi = 2 * np.pi * rng.random(n)
    u1 = np.array([-np.sin(th), np.cos(th), 0.0])
    u2 = np.array([0.0, 0.0, 1.0])
    centre = np.array([s.L1 + s.rd * np.cos(th), s.rd * np.sin(th), 0.0])
    P = centre + (r * np.cos(phi))[:, None] * u1 + (r * np.sin(phi))[:, None] * u2
    normal = -np.array([np.cos(th), np.sin(th), 0.0])
    return P, normal, np.pi * s.a ** 2


def direct_acceptance(theta_deg, s: Setup, n=4000):
    P, nrm, area = aperture_points(theta_deg, s, n)
    D = np.linalg.norm(P, axis=1)
    psi = np.arccos(np.clip(P[:, 0] / D, -1, 1))
    cosa = np.abs(P @ nrm) / D
    return area * np.mean(beam_f(psi, s) * cosa / D ** 2)


def _cyl_roots(p, k, s: Setup, radius=None):
    radius = s.rod_d / 2 if radius is None else radius
    px, py = p[:, 0] - s.L1, p[:, 1] - s.y0
    a = k[:, 0] ** 2 + k[:, 1] ** 2
    b = 2 * (px * k[:, 0] + py * k[:, 1])
    c = px ** 2 + py ** 2 - radius ** 2
    disc = b ** 2 - 4 * a * c
    ok = (disc > 0) & (a > 1e-14)
    sq = np.sqrt(np.where(ok, disc, 0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(ok, (-b - sq) / (2 * a), 0)
        t2 = np.where(ok, (-b + sq) / (2 * a), 0)
    return t1, t2, ok


def exit_path(P, k, s: Setup):
    t1, t2, ok = _cyl_roots(P, k, s)
    L = np.where(ok, np.maximum(t2, 0), 0)
    a = b = L
    if s.hole > 0:
        i1, i2, oki = _cyl_roots(P, k, s, radius=s.hole * s.rod_d / 2)
        cross = oki & (i1 > 0)
        a = np.where(cross, i1, L)
        b = np.where(cross, np.minimum(i2, L), L)
    return L, a, b


def material_exit(P, k, s: Setup):
    L, a, b = exit_path(P, k, s)
    return L - (b - a)


def advance(P, k, s_mat, s: Setup):
    L, a, b = exit_path(P, k, s)
    geo = np.where(s_mat < a, s_mat, s_mat + (b - a))
    return P + k * geo[:, None]


def sample_wall(n, s: Setup, rng):
    R, Ri = s.rod_d / 2, s.hole * s.rod_d / 2
    zmax = s.L1 * np.tan(min(s.psi0 + 8 * max(s.w, s.lam if s.halo else 0), np.radians(40)))
    rr = np.sqrt(Ri ** 2 + (R ** 2 - Ri ** 2) * rng.random(n))
    pp = 2 * np.pi * rng.random(n)
    P = np.stack([s.L1 + rr * np.cos(pp), s.y0 + rr * np.sin(pp), (2 * rng.random(n) - 1) * zmax], 1)
    return P, np.pi * (R ** 2 - Ri ** 2) * 2 * zmax


def chord_from_source(Q, s: Setup):
    D = np.linalg.norm(Q, axis=1)
    k = Q / D[:, None]
    t1, t2, ok = _cyl_roots(np.zeros_like(Q), k, s)
    t1, t2 = np.clip(t1, 0, D), np.clip(t2, 0, D)
    chord = np.where(ok, t2 - t1, 0.0)
    if s.hole > 0:
        i1, i2, oki = _cyl_roots(np.zeros_like(Q), k, s, radius=s.hole * s.rod_d / 2)
        i1, i2 = np.clip(i1, 0, D), np.clip(i2, 0, D)
        chord = chord - np.where(oki, i2 - i1, 0.0)
    return chord


def transmission_0deg(s: Setup, n=6000, addback=0.0, theta_deg=0.0):
    P, nrm, area = aperture_points(theta_deg, s, n)
    D = np.linalg.norm(P, axis=1)
    psi = np.arccos(np.clip(P[:, 0] / D, -1, 1))
    wgt = beam_f(psi, s) / D ** 2 * np.abs(P @ nrm) / D
    ell = chord_from_source(P, s)
    T = np.sum(wgt * np.exp(-s.mu(ph.E_CS137) * ell)) / np.sum(wgt)
    removed = (1 - T) * (1 - addback)
    return 1 - removed


def scatter_mc(theta_deg, s: Setup, eff_fep, dsdo=None, n=400_000, seed=11,
               attenuation=True, return_samples=False):
    dsdo = dsdo or ph.klein_nishina_dsdo_cos
    rng = np.random.default_rng(seed)
    P, vol = sample_wall(n, s, rng)
    Q, nrm, area = aperture_points(theta_deg, s, n, rng)
    DP = np.linalg.norm(P, axis=1)
    k0 = P / DP[:, None]
    psi = np.arccos(np.clip(k0[:, 0], -1, 1))
    v = Q - P
    DQ = np.linalg.norm(v, axis=1)
    k1 = v / DQ[:, None]
    cos_s = np.sum(k0 * k1, axis=1)
    Ep = ph.compton_energy_cos(cos_s)
    cosa = np.abs(k1 @ nrm)
    w = beam_f(psi, s) / DP ** 2 * cosa / DQ ** 2 * dsdo(cos_s) * eff_fep(Ep) * s.n_e
    if attenuation:
        l_in = material_exit(P, -k0, s)
        l_out = material_exit(P, k1, s)
        A = np.exp(-s.mu(ph.E_CS137) * l_in - s.mu(Ep) * l_out)
        w = w * A
    integral = vol * area * np.mean(w)
    err = vol * area * np.std(w) / np.sqrt(n)
    W = np.sum(w)
    th = np.degrees(np.arccos(np.clip(cos_s, -1, 1)))
    out = dict(I=integral, dI=err,
               theta_mean=np.sum(w * th) / W,
               theta_rms=np.sqrt(np.sum(w * (th - np.sum(w * th) / W) ** 2) / W),
               onemcos_mean=np.sum(w * (1 - cos_s)) / W,
               E_mean=np.sum(w * Ep) / W,
               E_rms=np.sqrt(np.sum(w * (Ep - np.sum(w * Ep) / W) ** 2) / W),
               invE_mean=np.sum(w / Ep) / W,
               A_mean=(np.sum(w) / np.sum(w / A)) if attenuation else 1.0)
    if return_samples:
        out["samples"] = (Ep, w, th)
    return out


def illumination_integral(s: Setup, n=200_000, seed=5):
    rng = np.random.default_rng(seed)
    P, vol = sample_wall(n, s, rng)
    DP = np.linalg.norm(P, axis=1)
    psi = np.arccos(np.clip(P[:, 0] / DP, -1, 1))
    return vol * np.mean(beam_f(psi, s) / DP ** 2)
