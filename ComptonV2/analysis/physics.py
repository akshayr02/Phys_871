"""Compton kinematics, Klein-Nishina and Thomson cross sections, photon attenuation (keV, cm)."""
from functools import lru_cache
from pathlib import Path

import numpy as np

ME_C2 = 510.99895
R_E = 2.8179403262e-13
N_A = 6.02214076e23
E_CS137 = 661.657

ELEMENTS = {
    "Na": (11, 22.98977), "I": (53, 126.90447), "Al": (13, 26.981538),
    "Fe": (26, 55.845), "Cu": (29, 63.546), "Pb": (82, 207.2),
}
TARGET_MATERIALS = {
    "steel": {"xsec": "Fe", "Z_over_A": 26 / 55.845, "density": 7.85},
    "Al": {"xsec": "Al", "Z_over_A": 13 / 26.981538, "density": 2.699},
    "Cu": {"xsec": "Cu", "Z_over_A": 29 / 63.546, "density": 8.96},
}


def compton_energy(theta_deg, e0=E_CS137, mc2=ME_C2):
    th = np.radians(theta_deg)
    return e0 / (1.0 + (e0 / mc2) * (1.0 - np.cos(th)))


def compton_energy_cos(cos_t, e0=E_CS137, mc2=ME_C2):
    return e0 / (1.0 + (e0 / mc2) * (1.0 - cos_t))


def dE_dtheta(theta_deg, e0=E_CS137, mc2=ME_C2):
    th = np.radians(theta_deg)
    ep = compton_energy(theta_deg, e0, mc2)
    return -(ep ** 2 / mc2) * np.sin(th) * np.pi / 180.0


def thomson_dsdo(theta_deg):
    c = np.cos(np.radians(theta_deg))
    return R_E ** 2 * (1.0 + c ** 2) / 2.0


def klein_nishina_dsdo_cos(cos_t, e0=E_CS137):
    p = 1.0 / (1.0 + (e0 / ME_C2) * (1.0 - cos_t))
    return 0.5 * R_E ** 2 * p ** 2 * (p + 1.0 / p - (1.0 - cos_t ** 2))


def klein_nishina_dsdo(theta_deg, e0=E_CS137):
    return klein_nishina_dsdo_cos(np.cos(np.radians(theta_deg)), e0)


def klein_nishina_total(e0=E_CS137):
    k = e0 / ME_C2
    a = (1 + k) / k ** 2 * (2 * (1 + k) / (1 + 2 * k) - np.log(1 + 2 * k) / k)
    b = np.log(1 + 2 * k) / (2 * k) - (1 + 3 * k) / (1 + 2 * k) ** 2
    return 2 * np.pi * R_E ** 2 * (a + b)


def sample_kn_cos(e0, rng):
    e0 = np.atleast_1d(np.asarray(e0, float))
    k = e0 / ME_C2
    out = np.empty_like(e0)
    todo = np.arange(e0.size)
    while todo.size:
        kk = k[todo]
        r1, r2, r3 = rng.random((3, todo.size))
        branch = r1 <= (1 + 2 * kk) / (9 + 2 * kk)
        x = np.where(branch, 1 + 2 * kk * r2, (1 + 2 * kk) / (1 + 2 * kk * r2))
        cos_t = 1 - (x - 1) / kk
        accept = np.where(branch, r3 <= 4 * (1 / x - 1 / x ** 2),
                          r3 <= 0.5 * (cos_t ** 2 + 1 / x))
        out[todo[accept]] = cos_t[accept]
        todo = todo[~accept]
    return out


_XSEC_FILE = Path(__file__).resolve().parent / "data" / "xsec_elam.csv"


@lru_cache(maxsize=None)
def _table(material):
    rows = []
    with open(_XSEC_FILE) as fh:
        for line in fh:
            if line.startswith("#") or line.startswith("material"):
                continue
            f = line.strip().split(",")
            if f[0] == material:
                rows.append([float(v) for v in f[1:]])
    if not rows:
        raise KeyError(material)
    e, ph, co, inc, rho = np.array(rows).T
    return e, ph, co, inc, float(rho[0])


def mu_rho(material, e_kev, kind="total"):
    e, ph, co, inc, _ = _table(material)
    parts = {"photo": ph, "coherent": co, "incoherent": inc,
             "total": ph + co + inc, "nocoh": ph + inc}[kind]
    x = np.log(np.asarray(e_kev, float))
    return np.exp(np.interp(x, np.log(e), np.log(parts)))


def mu_linear(material, e_kev, density=None, kind="total"):
    rho = _table(material)[4] if density is None else density
    return mu_rho(material, e_kev, kind) * rho


def electron_density(material_key, density=None):
    m = TARGET_MATERIALS[material_key]
    rho = m["density"] if density is None else density
    return rho * m["Z_over_A"] * N_A
