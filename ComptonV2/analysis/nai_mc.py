"""Photon-transport Monte Carlo of the NaI(Tl) crystal: total and full-energy-peak efficiencies."""
import json

import numpy as np

import config as C
import physics as ph

K_EDGE_I = 33.169
E_KX_I = 28.6
OMEGA_K_I = 0.882
K_FRACTION = 1 - 1 / 6.4
I_PHOTO_FRACTION = 0.995


def _exit_distance(pos, d, R, L):
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (pos[:, 0] * d[:, 0] + pos[:, 1] * d[:, 1])
    c = pos[:, 0] ** 2 + pos[:, 1] ** 2 - R ** 2
    with np.errstate(invalid="ignore", divide="ignore"):
        disc = np.sqrt(np.maximum(b ** 2 - 4 * a * c, 0))
        t_side = np.where(a > 1e-14, (-b + disc) / (2 * a), np.inf)
        t_z = np.where(d[:, 2] > 0, (L - pos[:, 2]) / d[:, 2],
                       np.where(d[:, 2] < 0, -pos[:, 2] / d[:, 2], np.inf))
    return np.minimum(t_side, t_z)


def _rotate(d, cos_t, rng):
    phi = 2 * np.pi * rng.random(cos_t.size)
    sin_t = np.sqrt(np.maximum(1 - cos_t ** 2, 0))
    ux, uy, uz = d[:, 0], d[:, 1], d[:, 2]
    out = np.empty_like(d)
    near_pole = np.abs(uz) > 0.99999
    s = np.sqrt(np.maximum(1 - uz ** 2, 1e-30))
    cp, sp = np.cos(phi), np.sin(phi)
    out[:, 0] = sin_t * (ux * uz * cp - uy * sp) / s + ux * cos_t
    out[:, 1] = sin_t * (uy * uz * cp + ux * sp) / s + uy * cos_t
    out[:, 2] = -sin_t * cp * s + uz * cos_t
    if np.any(near_pole):
        sgn = np.sign(uz[near_pole])
        out[near_pole, 0] = sin_t[near_pole] * cp[near_pole]
        out[near_pole, 1] = sin_t[near_pole] * sp[near_pole]
        out[near_pole, 2] = sgn * cos_t[near_pole]
    return out / np.linalg.norm(out, axis=1)[:, None]


def simulate(E0, n=200_000, diameter=5.08, length=5.08, d_src=20.0, seed=1,
             return_deposits=False):
    rng = np.random.default_rng(seed)
    R, L = diameter / 2, length
    cos_max = d_src / np.hypot(d_src, R)
    ct = 1 - rng.random(n) * (1 - cos_max)
    st = np.sqrt(1 - ct ** 2)
    phi = 2 * np.pi * rng.random(n)
    d = np.stack([st * np.cos(phi), st * np.sin(phi), ct], axis=1)
    rho_entry = d_src * st / ct
    pos = np.stack([rho_entry * np.cos(phi), rho_entry * np.sin(phi), np.zeros(n)], axis=1)
    pos[:, 2] = 1e-9
    E = np.full(n, float(E0))
    ev = np.arange(n)
    dep = np.zeros(n)
    interacted = np.zeros(n, bool)

    while ev.size:
        mu_tot = ph.mu_linear("NaI", E, kind="nocoh")
        s = -np.log(rng.random(ev.size)) / mu_tot
        t_exit = _exit_distance(pos, d, R, L)
        inside = s < t_exit
        ev, pos, d, E, s = ev[inside], pos[inside], d[inside], E[inside], s[inside]
        if ev.size == 0:
            break
        pos = pos + d * s[:, None]
        interacted[ev] = True
        p_photo = ph.mu_rho("NaI", E, "photo") / ph.mu_rho("NaI", E, "nocoh")
        is_photo = rng.random(ev.size) < p_photo
        pe = np.where(is_photo)[0]
        kx = pe[(E[pe] > K_EDGE_I) &
                (rng.random(pe.size) < I_PHOTO_FRACTION * K_FRACTION * OMEGA_K_I)]
        dep_pe = E[pe].copy()
        np.add.at(dep, ev[pe], dep_pe)
        np.add.at(dep, ev[kx], -E_KX_I)
        cs = np.where(~is_photo)[0]
        cos_t = ph.sample_kn_cos(E[cs], rng)
        E_new = ph.compton_energy_cos(cos_t, E[cs])
        np.add.at(dep, ev[cs], E[cs] - E_new)
        d_cs = _rotate(d[cs], cos_t, rng)
        nkx = kx.size
        u = 2 * rng.random(nkx) - 1
        ph_k = 2 * np.pi * rng.random(nkx)
        d_kx = np.stack([np.sqrt(1 - u ** 2) * np.cos(ph_k), np.sqrt(1 - u ** 2) * np.sin(ph_k), u], 1)
        ev = np.concatenate([ev[cs], ev[kx]])
        pos = np.concatenate([pos[cs], pos[kx]])
        d = np.concatenate([d_cs, d_kx])
        E = np.concatenate([E_new, np.full(nkx, E_KX_I)])
        keep = E > 1.0
        ev, pos, d, E = ev[keep], pos[keep], d[keep], E[keep]

    fep = np.abs(dep - E0) < 0.5
    out = dict(E=E0, n=n, eps_tot=interacted.mean(), eps_fep=fep.mean(),
               escape_kx=(np.abs(dep - (E0 - E_KX_I)) < 0.5).mean())
    out["deps_tot"] = np.sqrt(out["eps_tot"] * (1 - out["eps_tot"]) / n)
    out["deps_fep"] = np.sqrt(out["eps_fep"] * (1 - out["eps_fep"]) / n)
    out["PT"] = out["eps_fep"] / out["eps_tot"]
    if return_deposits:
        out["deposits"] = dep[interacted]
    return out


def efficiency_table(diameter=5.08, length=5.08, d_src=20.0, energies=None,
                     n=200_000, seed=7):
    energies = np.arange(150, 701, 25.0) if energies is None else np.asarray(energies)
    energies = np.unique(np.append(energies, ph.E_CS137))
    rows = [simulate(E, n, diameter, length, d_src, seed + i) for i, E in enumerate(energies)]
    return {k: np.array([r[k] for r in rows]) for k in
            ("E", "eps_tot", "eps_fep", "PT", "deps_tot", "deps_fep", "escape_kx")}


def window_transmission(E, t_al=C.WINDOW_AL_CM[0]):
    return np.exp(-ph.mu_linear("Al", E) * t_al)


class Efficiency:
    def __init__(self, table):
        self.t = table

    def fep(self, E):
        x = np.log(self.t["E"])
        return np.exp(np.interp(np.log(E), x, np.log(self.t["eps_fep"]))) * window_transmission(E)

    def tot(self, E):
        return np.interp(E, self.t["E"], self.t["eps_tot"])

    def pt(self, E):
        return np.interp(E, self.t["E"], self.t["PT"])


def run(verbose=True, n=200_000):
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = C.CACHE_DIR / "nai_efficiency_tables.json"
    key = dict(n=n, crystal=C.CRYSTAL, alts=C.CRYSTAL_ALTERNATIVES, version=4,
               geom=[C.SOURCE_TO_TARGET[0], C.SOURCE_TO_DETECTOR[0]], window=C.WINDOW_AL_CM[0])
    if cache.exists():
        js = json.loads(cache.read_text())
        if js.get("key") == key:
            return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in js["tables"].items()}
    tables = {}
    crystals = [("nominal", C.CRYSTAL)] + [(f"alt{i}", c) for i, c in enumerate(C.CRYSTAL_ALTERNATIVES)]
    d_scat = C.SOURCE_TO_DETECTOR[0] - C.SOURCE_TO_TARGET[0]
    for name, cr in crystals:
        tables[name] = efficiency_table(cr["diameter"], cr["length"], d_src=d_scat, n=n)
        tables[name + "_direct"] = efficiency_table(cr["diameter"], cr["length"],
                                                    d_src=C.SOURCE_TO_DETECTOR[0],
                                                    energies=[ph.E_CS137], n=4 * n, seed=99)
    tables["val3x3"] = efficiency_table(7.62, 7.62, d_src=60.0,
                                        energies=[263, 320, 402, 508, 564, 614], n=n)
    js = {"key": key, "tables": {k: {kk: vv.tolist() for kk, vv in v.items()} for k, v in tables.items()}}
    cache.write_text(json.dumps(js))
    if verbose:
        t = tables["nominal"]
        for i in range(len(t["E"])):
            print(f"  E={t['E'][i]:6.1f}  eps_tot={t['eps_tot'][i]:.4f}  eps_fep={t['eps_fep'][i]:.4f}  P/T={t['PT'][i]:.3f}")
    return tables


BICRON_3X3 = np.array([[614, 0.47, 0.865], [564, 0.50, 0.890], [508, 0.53, 0.930],
                       [402, 0.57, 0.960], [320, 0.65, 0.990], [263, 0.72, 0.999]])

if __name__ == "__main__":
    tb = run()
    v = tb["val3x3"]
    print("3x3 validation (E, MC P/T, Bicron P/T, MC eps_tot, Bicron eff):")
    for row in BICRON_3X3:
        i = np.argmin(np.abs(v["E"] - row[0]))
        print(f"  {row[0]:.0f}  {v['PT'][i]:.3f}  {row[1]:.2f}   {v['eps_tot'][i]:.3f}  {row[2]:.3f}")
