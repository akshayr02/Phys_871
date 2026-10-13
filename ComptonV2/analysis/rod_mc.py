"""Single and multiple Compton scattering in the target, scored at the detector aperture."""
import numpy as np

import geometry as G
import physics as ph


def _aperture_sample(theta_deg, s, n, rng):
    Q, nrm, area = G.aperture_points(theta_deg, s, n, rng)
    return Q, nrm, area


def _nee(P, k_in, E_in, Q, nrm, area, s):
    v = Q - P
    DQ = np.linalg.norm(v, axis=1)
    k1 = v / DQ[:, None]
    cos_s = np.sum(k_in * k1, axis=1)
    E_out = ph.compton_energy_cos(cos_s, E_in)
    p = ph.klein_nishina_dsdo_cos(cos_s, E_in) / ph.klein_nishina_total(E_in)
    dOm = area * np.abs(k1 @ nrm) / DQ ** 2
    l_out = G.material_exit(P, k1, s)
    return p * dOm * np.exp(-s.mu(E_out) * l_out), E_out


def _rotate(d, cos_t, rng):
    phi = 2 * np.pi * rng.random(cos_t.size)
    sin_t = np.sqrt(np.maximum(1 - cos_t ** 2, 0))
    a = np.where(np.abs(d[:, 2:3]) < 0.9, np.array([[0, 0, 1.0]]), np.array([[1.0, 0, 0]]))
    u = np.cross(d, a)
    u /= np.linalg.norm(u, axis=1)[:, None]
    v = np.cross(d, u)
    return (cos_t[:, None] * d + (sin_t * np.cos(phi))[:, None] * u
            + (sin_t * np.sin(phi))[:, None] * v)


def transport(theta_deg, s: G.Setup, n=300_000, max_order=3, seed=21):
    rng = np.random.default_rng(seed)
    n_e = s.n_e
    P, vol = G.sample_wall(n, s, rng)
    DP = np.linalg.norm(P, axis=1)
    k = P / DP[:, None]
    psi = np.arccos(np.clip(k[:, 0], -1, 1))
    l_in = G.material_exit(P, -k, s)
    E = np.full(n, ph.E_CS137)
    muC0 = n_e * ph.klein_nishina_total(ph.E_CS137)
    W = vol / n * G.beam_f(psi, s) / DP ** 2 * np.exp(-s.mu(ph.E_CS137) * l_in) * muC0
    out = []
    for order in range(1, max_order + 1):
        Q, nrm, area = _aperture_sample(theta_deg, s, W.size, rng)
        fac, E_det = _nee(P, k, E, Q, nrm, area, s)
        out.append((E_det, W * fac))
        if order == max_order:
            break
        cos_t = ph.sample_kn_cos(E, rng)
        E_new = ph.compton_energy_cos(cos_t, E)
        k = _rotate(k, cos_t, rng)
        l_exit = G.material_exit(P, k, s)
        mu_t = s.mu(E_new)
        p_coll = 1 - np.exp(-mu_t * l_exit)
        u = rng.random(W.size)
        dist = -np.log(1 - u * p_coll) / mu_t
        P = G.advance(P, k, dist, s)
        muC = n_e * ph.klein_nishina_total(E_new)
        W = W * p_coll * muC / mu_t
        E = E_new
        keep = W > 0
        P, k, E, W = P[keep], k[keep], E[keep], W[keep]
    return out
