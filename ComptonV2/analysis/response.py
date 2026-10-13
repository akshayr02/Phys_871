"""NaI(Tl) response matrix and folding of photon spectra into pulse-height spectra."""
import numpy as np

import calibration as cb
import config as C
import nai_mc

DE = 2.0
E_EDGES = np.arange(0.0, 800.0 + DE, DE)
E_CENT = 0.5 * (E_EDGES[1:] + E_EDGES[:-1])
GRID = np.arange(60.0, 701.0, 10.0)


def build(crystal=None, n=60_000, seed=101):
    crystal = crystal or C.CRYSTAL
    cache = C.CACHE_DIR / f"nai_response_{crystal['diameter']:.2f}x{crystal['length']:.2f}cm.npz"
    d_scat = C.SOURCE_TO_DETECTOR[0] - C.SOURCE_TO_TARGET[0]
    if cache.exists():
        z = np.load(cache)
        if z["n"] == n and np.isclose(z["dsrc"], d_scat):
            return {k: z[k] for k in z.files}
    fep = np.zeros(GRID.size)
    cont = np.zeros((GRID.size, E_CENT.size))
    for i, E in enumerate(GRID):
        r = nai_mc.simulate(E, n, crystal["diameter"], crystal["length"], d_scat,
                            seed + i, return_deposits=True)
        dep = r["deposits"]
        full = np.abs(dep - E) < 0.5
        fep[i] = full.sum() / n
        h, _ = np.histogram(dep[~full], bins=E_EDGES)
        cont[i] = h / n
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(cache, grid=GRID, fep=fep, cont=cont, n=n, dsrc=d_scat)
    return dict(grid=GRID, fep=fep, cont=cont, n=n, dsrc=d_scat)


def deposit_spectrum(E_arr, w_arr, resp, photopeak_only=False, continuum_only=False):
    E_arr = np.asarray(E_arr)
    w_arr = np.asarray(w_arr) * nai_mc.window_transmission(E_arr)
    out = np.zeros(E_CENT.size)
    if not continuum_only:
        fep = np.exp(np.interp(np.log(E_arr), np.log(resp["grid"]), np.log(resp["fep"])))
        out += np.histogram(E_arr, bins=E_EDGES, weights=w_arr * fep)[0]
    if not photopeak_only:
        h_in, _ = np.histogram(E_arr, bins=E_EDGES, weights=w_arr)
        for j in np.nonzero(h_in)[0]:
            E = E_CENT[j]
            g = int(np.clip(np.round((E - resp["grid"][0]) / 10.0), 0, resp["grid"].size - 1))
            scale = E / resp["grid"][g]
            c = resp["cont"][g]
            out += h_in[j] * np.interp(E_CENT, E_CENT * scale, c / scale, left=0, right=0)
    return out


def smear(dep, res):
    sig = cb.sigma_E(res, np.maximum(E_CENT, 20.0))
    z = (E_CENT[:, None] - E_CENT[None, :]) / sig[None, :]
    M = np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * sig[None, :]) * DE
    return M @ dep


def to_channels(spec_E, cal, x_ch, rebin):
    b, a = cal["coef"][0], cal["coef"][1]
    dens = spec_E / DE
    E_c = a + b * np.asarray(x_ch)
    return np.interp(E_c, E_CENT, dens, left=0, right=0) * b * rebin
