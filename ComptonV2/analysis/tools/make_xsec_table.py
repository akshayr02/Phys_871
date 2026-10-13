"""Build data/xsec_elam.csv from the Elam photon cross-section tables (requires xraydb)."""
from pathlib import Path

import numpy as np
import xraydb

MATERIALS = {
    "NaI": ({"Na": 22.98977 / 149.894, "I": 126.90447 / 149.894}, 3.667),
    "Al": ({"Al": 1.0}, 2.699),
    "Fe": ({"Fe": 1.0}, 7.874),
    "Cu": ({"Cu": 1.0}, 8.96),
    "Pb": ({"Pb": 1.0}, 11.35),
}
E_KEV = np.unique(np.concatenate([np.geomspace(10, 800, 160), [661.657]]))

rows = []
for mat, (frac, rho) in MATERIALS.items():
    parts = {k: np.zeros_like(E_KEV) for k in ("photo", "coh", "incoh")}
    for el, w in frac.items():
        for k in parts:
            parts[k] += w * xraydb.mu_elam(el, E_KEV * 1e3, kind=k)
    for i, e in enumerate(E_KEV):
        rows.append((mat, e, parts["photo"][i], parts["coh"][i], parts["incoh"][i], rho))

out = Path(__file__).resolve().parents[1] / "data" / "xsec_elam.csv"
with open(out, "w") as fh:
    fh.write("# mass attenuation coefficients (cm^2/g) from Elam, Ravel & Sieber, "
             "Radiat. Phys. Chem. 63, 121 (2002), via xraydb\n")
    fh.write("material,E_keV,photo,coherent,incoherent,density\n")
    for r in rows:
        fh.write("%s,%.4f,%.6e,%.6e,%.6e,%.4f\n" % r)
print("wrote", out, len(rows), "rows")
