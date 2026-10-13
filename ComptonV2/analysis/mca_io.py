"""Reader for Amptek .mca spectrum files."""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Spectrum:
    path: Path
    counts: np.ndarray
    live_time: float
    real_time: float
    start_time: str = ""
    header: dict = field(default_factory=dict)

    @property
    def name(self):
        return self.path.stem

    @property
    def channels(self):
        return np.arange(self.counts.size)

    @property
    def dead_fraction(self):
        if self.real_time > 0 and self.live_time > 0:
            return 1.0 - self.live_time / self.real_time
        return np.nan


def read_mca(path, live_time_override=None):
    path = Path(path)
    header, data = {}, []
    section = None
    with open(path, "r", encoding="latin-1") as fh:
        for raw in fh:
            line = raw.strip()
            if line.startswith("<<") and line.endswith(">>"):
                section = line.strip("<>").strip()
                continue
            if section == "PMCA SPECTRUM" and " - " in line:
                k, v = line.split(" - ", 1)
                header[k.strip()] = v.strip()
            elif section == "DATA" and line:
                data.append(float(line))
    lt = float(header.get("LIVE_TIME", "0") or 0)
    rt = float(header.get("REAL_TIME", "0") or 0)
    if live_time_override is not None:
        lt = live_time_override
    return Spectrum(path, np.asarray(data), lt, rt, header.get("START_TIME", ""), header)
