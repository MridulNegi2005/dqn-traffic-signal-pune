"""Build SUMO route files from the 5-minute Pune counts."""
from pathlib import Path

import numpy as np
import pandas as pd

from config import (ARM_CAMERA, CLASSES, DATA_CSV, INBOUND_SHARE, MINOR_ARM,
                    MINOR_ARM_FACTOR, TURN_SPLIT)

OUT_OF = {"N": "S", "S": "N", "E": "W", "W": "E"}
# left-hand traffic: from N heading south, a left turn goes to E
LEFT_OF = {"N": "E", "E": "S", "S": "W", "W": "N"}
RIGHT_OF = {"N": "W", "W": "S", "S": "E", "E": "N"}


def load_counts(junction: str = "JehangirChowk") -> pd.DataFrame:
    d = pd.read_csv(DATA_CSV, parse_dates=["time"])
    d = d[d.junction == junction]
    return d.groupby(["camera", "date", "time"], as_index=False)[CLASSES].sum()


def arm_rates(counts: pd.DataFrame, date: str) -> pd.DataFrame:
    """Vehicles per hour, per arm and class, for each 5-minute bin of one day."""
    day = counts[counts.date == date]
    rows = []
    for arm, cam in ARM_CAMERA.items():
        c = day[day.camera == cam].set_index("time")[CLASSES] * 12 * INBOUND_SHARE
        rows.append(c.assign(arm=arm))
    df = pd.concat(rows).reset_index()
    major = df.groupby("time")[CLASSES].mean() * MINOR_ARM_FACTOR
    df = pd.concat([df, major.assign(arm=MINOR_ARM).reset_index()])
    return df.sort_values(["time", "arm"]).reset_index(drop=True)


def write_routes(rates: pd.DataFrame, start: pd.Timestamp, seconds: int,
                 path: Path, seed: int, scale: float = 1.0) -> int:
    """Poisson arrivals per second, from 5-minute rates. Returns vehicle count."""
    rng = np.random.default_rng(seed)
    w = rates[(rates.time >= start) & (rates.time < start + pd.Timedelta(seconds=seconds))]
    moves = [("straight", OUT_OF), ("left", LEFT_OF), ("right", RIGHT_OF)]
    probs = [TURN_SPLIT[m] for m, _ in moves]
    veh = []
    for _, r in w.iterrows():
        t0 = int((r.time - start).total_seconds())
        for cls in CLASSES:
            lam = r[cls] * scale / 3600.0
            n = rng.poisson(lam * 300)
            for t in np.sort(rng.uniform(t0, t0 + 300, n)):
                m = rng.choice(len(moves), p=probs)
                dest = moves[m][1][r.arm]
                veh.append((t, cls, r.arm, dest))
    veh.sort()
    lines = ['<routes>']
    for i, (t, cls, o, dst) in enumerate(veh):
        lines.append(f'  <trip id="{i}" type="{cls}" depart="{t:.1f}" from="{o}_in" '
                     f'to="{dst}_out" departLane="best" departSpeed="max"/>')
    lines.append('</routes>')
    path.write_text("\n".join(lines))
    return len(veh)
