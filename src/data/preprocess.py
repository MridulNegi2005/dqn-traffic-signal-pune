"""Turn the Pune per-frame cumulative counts into 5-minute vehicle counts.

Each CSV row is one video frame and holds running totals since the start of the
recording session. A `_one.csv` file continues the session but restarts at zero.
"""
import argparse
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from config import ROOT

CLASSES = ["car", "motorbike", "bus", "truck"]
JUNCTIONS = {"a": "AlankarChowk", "j": "JehangirChowk", "r": "RTOChowk"}
NAME_RE = re.compile(r"traffic/(\w+)/(\w\d)/\w\d_(\d+)(_one)?\.csv$")


def file_to_bins(raw: pd.DataFrame, bin_s: int) -> pd.DataFrame:
    raw = raw.dropna(subset=["Time", "Direction"])
    t = pd.to_timedelta(raw["Time"]).dt.total_seconds().to_numpy()
    # sessions that run past midnight: unwrap the clock
    wrap = np.concatenate([[0], np.cumsum(np.diff(t) < -43200)])
    raw = raw.assign(sec=t + wrap * 86400)
    raw["bin"] = (raw["sec"] // bin_s * bin_s).astype(int)

    last = raw.groupby(["Direction", "bin"])[CLASSES].last().sort_index()
    inc = last.groupby(level="Direction").diff()
    first = last.groupby(level="Direction").head(1)
    inc.loc[first.index] = first
    return inc.clip(lower=0).reset_index()


def main(zip_path: Path, out: Path, bin_s: int) -> None:
    z = zipfile.ZipFile(zip_path)
    parts = []
    for name in sorted(z.namelist()):
        m = NAME_RE.search(name)
        if not m:
            continue
        junction, camera, day, _ = m.groups()
        raw = pd.read_csv(z.open(name), usecols=["Time", "Direction", *CLASSES])
        b = file_to_bins(raw, bin_s)
        b.insert(0, "date", f"2023-01-{int(day):02d}")
        b.insert(0, "camera", camera)
        b.insert(0, "junction", junction)
        parts.append(b)
        print(f"{name}: {len(raw):>8} frames -> {len(b):>5} rows")

    df = pd.concat(parts, ignore_index=True)
    key = ["junction", "camera", "date", "bin", "Direction"]
    df = df.groupby(key, as_index=False)[CLASSES].sum()
    df[CLASSES] = df[CLASSES].astype(int)
    df["time"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["bin"], unit="s")
    df = df.rename(columns={"Direction": "direction"}).drop(columns="bin")
    df = df[["junction", "camera", "date", "time", "direction", *CLASSES]]
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=ROOT / "data" / "raw" / "traffic.zip")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "pune_5min.csv")
    p.add_argument("--bin", type=int, default=300, help="bucket size in seconds")
    a = p.parse_args()
    main(a.zip, a.out, a.bin)
