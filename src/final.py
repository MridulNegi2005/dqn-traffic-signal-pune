"""Merge the final Kaggle evaluation files into summary tables for plots and the report."""
import glob
import sys

import pandas as pd

import config as C
from evaluate import per_run, summarise

RES = C.ROOT / "results"
RENAME = {
    "r03_ws_seed1": "dqn_camera_seed1", "r04_ws_seed2": "dqn_camera_seed2",
    "r05_ws_seed3": "dqn_camera_seed3", "r10_ws_fullobs": "dqn_full_seed1",
    "r10_ws_fullobs_seed2": "dqn_full_seed2", "r10_ws_fullobs_seed3": "dqn_full_seed3",
}


def main(pattern):
    files = glob.glob(str(RES / pattern))
    df = pd.concat(pd.read_csv(f) for f in files)
    df["controller"] = df.controller.replace(RENAME)
    df.to_csv(RES / "eval_trips.csv.gz", index=False)
    runs = per_run(df)
    runs.to_csv(RES / "eval_runs.csv", index=False)
    s = summarise(runs)
    s.to_csv(RES / "summary.csv", index=False)
    df["family"] = df.controller.str.replace(r"_seed\d+$", "", regex=True)
    by_cls = df.groupby(["scenario", "family", "vtype"]).wait.mean().unstack().round(2)
    by_cls.to_csv(RES / "wait_by_class.csv")
    pd.set_option("display.width", 250)
    print(s[["scenario", "controller", "entered_pct", "throughput_per_h", "mean_wait_s",
             "mean_wait_s_std", "mean_trip_s", "co2_g_per_veh", "fuel_g_per_veh"]].to_string(index=False))
    print(by_cls.to_string())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "final/eval_trips_fe_*.csv.gz")
