"""Compare fixed-time, actuated and DQN control on unseen days, demand and junctions."""
import argparse
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN

import config as C
from env import PuneCrossingEnv

NETS = {"fixed": C.SUMO_DIR / "cross.net.xml", "actuated": C.SUMO_DIR / "cross_actuated.net.xml"}

# name -> (junction, demand scale). All use the held-out test days.
SCENARIOS = {
    "unseen_days": ("JehangirChowk", 1.0),
    "heavy_demand_1.3x": ("JehangirChowk", 1.3),
    "unseen_junction_RTO": ("RTOChowk", 1.0),
}
KEYS = ["scenario", "controller"]


def read_tripinfo(path):
    rows = []
    for t in ET.parse(path).getroot().iter("tripinfo"):
        cls = t.get("vType")
        em = t.find("emissions")
        rows.append(dict(vtype=cls, pcu=C.PCU.get(cls, 1.0),
                         finished=float(t.get("arrival", "-1")) >= 0,
                         wait=float(t.get("waitingTime")) + float(t.get("departDelay")),
                         delay=float(t.get("timeLoss")) + float(t.get("departDelay")),
                         duration=float(t.get("duration")),
                         co2_g=float(em.get("CO2_abs")) / 1000 if em is not None else 0.0,
                         fuel_g=float(em.get("fuel_abs")) / 1000 if em is not None else 0.0))
    return rows


def run(job):
    scenario, controller, model_path, date, seed, tmp = job
    junction, scale = SCENARIOS[scenario]
    ti = Path(tmp) / f"trip_{scenario}_{controller}_{date}_{seed}.xml"
    env = PuneCrossingEnv(dates=[date], episode_seconds=C.DAY_SECONDS, random_start=False,
                          net_file=NETS.get(controller, NETS["fixed"]), tripinfo=ti,
                          scale=scale, junction=junction)
    opts = {"date": date, "route_seed": seed}
    if model_path is None:
        env.reset(seed=seed, options=opts)
        env.run_uncontrolled(options=opts)
    else:
        model = DQN.load(model_path, device="cpu")
        obs, _ = env.reset(seed=seed, options=opts)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(act)
        env.close()
    rows = read_tripinfo(ti)
    for r in rows:
        r.update(scenario=scenario, controller=controller, date=date, seed=seed,
                 generated=env.n_vehicles)
    ti.unlink()
    print(scenario, controller, date, seed, "done", flush=True)
    return rows


def per_run(df):
    """One row of metrics per simulation run."""
    g = df.groupby(KEYS + ["date", "seed"])
    hours = C.DAY_SECONDS / 3600
    return pd.DataFrame({
        "entered_pct": 100 * g.size() / g.generated.first(),
        "throughput_per_h": g.finished.sum() / hours,
        "mean_wait_s": g.wait.mean(),
        "mean_delay_s": g.delay.mean(),
        "mean_trip_s": g.duration.mean(),
        "co2_g_per_veh": g.co2_g.mean(),
        "fuel_g_per_veh": g.fuel_g.mean(),
        "pcu_weighted_wait_s": g.apply(lambda x: (x.wait * x.pcu).sum() / x.pcu.sum()),
    }).reset_index()


def summarise(runs):
    """Mean over days and route seeds; for DQN also the std over training seeds."""
    runs = runs.copy()
    runs["family"] = runs.controller.str.replace(r"_seed\d+$", "", regex=True)
    metrics = [c for c in runs.columns if c not in KEYS + ["date", "seed", "family"]]
    per_model = runs.groupby(["scenario", "family", "controller"])[metrics].mean()
    mean = per_model.groupby(level=["scenario", "family"]).mean()
    std = per_model.groupby(level=["scenario", "family"]).std().fillna(0)
    out = mean.round(2)
    for m in ["mean_wait_s", "mean_delay_s", "throughput_per_h"]:
        out[m + "_std"] = std[m].round(2)
    return out.reset_index().rename(columns={"family": "controller"})


def main(a):
    tmp = C.ROOT / "results" / "runs" / "eval"
    tmp.mkdir(parents=True, exist_ok=True)
    controllers = [("fixed", None), ("actuated", None)]
    controllers += [(Path(m).stem.replace("dqn_", "dqn_"), m) for m in a.models]
    jobs = [(sc, name, path, date, seed, str(tmp))
            for sc in SCENARIOS for name, path in controllers
            for date in C.TEST_DATES for seed in range(a.route_seeds)]
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(run, jobs) for r in part]
    df = pd.DataFrame(rows)
    df.to_csv(C.ROOT / "results" / "eval_trips.csv.gz", index=False)
    runs = per_run(df)
    runs.to_csv(C.ROOT / "results" / "eval_runs.csv", index=False)
    s = summarise(runs)
    s.to_csv(C.ROOT / "results" / "summary.csv", index=False)
    df["family"] = df.controller.str.replace(r"_seed\d+$", "", regex=True)
    by_cls = df.groupby(["scenario", "family", "vtype"]).wait.mean().unstack().round(2)
    by_cls.to_csv(C.ROOT / "results" / "wait_by_class.csv")
    pd.set_option("display.width", 200)
    print(s.to_string())
    print(by_cls.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, help="paths to trained DQN .zip files")
    p.add_argument("--route-seeds", type=int, default=2)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    main(p.parse_args())
