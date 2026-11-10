"""Compare fixed-time, actuated and DQN control on unseen days, demand and junctions."""
import argparse
import json
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from stable_baselines3 import DQN

import config as C
from env import PuneCrossingEnv

# name -> (junction, demand scale, free left). All use the held-out test days.
SCENARIOS = {
    "unseen_days": ("JehangirChowk", 1.0, False),
    "heavy_demand_1.3x": ("JehangirChowk", 1.3, False),
    "unseen_junction_RTO": ("RTOChowk", 1.0, False),
    "free_left": ("JehangirChowk", 1.0, True),
    "free_left_1.3x": ("JehangirChowk", 1.3, True),
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


def load_agent(model_path):
    """Return (model, env kwargs) using the meta file saved next to the model."""
    meta_path = Path(str(model_path).replace(".zip", "_meta.json"))
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    env_kw = meta.get("env", {})
    if meta.get("algo") == "qrdqn":
        from sb3_contrib import QRDQN
        return QRDQN.load(model_path, device="cpu"), env_kw
    return DQN.load(model_path, device="cpu"), env_kw


def run(job):
    scenario, controller, model_path, date, seed, tmp = job
    junction, scale, free_left = SCENARIOS[scenario]
    ti = Path(tmp) / f"trip_{scenario}_{controller}_{date}_{seed}.xml"
    model, env_kw = (None, {}) if model_path is None else load_agent(model_path)
    if controller == "camera_rule":
        env_kw = dict(delta_t=2)
    base = controller in ("fixed", "actuated")
    env = PuneCrossingEnv(dates=[date], episode_seconds=C.DAY_SECONDS, random_start=False,
                          tripinfo=ti, scale=scale, junction=junction, free_left=free_left,
                          controller=controller if base else "agent", **env_kw)
    opts = {"date": date, "route_seed": seed}
    greens = []
    if base:
        env.run_baseline(options=opts)
    else:
        obs, _ = env.reset(seed=seed, options=opts)
        done = False
        while not done:
            act = env.teacher_action() if model is None else model.predict(obs, deterministic=True)[0]
            obs, _, done, _, _ = env.step(act)
        greens = env.green_log
        env.close()
    rows = read_tripinfo(ti)
    for r in rows:
        r.update(scenario=scenario, controller=controller, date=date, seed=seed,
                 generated=env.n_vehicles)
    ti.unlink()
    if greens:
        g = pd.DataFrame(greens, columns=["arm", "green_s"]).groupby("arm").green_s
        info = ", ".join(f"{a}: {n} greens avg {m:.0f}s" for a, (n, m)
                         in g.agg(["count", "mean"]).iterrows())
    else:
        info = ""
    print(scenario, controller, date, seed, "done", info, flush=True)
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
    controllers = [("fixed", None), ("actuated", None), ("camera_rule", None)]
    controllers += [(Path(m).stem, m) for m in a.models]
    jobs = [(sc, name, path, date, seed, str(tmp))
            for sc in a.scenarios for name, path in controllers
            for date in C.TEST_DATES for seed in range(a.route_seeds)]
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(run, jobs) for r in part]
    df = pd.DataFrame(rows)
    tag = "" if len(a.scenarios) == len(SCENARIOS) else "_" + "_".join(a.scenarios)
    df.to_csv(C.ROOT / "results" / f"eval_trips{tag}.csv.gz", index=False)
    runs = per_run(df)
    runs.to_csv(C.ROOT / "results" / f"eval_runs{tag}.csv", index=False)
    report(df, runs)


def report(df, runs):
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
    p.add_argument("--models", nargs="+", default=[], help="paths to trained DQN .zip files")
    p.add_argument("--route-seeds", type=int, default=2)
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    p.add_argument("--merge", nargs="+", help="merge eval_trips*.csv.gz files instead of running")
    a = p.parse_args()
    if a.merge:
        df = pd.concat(pd.read_csv(f) for f in a.merge)
        df.to_csv(C.ROOT / "results" / "eval_trips.csv.gz", index=False)
        runs = per_run(df)
        runs.to_csv(C.ROOT / "results" / "eval_runs.csv", index=False)
        report(df, runs)
    else:
        main(a)
