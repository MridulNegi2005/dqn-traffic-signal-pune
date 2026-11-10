"""Compare fixed-time, actuated and DQN control on unseen test days."""
import argparse
import os
import xml.etree.ElementTree as ET
from multiprocessing import Pool

import pandas as pd
from stable_baselines3 import DQN

import config as C
from env import PuneCrossingEnv

NETS = {"fixed": C.SUMO_DIR / "cross.net.xml", "actuated": C.SUMO_DIR / "cross_actuated.net.xml"}


def read_tripinfo(path, controller, date, seed):
    rows = []
    for t in ET.parse(path).getroot().iter("tripinfo"):
        cls = t.get("vType")
        rows.append(dict(controller=controller, date=date, seed=seed, vtype=cls,
                         pcu=C.PCU.get(cls, 1.0),
                         finished=float(t.get("arrival", "-1")) >= 0,
                         wait=float(t.get("waitingTime")) + float(t.get("departDelay")),
                         delay=float(t.get("timeLoss")) + float(t.get("departDelay")),
                         duration=float(t.get("duration"))))
    return rows


def job(args):
    controller, date, seed, scale, model_path, tmp = args
    model = DQN.load(model_path, device="cpu") if controller == "dqn" else None
    rows = run(controller, date, seed, model, scale, tmp)
    print(scale, date, seed, controller, "done", flush=True)
    return rows


def run(controller, date, seed, model=None, scale=1.0, tmp=None):
    ti = tmp / f"trip_{controller}_{date}_{seed}_{scale}.xml"
    net = NETS.get(controller, NETS["fixed"])
    env = PuneCrossingEnv(dates=[date], episode_seconds=C.DAY_SECONDS, random_start=False,
                          net_file=net, tripinfo=ti, scale=scale)
    opts = {"date": date, "route_seed": seed}
    if model is None:
        env.reset(seed=seed, options=opts)
        env.run_uncontrolled(options=opts)
    else:
        obs, _ = env.reset(seed=seed, options=opts)
        done = False
        while not done:
            act, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(act)
        env.close()
    rows = read_tripinfo(ti, controller, date, seed)
    for r in rows:
        r["scale"] = scale
        r["generated"] = env.n_vehicles
    return rows


def summarise(df):
    g = df.groupby(["controller", "scale"])
    runs = df.groupby(["controller", "scale", "date", "seed"]).agg(
        entered=("wait", "size"), arrived=("finished", "sum"), generated=("generated", "first"))
    rg = runs.groupby(level=["controller", "scale"]).sum()
    s = pd.DataFrame({
        "entered_pct": 100 * rg.entered / rg.generated,
        "finished_pct": 100 * rg.arrived / rg.generated,
        "throughput_per_h": rg.arrived / runs.groupby(level=["controller", "scale"]).size()
                            / (C.DAY_SECONDS / 3600),
        "mean_wait_s": g.wait.mean(),
        "mean_delay_s": g.delay.mean(),
        "mean_trip_s": g.duration.mean(),
        "pcu_weighted_wait_s": g.apply(lambda x: (x.wait * x.pcu).sum() / x.pcu.sum()),
    })
    return s.round(2)


def main(a):
    model_path = C.ROOT / "results" / f"{a.model}.zip"
    tmp = C.ROOT / "results" / "runs" / "eval"
    tmp.mkdir(parents=True, exist_ok=True)
    jobs = [(ctl, date, seed, scale, model_path, tmp)
            for scale in a.scales for date in C.TEST_DATES
            for seed in range(a.seeds) for ctl in ["fixed", "actuated", "dqn"]]
    with Pool(a.workers) as pool:
        rows = [r for part in pool.imap_unordered(job, jobs) for r in part]
    df = pd.DataFrame(rows)
    df.to_csv(C.ROOT / "results" / "eval_trips.csv.gz", index=False)
    s = summarise(df)
    s.to_csv(C.ROOT / "results" / "summary.csv")
    by_cls = df.groupby(["scale", "controller", "vtype"]).wait.mean().unstack().round(2)
    by_cls.to_csv(C.ROOT / "results" / "wait_by_class.csv")
    print(s.to_string())
    print(by_cls.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="dqn_pune")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--scales", type=float, nargs="+", default=[1.0, 1.3])
    p.add_argument("--workers", type=int, default=os.cpu_count())
    main(p.parse_args())
