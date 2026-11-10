"""Compare fixed-time, actuated and DQN control on unseen test days."""
import argparse
import xml.etree.ElementTree as ET

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
                         wait=float(t.get("waitingTime")), delay=float(t.get("timeLoss")),
                         duration=float(t.get("duration"))))
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
    return rows


def summarise(df):
    g = df.groupby(["controller", "scale"])
    s = pd.DataFrame({
        "vehicles": g.size() / df.groupby(["controller", "scale"]).apply(
            lambda x: x[["date", "seed"]].drop_duplicates().shape[0]),
        "finished_pct": g.finished.mean() * 100,
        "mean_wait_s": g.wait.mean(),
        "mean_delay_s": g.delay.mean(),
        "mean_trip_s": g.duration.mean(),
        "pcu_weighted_wait_s": g.apply(lambda x: (x.wait * x.pcu).sum() / x.pcu.sum()),
    })
    return s.round(2)


def main(a):
    model = DQN.load(C.ROOT / "results" / f"{a.model}.zip", device="cpu")
    tmp = C.ROOT / "results" / "runs" / "eval"
    tmp.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale in a.scales:
        for date in C.TEST_DATES:
            for seed in range(a.seeds):
                for ctl in ["fixed", "actuated", "dqn"]:
                    rows += run(ctl, date, seed, model if ctl == "dqn" else None, scale, tmp)
                    print(scale, date, seed, ctl, "done", flush=True)
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
    main(p.parse_args())
