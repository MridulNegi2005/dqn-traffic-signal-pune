"""Write LaTeX tables and macros for the report from results/*.csv."""
import json
from glob import glob

import pandas as pd

import config as C

RES = C.ROOT / "results"
OUT = C.ROOT / "report" / "generated"
ORDER = ["fixed", "actuated", "camera_rule", "dqn_camera"]
NAMES = {"fixed": "Fixed-time", "actuated": "Actuated (loops)", "camera_rule": "Camera rule",
         "dqn_camera": r"\textbf{DQN, camera view}", "dqn_full": "DQN, full view"}
ROWCOLOUR = {"camera_rule": r"\rowcolor{bench}", "dqn_camera": r"\rowcolor{chosen}"}
SCEN = {"unseen_days": "Unseen days",
        "heavy_demand_1.3x": r"1.3$\times$ demand",
        "unseen_junction_RTO": "Unseen junction (RTO)",
        "free_left": "Free left",
        "free_left_1.3x": r"Free left, 1.3$\times$"}


def summary_table(s):
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Scenario & Controller & Wait (s) & Delay (s) & Trip (s) & "
             r"Thr.\ (veh/h) & CO$_2$ (g) & Fuel (g) \\", r"\midrule"]
    for sc, label in SCEN.items():
        g = s[s.scenario == sc].set_index("controller")
        if g.empty:
            continue
        for i, ctl in enumerate([c for c in ORDER if c in g.index]):
            r = g.loc[ctl]
            wait = f"{r.mean_wait_s:.1f}"
            if ctl.startswith("dqn"):
                wait += rf" $\pm$ {r.mean_wait_s_std:.1f}"
            lines.append(f"{ROWCOLOUR.get(ctl, '')}{label if i == 0 else ''} & {NAMES[ctl]} & {wait} & "
                         f"{r.mean_delay_s:.1f} & {r.mean_trip_s:.1f} & {r.throughput_per_h:.0f} & "
                         f"{r.co2_g_per_veh:.0f} & {r.fuel_g_per_veh:.0f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def class_table(w):
    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Controller & Car & Motorbike & Bus & Truck \\", r"\midrule"]
    g = w[w.scenario == "unseen_days"].set_index("family")
    for ctl in [c for c in ORDER if c in g.index]:
        r = g.loc[ctl]
        lines.append(f"{NAMES[ctl]} & {r.car:.1f} & {r.motorbike:.1f} & {r.bus:.1f} & {r.truck:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(RES / "summary.csv")
    (OUT / "summary.tex").write_text(summary_table(s))
    (OUT / "by_class.tex").write_text(class_table(pd.read_csv(RES / "wait_by_class.csv")))
    metas = [json.loads(open(f).read()) for f in glob(str(RES / "strat" / "**" / "*_meta.json"), recursive=True)]
    mins = [m["minutes"] for m in metas] or [0]
    runs = pd.read_csv(RES / "eval_runs.csv")
    cam = runs[runs.controller.str.startswith("dqn_camera")]
    words = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}
    n_seeds, n_routes = cam.controller.nunique(), runs.seed.nunique()
    macros = [r"\newcommand{\trainsteps}{150,000}", r"\newcommand{\trainenvs}{4}",
              rf"\newcommand{{\trainseeds}}{{{n_seeds}}}",
              rf"\newcommand{{\camseedsword}}{{{words.get(n_seeds, n_seeds)}}}",
              rf"\newcommand{{\routeseeds}}{{{n_routes}}}",
              rf"\newcommand{{\routeseedsword}}{{{words.get(n_routes, n_routes)}}}",
              rf"\newcommand{{\trainminutes}}{{{min(mins):.0f}--{max(mins):.0f}}}"]
    (OUT / "macros.tex").write_text("\n".join(macros))
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
