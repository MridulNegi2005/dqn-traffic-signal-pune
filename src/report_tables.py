"""Write LaTeX tables and macros for the report from results/*.csv."""
import json
from glob import glob

import pandas as pd

import config as C

RES = C.ROOT / "results"
OUT = C.ROOT / "report" / "generated"
NAMES = {"fixed": "Fixed-time", "actuated": "Actuated", "dqn": r"\textbf{DQN (ours)}"}
SCEN = {"unseen_days": "Unseen days (Jehangir)",
        "heavy_demand_1.3x": r"Unseen days, 1.3$\times$ demand",
        "unseen_junction_RTO": "Unseen junction (RTO)"}


def pct(new, base):
    return 100 * (new - base) / base


def summary_table(s):
    lines = [r"\begin{tabular}{llrrrrrr}", r"\toprule",
             r"Scenario & Controller & Wait (s) & Delay (s) & Trip (s) & "
             r"Throughput (veh/h) & CO$_2$ (g/veh) & Fuel (g/veh) \\", r"\midrule"]
    for sc, label in SCEN.items():
        g = s[s.scenario == sc].set_index("controller")
        if g.empty:
            continue
        for i, ctl in enumerate(["fixed", "actuated", "dqn"]):
            r = g.loc[ctl]
            wait = f"{r.mean_wait_s:.1f}"
            if ctl == "dqn":
                wait += rf" $\pm$ {r.mean_wait_s_std:.1f}"
            lines.append(f"{label if i == 0 else ''} & {NAMES[ctl]} & {wait} & "
                         f"{r.mean_delay_s:.1f} & {r.mean_trip_s:.1f} & {r.throughput_per_h:.0f} & "
                         f"{r.co2_g_per_veh:.1f} & {r.fuel_g_per_veh:.1f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def class_table(w):
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Scenario & Controller & Car & Motorbike & Bus & Truck \\", r"\midrule"]
    for sc, label in SCEN.items():
        g = w[w.scenario == sc].set_index("family")
        for i, ctl in enumerate(["fixed", "actuated", "dqn"]):
            r = g.loc[ctl]
            lines.append(f"{label if i == 0 else ''} & {NAMES[ctl]} & {r.car:.1f} & "
                         f"{r.motorbike:.1f} & {r.bus:.1f} & {r.truck:.1f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(RES / "summary.csv")
    (OUT / "summary.tex").write_text(summary_table(s))
    (OUT / "by_class.tex").write_text(class_table(pd.read_csv(RES / "wait_by_class.csv")))

    metas = [json.loads(open(f).read()) for f in sorted(glob(str(RES / "dqn_seed*_meta.json")))]
    macros = [rf"\newcommand{{\trainsteps}}{{{metas[0]['steps']:,}}}",
              rf"\newcommand{{\trainenvs}}{{{metas[0]['envs']}}}",
              rf"\newcommand{{\trainseeds}}{{{len(metas)}}}",
              rf"\newcommand{{\trainminutes}}{{{min(m['minutes'] for m in metas):.0f}--"
              rf"{max(m['minutes'] for m in metas):.0f}}}"]
    for sc, key in [("unseen_days", "Days"), ("heavy_demand_1.3x", "Heavy"),
                    ("unseen_junction_RTO", "Rto")]:
        g = s[s.scenario == sc].set_index("controller")
        if g.empty:
            continue
        for base in ["fixed", "actuated"]:
            for m, short in [("mean_wait_s", "Wait"), ("mean_trip_s", "Trip"),
                             ("co2_g_per_veh", "Co"), ("throughput_per_h", "Thr")]:
                v = pct(g.loc["dqn", m], g.loc[base, m])
                macros.append(rf"\newcommand{{\d{short}{key}{base.capitalize()}}}{{{v:+.1f}\%}}")
    (OUT / "macros.tex").write_text("\n".join(macros))
    print((OUT / "macros.tex").read_text())


if __name__ == "__main__":
    main()
