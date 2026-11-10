"""Write LaTeX tables for the report from results/*.csv."""
import json

import pandas as pd

import config as C

RES = C.ROOT / "results"
OUT = C.ROOT / "report" / "generated"
NAMES = {"fixed": "Fixed-time (90 s cycle)", "actuated": "Actuated (SUMO)", "dqn": "DQN (ours)"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s = pd.read_csv(RES / "summary.csv")
    lines = [r"\begin{tabular}{llrrrrr}", r"\toprule",
             r"Demand & Controller & Finished (\%) & Wait (s) & Delay (s) & Trip (s) & PCU wait (s) \\",
             r"\midrule"]
    for sc, g in s.groupby("scale"):
        base = g[g.controller == "fixed"].iloc[0]
        for ctl in ["fixed", "actuated", "dqn"]:
            r = g[g.controller == ctl].iloc[0]
            gain = 100 * (base.mean_wait_s - r.mean_wait_s) / base.mean_wait_s
            tag = "" if ctl == "fixed" else f" ({gain:+.0f}\\%)"
            name = rf"\textbf{{{NAMES[ctl]}}}" if ctl == "dqn" else NAMES[ctl]
            lines.append(f"{sc:g}$\\times$ & {name} & {r.finished_pct:.1f} & "
                         f"{r.mean_wait_s:.1f}{tag} & {r.mean_delay_s:.1f} & "
                         f"{r.mean_trip_s:.1f} & {r.pcu_weighted_wait_s:.1f} \\\\")
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (OUT / "summary.tex").write_text("\n".join(lines))

    w = pd.read_csv(RES / "wait_by_class.csv")
    lines = [r"\begin{tabular}{llrrrr}", r"\toprule",
             r"Demand & Controller & Car & Motorbike & Bus & Truck \\", r"\midrule"]
    for _, r in w.iterrows():
        lines.append(f"{r.scale:g}$\\times$ & {NAMES[r.controller]} & {r.car:.1f} & "
                     f"{r.motorbike:.1f} & {r.bus:.1f} & {r.truck:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "by_class.tex").write_text("\n".join(lines))

    meta = json.loads((RES / "dqn_pune_meta.json").read_text())
    macros = [rf"\newcommand{{\trainsteps}}{{{meta['steps']:,}}}",
              rf"\newcommand{{\trainminutes}}{{{meta['minutes']}}}",
              rf"\newcommand{{\trainenvs}}{{{meta['envs']}}}",
              rf"\newcommand{{\traindevice}}{{{meta['device'].upper()}}}"]
    (OUT / "macros.tex").write_text("\n".join(macros))
    print("tables written to", OUT)


if __name__ == "__main__":
    main()
