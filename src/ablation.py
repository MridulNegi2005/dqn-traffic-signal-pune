"""Rank the training strategies from the quick-test runs and write the ablation table."""
import glob
import sys

import pandas as pd

import config as C

RES = C.ROOT / "results"
OUT = C.ROOT / "report" / "generated"
LABELS = {
    "fixed": "Fixed-time", "actuated": "Actuated (loop sensors)", "camera_rule": "Camera rule",
    "r01_dqn_5s": "R1 DQN, 5 s", "r02_dqn_2s": "R2 DQN, 2 s",
    "r03_ws": "R3--R5 warm start, camera view (3 seeds)", "r06_ws_keepswitch": "R6 keep/switch action",
    "r07_ws_qrdqn": "R7 QR-DQN", "r08_ws_300k": "R8 300k steps",
    "r09_ws_oldreward": "R9 old reward", "r10_ws_fullobs": "R10 full-lane view",
}


def family(name):
    return "r03_ws" if name.startswith(("r03_ws", "r04_ws", "r05_ws")) else name


def main(pattern):
    runs = pd.concat(pd.read_csv(f) for f in glob.glob(str(RES / pattern)))
    runs["family"] = runs.controller.map(family)
    metrics = ["mean_wait_s", "throughput_per_h", "entered_pct", "mean_trip_s", "co2_g_per_veh"]
    per_model = runs.groupby(["scenario", "family", "controller"])[metrics].mean()
    mean = per_model.groupby(level=["scenario", "family"]).mean()
    std = per_model.groupby(level=["scenario", "family"]).std()
    t = mean.unstack("scenario")
    t.columns = [f"{m}|{s}" for m, s in t.columns]
    t["wait_std|unseen_days"] = std["mean_wait_s"].unstack("scenario").get("unseen_days")
    t["score"] = t["mean_wait_s|unseen_days"] + t["mean_wait_s|heavy_demand_1.3x"]
    t = t.sort_values("score")
    t.to_csv(RES / "ablation.csv")
    pd.set_option("display.width", 250)
    cols = ["mean_wait_s|unseen_days", "mean_wait_s|heavy_demand_1.3x", "throughput_per_h|unseen_days",
            "throughput_per_h|heavy_demand_1.3x", "co2_g_per_veh|unseen_days", "score"]
    print(t[cols].round(1).to_string())

    OUT.mkdir(parents=True, exist_ok=True)
    baselines = ["fixed", "actuated", "camera_rule"]
    runs_order = ["r01_dqn_5s", "r02_dqn_2s", "r03_ws", "r06_ws_keepswitch", "r07_ws_qrdqn",
                  "r08_ws_300k", "r09_ws_oldreward", "r10_ws_fullobs"]
    colour = {"camera_rule": "bench", "r03_ws": "chosen"}

    def row(name):
        r = t.loc[name]
        pre = rf"\rowcolor{{{colour[name]}}}" if name in colour else ""
        bold = (lambda x: rf"\textbf{{{x}}}") if name in colour else (lambda x: x)
        cells = [LABELS.get(name, name), f"{r['mean_wait_s|unseen_days']:.0f}",
                 f"{r['mean_wait_s|heavy_demand_1.3x']:.0f}", f"{r['throughput_per_h|unseen_days']:.0f}",
                 f"{r['throughput_per_h|heavy_demand_1.3x']:.0f}", f"{r['co2_g_per_veh|unseen_days']:.0f}"]
        return pre + " & ".join(bold(c) for c in cells) + r" \\"

    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r" & \multicolumn{2}{c}{Mean wait (s)} & \multicolumn{2}{c}{Throughput (veh/h)} & CO$_2$ \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Controller & 1.0$\times$ & 1.3$\times$ & 1.0$\times$ & 1.3$\times$ & (g/veh) \\", r"\midrule",
             r"\multicolumn{6}{l}{\textit{Baselines}} \\"]
    lines += [row(n) for n in baselines if n in t.index]
    lines += [r"\midrule", r"\multicolumn{6}{l}{\textit{DQN training runs}} \\"]
    lines += [row(n) for n in runs_order if n in t.index]
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUT / "ablation.tex").write_text("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "strat/qt/eval_runs_qt_*.csv")
