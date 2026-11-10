import glob

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402

RES = C.ROOT / "results"
COLORS = {"fixed": "#8a8f98", "actuated": "#d9822b", "camera_rule": "#6b8e23",
          "dqn_camera": "#2b6cb0", "dqn_full": "#7b3fa0"}
ORDER = ["fixed", "actuated", "camera_rule", "dqn_camera", "dqn_full"]
LABELS = {"fixed": "Fixed-time", "actuated": "Actuated (loop sensors)", "camera_rule": "Camera rule",
          "dqn_camera": "DQN, camera view (5 seeds)", "dqn_full": "DQN, full view (3 seeds)"}
SCEN = {"unseen_days": "Unseen\ndays", "heavy_demand_1.3x": "1.3x\ndemand",
        "unseen_junction_RTO": "Unseen junction\n(RTO)", "free_left": "Free\nleft",
        "free_left_1.3x": "Free left,\n1.3x"}
CURVES = {"dqn_camera": ["r03_ws_seed1", "r04_ws_seed2", "r05_ws_seed3", "cam_ws_seed4", "cam_ws_seed5"],
          "dqn_full": ["r10_ws_fullobs", "r10_ws_fullobs_seed2", "r10_ws_fullobs_seed3"]}


def learning_curve():
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    for fam, names in CURVES.items():
        curves = []
        for n in names:
            files = glob.glob(str(RES / "strat" / "**" / f"{n}_monitor" / "*.monitor.csv"), recursive=True)
            if not files:
                continue
            df = pd.concat(pd.read_csv(f, skiprows=1) for f in files).sort_values("t")
            r = df.r.rolling(15, min_periods=1).mean().to_numpy()
            curves.append(r)
            ax.plot(np.arange(1, len(r) + 1), r, lw=0.8, alpha=0.35, color=COLORS[fam])
        if curves:
            n = min(len(c) for c in curves)
            ax.plot(np.arange(1, n + 1), np.mean([c[:n] for c in curves], axis=0), lw=2.2,
                    color=COLORS[fam], label=LABELS[fam].replace(" (3 seeds)", "") + " (mean)")
    ax.set_xlabel("Training episode (1 hour of Pune traffic, random demand 0.7-1.4x)")
    ax.set_ylabel("Episode return (15-ep. mean)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "learning_curve.png", dpi=160)


def comparison():
    s = pd.read_csv(RES / "summary.csv")
    metrics = [("mean_wait_s", "Mean waiting time (s)"), ("mean_trip_s", "Mean trip time (s)"),
               ("throughput_per_h", "Throughput (veh/h)"), ("co2_g_per_veh", "CO$_2$ per vehicle (g)")]
    scen = [k for k in SCEN if k in set(s.scenario)]
    ctls = [c for c in ORDER if c in set(s.controller)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.2))
    w = 0.8 / len(ctls)
    for ax, (m, label) in zip(axes.flat, metrics):
        for i, ctl in enumerate(ctls):
            vals, errs = [], []
            for sc in scen:
                row = s[(s.scenario == sc) & (s.controller == ctl)].iloc[0]
                vals.append(row[m])
                errs.append(row.get(m + "_std", 0) if ctl.startswith("dqn") else 0)
            ax.bar(np.arange(len(scen)) + (i - (len(ctls) - 1) / 2) * w, vals, w, yerr=errs,
                   capsize=2, label=LABELS[ctl], color=COLORS[ctl])
        ax.set_xticks(range(len(scen)), [SCEN[k] for k in scen], fontsize=8)
        ax.set_title(label, fontsize=10)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "comparison.png", dpi=160)


def wait_by_class():
    w = pd.read_csv(RES / "wait_by_class.csv")
    w = w[w.scenario == "unseen_days"].set_index("family")
    w = w.loc[[c for c in ORDER if c in w.index], C.CLASSES]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    w.T.plot.bar(ax=ax, color=[COLORS[c] for c in w.index], rot=0)
    ax.set_ylabel("Mean waiting time (s)")
    ax.set_title("Waiting time by vehicle class (unseen days)")
    ax.legend([LABELS[c] for c in w.index], frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "wait_by_class.png", dpi=160)


def ablation_plot():
    t = pd.read_csv(RES / "ablation.csv").set_index("family")
    from ablation import LABELS as AL
    t = t.sort_values("mean_wait_s|unseen_days")
    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = np.arange(len(t))
    ax.barh(y - 0.2, t["mean_wait_s|unseen_days"], 0.4, label="1.0x demand", color="#2b6cb0")
    ax.barh(y + 0.2, t["mean_wait_s|heavy_demand_1.3x"], 0.4, label="1.3x demand", color="#9fbfe0")
    ax.set_yticks(y, [AL.get(i, i).replace("--", "-") for i in t.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Mean waiting time on unseen days (s)")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "ablation.png", dpi=160)


def demand_profile():
    from demand import arm_rates, load_counts
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, junction in zip(axes, C.JUNCTION_CAMERAS):
        counts = load_counts(junction)
        for date in C.TRAIN_DATES + C.TEST_DATES:
            r = arm_rates(counts, date, junction)
            tot = r.groupby("time")[C.CLASSES].sum().sum(axis=1)
            test = date in C.TEST_DATES
            ax.plot(tot.index.strftime("%H:%M"), tot.values, lw=1.2,
                    color="#c53030" if test else "#2b6cb0", alpha=0.9 if test else 0.4,
                    label=("test day" if test else "training day"))
        ax.set_title(junction.replace("Chowk", " Chowk"))
        ax.set_xticks(ax.get_xticks()[::6])
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Inbound demand (veh/h)")
    h, lab = axes[0].get_legend_handles_labels()
    axes[0].legend(dict(zip(lab, h)).values(), dict(zip(lab, h)).keys(), frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "demand_profile.png", dpi=160)


if __name__ == "__main__":
    demand_profile()
    learning_curve()
    if (RES / "ablation.csv").exists():
        ablation_plot()
    if (RES / "summary.csv").exists():
        comparison()
        wait_by_class()
    print("plots written to", RES)
