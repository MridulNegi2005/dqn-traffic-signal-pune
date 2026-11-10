import glob

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402

RES = C.ROOT / "results"
COLORS = {"fixed": "#8a8f98", "actuated": "#d9822b", "dqn": "#2b6cb0"}
ORDER = ["fixed", "actuated", "dqn"]
LABELS = {"fixed": "Fixed-time", "actuated": "Actuated", "dqn": "DQN (3 seeds)"}
SCEN = {"unseen_days": "Unseen days", "heavy_demand_1.3x": "Unseen days,\n1.3x demand",
        "unseen_junction_RTO": "Unseen junction\n(RTO Chowk)"}


def learning_curve():
    fig, ax = plt.subplots(figsize=(7, 3.5))
    curves = []
    dirs = glob.glob(str(RES / "monitor" / "dqn_seed*_monitor"))
    dirs += glob.glob(str(RES / "runs" / "dqn_seed*" / "monitor"))
    for d in sorted(dirs):
        files = glob.glob(d + "/*.monitor.csv")
        df = pd.concat(pd.read_csv(f, skiprows=1) for f in files).sort_values("t")
        r = df.r.rolling(20, min_periods=1).mean().to_numpy()
        curves.append(r)
        ax.plot(np.arange(1, len(r) + 1), r, lw=1, alpha=0.5, color=COLORS["dqn"])
    if not curves:
        return
    n = min(len(c) for c in curves)
    ax.plot(np.arange(1, n + 1), np.mean([c[:n] for c in curves], axis=0),
            lw=2.2, color=COLORS["dqn"], label="mean of seeds")
    ax.set_xlabel("Training episode (1 hour of Pune traffic, random demand 0.7-1.4x)")
    ax.set_ylabel("Episode return (20-ep. mean)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "learning_curve.png", dpi=160)


def comparison():
    s = pd.read_csv(RES / "summary.csv")
    metrics = [("mean_wait_s", "Mean waiting time (s)"),
               ("mean_trip_s", "Mean trip time (s)"),
               ("throughput_per_h", "Throughput (veh/h)"),
               ("co2_g_per_veh", "CO$_2$ per vehicle (g)")]
    scen = [k for k in SCEN if k in set(s.scenario)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.8))
    w = 0.26
    for ax, (m, label) in zip(axes, metrics):
        for i, ctl in enumerate(ORDER):
            vals, errs = [], []
            for sc in scen:
                row = s[(s.scenario == sc) & (s.controller == ctl)].iloc[0]
                vals.append(row[m])
                errs.append(row.get(m + "_std", 0) if ctl == "dqn" else 0)
            ax.bar(np.arange(len(scen)) + (i - 1) * w, vals, w, yerr=errs, capsize=3,
                   label=LABELS[ctl], color=COLORS[ctl])
        ax.set_xticks(range(len(scen)), [SCEN[k] for k in scen], fontsize=8)
        ax.set_title(label, fontsize=10)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / "comparison.png", dpi=160)


def wait_by_class():
    w = pd.read_csv(RES / "wait_by_class.csv")
    w = w[w.scenario == "unseen_days"].set_index("family").loc[ORDER, C.CLASSES]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    w.T.plot.bar(ax=ax, color=[COLORS[c] for c in w.index], rot=0)
    ax.set_ylabel("Mean waiting time (s)")
    ax.set_title("Waiting time by vehicle class (unseen days)")
    ax.legend([LABELS[c] for c in w.index], frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "wait_by_class.png", dpi=160)


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
    comparison()
    wait_by_class()
    print("plots written to", RES)
