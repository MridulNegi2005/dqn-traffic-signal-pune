import glob

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import config as C  # noqa: E402

RES = C.ROOT / "results"
COLORS = {"fixed": "#8a8f98", "actuated": "#d9822b", "dqn": "#2b6cb0"}


def learning_curve(run="dqn_pune"):
    files = glob.glob(str(RES / "runs" / run / "monitor" / "*.monitor.csv"))
    if not files:
        return
    df = pd.concat(pd.read_csv(f, skiprows=1) for f in files).sort_values("t")
    df["episode"] = range(1, len(df) + 1)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(df.episode, df.r, alpha=0.25, color=COLORS["dqn"])
    ax.plot(df.episode, df.r.rolling(20, min_periods=1).mean(), color=COLORS["dqn"], lw=2)
    ax.set_xlabel("Training episode (1 hour of Pune traffic)")
    ax.set_ylabel("Episode return")
    ax.set_title("DQN learning curve")
    fig.tight_layout()
    fig.savefig(RES / "learning_curve.png", dpi=160)


def comparison():
    s = pd.read_csv(RES / "summary.csv")
    metrics = [("mean_wait_s", "Mean waiting time (s)"), ("mean_delay_s", "Mean delay (s)"),
               ("pcu_weighted_wait_s", "PCU-weighted waiting (s)")]
    scales = sorted(s.scale.unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(11, 3.6))
    w = 0.25
    for ax, (m, label) in zip(axes, metrics):
        for i, ctl in enumerate(["fixed", "actuated", "dqn"]):
            vals = [s[(s.controller == ctl) & (s.scale == sc)][m].iloc[0] for sc in scales]
            ax.bar([x + (i - 1) * w for x in range(len(scales))], vals, w,
                   label=ctl, color=COLORS[ctl])
        ax.set_xticks(range(len(scales)), [f"{sc:g}x demand" for sc in scales])
        ax.set_title(label, fontsize=10)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "comparison.png", dpi=160)


def wait_by_class():
    w = pd.read_csv(RES / "wait_by_class.csv")
    w = w[w.scale == w.scale.min()].set_index("controller")[C.CLASSES]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    w.T.plot.bar(ax=ax, color=[COLORS[c] for c in w.index], rot=0)
    ax.set_ylabel("Mean waiting time (s)")
    ax.set_title("Waiting time by vehicle class (real demand, test days)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "wait_by_class.png", dpi=160)


if __name__ == "__main__":
    learning_curve()
    comparison()
    wait_by_class()
    print("plots written to", RES)
