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
ORDER = ["fixed", "actuated", "camera_rule", "dqn_camera"]
LABELS = {"fixed": "Fixed-time", "actuated": "Actuated (loop sensors)", "camera_rule": "Camera rule",
          "dqn_camera": "DQN, camera view", "dqn_full": "DQN, full view (3 seeds)"}
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
    """Average inbound demand per road over the morning, and the vehicle mix."""
    from demand import arm_rates, load_counts
    junction = C.TRAIN_JUNCTION
    counts = load_counts(junction)
    days = []
    for date in C.TRAIN_DATES + C.TEST_DATES:
        r = arm_rates(counts, date, junction)
        r = r[(r.time.dt.strftime("%H:%M") >= "09:00") & (r.time.dt.strftime("%H:%M") <= "11:50")]
        r["clock"] = r.time.dt.floor("15min").dt.strftime("%H:%M")
        r["total"] = r[C.CLASSES].sum(axis=1)
        days.append(r)
    d = pd.concat(days)
    per_road = d.groupby(["clock", "arm"]).total.mean().unstack("arm")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1.6, 1]})
    names = {"N": "North (camera j1)", "E": "East (camera j2)", "S": "South (camera j3)",
             "W": "West (minor road)"}
    shades = {"N": "#1f4e79", "E": "#2e75b6", "S": "#9dc3e6", "W": "#bfbfbf"}
    bottom = np.zeros(len(per_road))
    for arm in ["N", "E", "S", "W"]:
        a1.bar(per_road.index, per_road[arm], bottom=bottom, color=shades[arm], label=names[arm], width=0.8)
        bottom += per_road[arm].to_numpy()
    a1.set_ylabel("Vehicles per hour")
    a1.set_title("Inbound demand at Jehangir Chowk (mean of 8 days)", fontsize=10)
    a1.tick_params(axis="x", labelsize=8, rotation=45)
    a1.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.22))

    mix = d[C.CLASSES].sum()
    mix = 100 * mix / mix.sum()
    colors = ["#2e75b6", "#ed7d31", "#70ad47", "#7f7f7f"]
    a2.barh(["Car", "Motorbike", "Bus", "Truck"], mix.values, color=colors)
    for i, v in enumerate(mix.values):
        a2.text(v + 1, i, f"{v:.0f}%", va="center", fontsize=9)
    a2.invert_yaxis()
    a2.set_xlim(0, 65)
    a2.set_xlabel("Share of vehicles (%)")
    a2.set_title("Vehicle mix", fontsize=10)
    for ax in (a1, a2):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(RES / "demand_profile.png", dpi=160)


def junction_diagram():
    """Top view of the simulated junction: roads, lanes, camera zone and signal order."""
    from matplotlib.patches import FancyArrowPatch, Rectangle
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    L, W = 10.0, 1.6
    road = "#d9d9d9"
    ax.add_patch(Rectangle((-W, -L), 2 * W, 2 * L, color=road, zorder=0))
    ax.add_patch(Rectangle((-L, -W), 2 * L, 2 * W, color=road, zorder=0))
    ax.plot([0, 0], [W, L], color="white", lw=1.5, ls=(0, (6, 4)))
    ax.plot([0, 0], [-L, -W], color="white", lw=1.5, ls=(0, (6, 4)))
    ax.plot([W, L], [0, 0], color="white", lw=1.5, ls=(0, (6, 4)))
    ax.plot([-L, -W], [0, 0], color="white", lw=1.5, ls=(0, (6, 4)))
    cam = 75 / 250 * (L - W)
    zones = [(0, W, W, cam), (W, -W, cam, W), (-W, -W - cam, W, cam), (-W - cam, 0, cam, W)]
    for x, y, w, h in zones:
        ax.add_patch(Rectangle((x, y), w, h, color="#f4b183", alpha=0.6, zorder=1))
    ax.text(W + 0.15, W + cam / 2, "camera view\n(last 75 m)", fontsize=8, va="center", color="#843c0c")
    # left-hand traffic: vehicles from the north drive on the east half of the north road
    arrows = [((0.8, 6.5), (0.8, 2.3)), ((-6.5, 0.8), (-2.3, 0.8)),
              ((-0.8, -6.5), (-0.8, -2.3)), ((6.5, -0.8), (2.3, -0.8))]
    for a, b in arrows:
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, color="#1f4e79", lw=1.6))
    labels = {"N": (0, L + 0.6), "S": (0, -L - 0.8), "E": (L + 0.7, 0), "W": (-L - 0.9, 0)}
    roads = {"N": "North\n(camera j1)", "S": "South\n(camera j3)", "E": "East\n(camera j2)",
             "W": "West\n(minor road)"}
    for k, (x, y) in labels.items():
        ax.text(x, y, roads[k], ha="center", va="center", fontsize=9, fontweight="bold")
    ax.add_patch(Rectangle((-W, -W), 2 * W, 2 * W, fill=False, ec="#c00000", lw=1.5))
    ax.text(0, 0, "signal", ha="center", va="center", fontsize=8, color="#c00000")
    ax.text(W + 0.4, -L + 0.2,
            "Signal plan: one road at a time\n"
            "green 10-60 s, yellow 3 s,\n"
            "all-red 2 s, then next road\n\n"
            "Lanes per road:\n"
            "left | straight | straight + right",
            fontsize=8, va="bottom")
    ax.set_xlim(-L - 1.8, L + 1.8)
    ax.set_ylim(-L - 1.6, L + 1.4)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(RES / "junction.png", dpi=170)


if __name__ == "__main__":
    demand_profile()
    junction_diagram()
    learning_curve()
    if (RES / "ablation.csv").exists():
        ablation_plot()
    if (RES / "summary.csv").exists():
        comparison()
        wait_by_class()
    print("plots written to", RES)
