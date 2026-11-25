"""Make the figures used in the README (training curves, difficulty curves, suite outcomes).

Usage:
  python scripts/plot_results.py                       # everything it can find, into figures/
  python scripts/plot_results.py --robust results/robust/s3_s4_baseline \\
      --evals s3_best_100 s4_best s4_final gotogoal_100

Inputs it reads (all optional, figures are skipped when their input is missing):
  logs/navrl/curriculum/<stage>/tb/       TensorBoard logs   -> training_curves.png
  results/robust/<name>/summary.json      robust_eval.py     -> difficulty_collisions.png, robustness_success.png
  results/<name>/summary.json             evaluate.py        -> suite_outcomes.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reference palette from the dataviz guide (validated for colour-blind separation).
# Categorical slots are assigned in this fixed order and always follow the same policy.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
GOOD, WARN, CRIT = "#0ca30c", "#fab219", "#d03b3b"     # success / timeout / collision
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9,
    "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 1.6, "lines.markersize": 5,
})


def pretty(name: str) -> str:
    """Readable policy label from a results/checkpoint name."""
    n = name.replace("/", " ").replace("_", " ")
    for a, b in (("best model", "best"), ("model ", "it "), ("gotogoal", "go-to-goal baseline"),
                 ("it1200", "(it 1200)"), (" 100", "")):
        n = n.replace(a, b)
    return n.strip()


def colour_map(names):
    return {n: SERIES[i % len(SERIES)] for i, n in enumerate(names)}


# training curves
def training_curves(run_root: Path, out: Path) -> None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    stages = [d for d in sorted(run_root.iterdir()) if (d / "tb").is_dir()]
    if not stages:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharex=True)
    offset = 0
    for k, st in enumerate(stages):
        cfg = json.loads((st / "config.json").read_text()) if (st / "config.json").exists() else {}
        spi = cfg.get("steps_per_iter", 4096)
        series = {"eval/success": {}, "eval/collision": {}}
        for tb in sorted((st / "tb").iterdir()):
            ea = EventAccumulator(str(tb), size_guidance={"scalars": 0})
            ea.Reload()
            for tag in series:
                if tag in ea.Tags()["scalars"]:
                    for e in ea.Scalars(tag):
                        series[tag][e.step // spi] = e.value     # later runs (resume) overwrite
        its = sorted(series["eval/success"])
        if not its:
            continue
        last = max(its)
        for ax, tag in zip(axes, series):
            x = np.array(its) + offset
            y = np.array([series[tag][i] for i in its])
            # light raw trace + 5-point moving average
            ax.plot(x, y, color=SERIES[0], alpha=0.25, linewidth=1)
            if len(y) >= 5:
                ax.plot(x[2:-2], np.convolve(y, np.ones(5) / 5, mode="valid"), color=SERIES[0])
            ax.axvline(offset, color=INK2, linewidth=0.8, alpha=0.5)
            if last < 150:          # stage 1 is ~100 iterations; its label would collide with s2
                continue
            ax.text(offset + 15, 0.03 if "success" in tag else 0.93, st.name, color=INK2, fontsize=8,
                    transform=ax.get_xaxis_transform())
        offset += last
    axes[0].set_title("Validation success rate")
    axes[1].set_title("Validation collision rate")
    for ax in axes:
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("PPO iteration (stages run back to back, each warm-started from the previous)")
    fig.text(0.01, 0.01, "s1 = first ~100 iterations (empty arena). 30 validation episodes every 10 iterations; line = 5-point moving average, faint = raw",
             color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(out / "training_curves.png", dpi=130)
    plt.close(fig)
    print("wrote", out / "training_curves.png")


# difficulty / robustness curves
SWEEP_LABEL = {
    "obs_speed": "Moving obstacle speed [m/s]", "n_dynamic": "Number of moving obstacles",
    "n_static": "Number of static obstacles", "crossing_speed": "Crossing obstacle speed [m/s]",
    "wander": "Obstacle heading noise [rad/sqrt(s)]", "robot_vmax": "Robot top speed [m/s]",
    "lidar_noise": "LiDAR noise std [m]", "walls": "Interior walls",
}


def _sweep_panel(ax, rows, sweep, policies, cmap, metric, label_ends):
    for pol in policies:
        rs = sorted((r for r in rows if r["sweep"] == sweep and r["policy"] == pol), key=lambda r: r["value"])
        if not rs:
            continue
        x = [r["value"] for r in rs]
        ax.fill_between(x, [r[f"{metric}_lo"] for r in rs], [r[f"{metric}_hi"] for r in rs],
                        color=cmap[pol], alpha=0.12, linewidth=0)
        ax.plot(x, [r[metric] for r in rs], "o-", color=cmap[pol], label=pretty(pol),
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        if label_ends:
            ax.annotate(pretty(pol), (x[-1], rs[-1][metric]), xytext=(4, 0), textcoords="offset points",
                        fontsize=7, color=INK2, va="center")
    ax.set_title(SWEEP_LABEL.get(sweep, sweep), fontsize=9)
    ax.set_ylim(-0.02, 1.02)


def difficulty_curves(robust_dir: Path, out: Path) -> None:
    f = robust_dir / "summary.json"
    if not f.exists():
        return
    rows = json.loads(f.read_text())["rows"]
    policies = list(dict.fromkeys(r["policy"] for r in rows))
    cmap = colour_map(policies)
    n_eps = rows[0]["n"]

    # headline: collision rate as the scene gets harder
    main = [s for s in ("obs_speed", "n_dynamic", "crossing_speed", "walls") if any(r["sweep"] == s for r in rows)]
    fig, axes = plt.subplots(1, len(main), figsize=(3.4 * len(main), 3.3), sharey=True, squeeze=False)
    for ax, sw in zip(axes[0], main):
        _sweep_panel(ax, rows, sw, policies, cmap, "collision", False)
    axes[0][0].set_ylabel("Collision rate")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=len(policies), fontsize=8)
    fig.text(0.01, 0.01, f"{n_eps} test episodes per point, identical scenarios for every policy; "
             f"band = 95% Wilson interval. Lower is better.", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(out / "difficulty_collisions.png", dpi=130)
    plt.close(fig)
    print("wrote", out / "difficulty_collisions.png")

    # all sweeps, success rate
    sweeps = list(dict.fromkeys(r["sweep"] for r in rows))
    cols = 4
    rws = (len(sweeps) + cols - 1) // cols
    fig, axes = plt.subplots(rws, cols, figsize=(13, 3.0 * rws), sharey=True, squeeze=False)
    for ax, sw in zip(axes.flat, sweeps):
        _sweep_panel(ax, rows, sw, policies, cmap, "success", False)
    for ax in list(axes.flat)[len(sweeps):]:
        ax.axis("off")
    for r in range(rws):
        axes[r][0].set_ylabel("Success rate")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=len(policies), fontsize=8)
    fig.text(0.01, 0.01, f"One factor varied at a time around 3 static + 3 moving obstacles at 0.3 m/s. "
             f"{n_eps} episodes per point, band = 95% Wilson interval.", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(out / "robustness_success.png", dpi=130)
    plt.close(fig)
    print("wrote", out / "robustness_success.png")


# suite outcomes
def suite_outcomes(eval_names: list[str], out: Path) -> None:
    data = {}
    for n in eval_names:
        f = ROOT / "results" / n / "summary.json"
        if f.exists():
            data[n] = json.loads(f.read_text())
    if not data:
        return
    def get(d):
        return d["suites"] if "suites" in d else d

    suites = list(dict.fromkeys(k for d in data.values() for k in get(d)))

    fig, axes = plt.subplots(1, len(data), figsize=(2.6 * len(data) + 1.2, 0.34 * len(suites) + 1.6),
                             sharey=True, squeeze=False)
    for ax, (name, d) in zip(axes[0], data.items()):
        s = get(d)
        y = np.arange(len(suites))[::-1]
        left = np.zeros(len(suites))
        for key, col, lab in (("success", GOOD, "success"), ("timeout", WARN, "timeout"), ("collision", CRIT, "collision")):
            v = np.array([s[su][key] if su in s else np.nan for su in suites])
            ax.barh(y, v, left=left, color=col, height=0.7, edgecolor=SURFACE, linewidth=1.5, label=lab)
            left += np.nan_to_num(v)
        for yi, su in zip(y, suites):
            if su in s and s[su]["success"] > 0.15:
                ax.text(0.02, yi, f"{s[su]['success']:.2f}", va="center", fontsize=7, color="white")
        ax.set_yticks(y, suites)
        ax.set_xlim(0, 1)
        ax.set_title(pretty(name), fontsize=9)
        ax.grid(axis="y", visible=False)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=3, fontsize=8)
    fig.text(0.01, 0.01, "Held-out test suites (seeds 100000+). Number = success rate.", color=INK2, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))
    fig.savefig(out / "suite_outcomes.png", dpi=130)
    plt.close(fig)
    print("wrote", out / "suite_outcomes.png")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=str(ROOT / "logs" / "navrl" / "curriculum"))
    ap.add_argument("--robust", default=str(ROOT / "results" / "robust" / "s3_s4_baseline"))
    ap.add_argument("--evals", nargs="*", default=["s3_best_100", "s4_best", "s4_final", "gotogoal_100"])
    ap.add_argument("--out", default=str(ROOT / "figures"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    if Path(args.runs).is_dir():
        training_curves(Path(args.runs), out)
    difficulty_curves(Path(args.robust), out)
    suite_outcomes(args.evals, out)


if __name__ == "__main__":
    main()
