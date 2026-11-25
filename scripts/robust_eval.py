"""Robustness evaluation: sweep one factor at a time and compare policies on identical scenarios.

Usage:
  python scripts/robust_eval.py --models logs/navrl/curriculum/s3/best_model.pt \\
                                         logs/navrl/curriculum/s4/best_model.pt --baseline
  python scripts/robust_eval.py --models ... --episodes 30 --sweeps obs_speed robot_vmax   # quick look
  python scripts/robust_eval.py --models ... --workers 6      # more CPU processes (default 4)

Sweeps (each varies ONE factor around a fixed base scenario; everything else held constant):
  obs_speed       moving-obstacle speed 0.15 .. 0.75 m/s   (3 static + 3 movers)    trained: <= 0.45 (s3), <= 0.6 (s4)
  n_dynamic       number of movers 1 .. 6 at 0.3 m/s
  n_static        static-only arena, 2 .. 8 obstacles
  crossing_speed  one mover aimed at the robot's path, 0.2 .. 0.65 m/s
  wander          mover heading random-walk 0 .. 2 rad/sqrt(s)  (unpredictable pedestrians)
  robot_vmax      robot top speed 0.3 .. 0.8 m/s (trained at 0.5; action a=1 maps to the new top speed)
  lidar_noise     Gaussian LiDAR range noise 0 .. 0.10 m (std)
  walls           interior walls 0 .. 6 (+3 static, 3 wandering movers at 0.3 m/s)

Fairness: every policy sees exactly the same scenarios (same seeds per sweep point, 100000+ range,
never used in training/validation). Success/collision rates come with 95% Wilson confidence
intervals, so you can tell real differences from noise.

Output: results/robust/<name>/summary.csv, summary.json, success.png, collision.png
"""
from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TEST_SEED0 = 100_000
BASE_DYN = dict(n_static=(3, 3), n_dynamic=(3, 3), dyn_speed=(0.3, 0.3), p_crossing=0.0,
                n_walls=(0, 0), dyn_wander=0.0, max_steps=300)

# name -> (label, values, function value -> (cfg overrides, env tweaks))
SWEEPS = {
    "obs_speed": ("obstacle speed [m/s]", [0.15, 0.3, 0.45, 0.6, 0.75],
                  lambda v: ({**BASE_DYN, "dyn_speed": (v, v)}, {})),
    "n_dynamic": ("number of moving obstacles", [1, 2, 4, 6],
                  lambda v: ({**BASE_DYN, "n_dynamic": (v, v)}, {})),
    "n_static": ("number of static obstacles (no movers)", [2, 4, 6, 8],
                 lambda v: ({**BASE_DYN, "n_static": (v, v), "n_dynamic": (0, 0)}, {})),
    "crossing_speed": ("crossing obstacle speed [m/s]", [0.2, 0.35, 0.5, 0.65],
                       lambda v: ({**BASE_DYN, "n_static": (2, 2), "n_dynamic": (1, 1),
                                   "dyn_speed": (v, v), "p_crossing": 1.0}, {})),
    "wander": ("mover heading noise [rad/sqrt(s)]", [0.0, 0.6, 1.2, 2.0],
               lambda v: ({**BASE_DYN, "dyn_wander": v}, {})),
    "robot_vmax": ("robot top speed [m/s]", [0.3, 0.5, 0.65, 0.8],
                   lambda v: (dict(BASE_DYN), {"v_max": v})),
    "lidar_noise": ("LiDAR noise std [m]", [0.0, 0.02, 0.05, 0.1],
                    lambda v: (dict(BASE_DYN), {"lidar_noise": v})),
    "walls": ("interior walls", [0, 2, 4, 6],
              lambda v: ({**BASE_DYN, "n_walls": (v, v), "dyn_wander": 0.6, "max_steps": 450}, {})),
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


# worker side
_POLICIES: dict = {}


def _policy(name: str):
    if name not in _POLICIES:
        if name == "gotogoal":
            from navrl.controllers import GoToGoal
            _POLICIES[name] = (GoToGoal().act, {"n_stack": 1})
        else:
            from navrl.checkpoint import load_policy, n_stack_of, obs_kwargs_of
            pol, ckpt = load_policy(name)
            _POLICIES[name] = ((lambda o, e, p=pol: p.predict(o, deterministic=True)[0]), obs_kwargs_of(ckpt))
    return _POLICIES[name]


def _run_task(task):
    import torch
    torch.set_num_threads(1)
    from navrl.callbacks import run_episodes
    from navrl.config import make_config
    from navrl.env import NavRLEnv

    policy_name, sweep, value, seeds = task
    fn, n_stack = _policy(policy_name)
    overrides, tweaks = SWEEPS[sweep][2](value)
    cfg = make_config("s4_complex", **{**overrides, **n_stack})
    env = NavRLEnv(cfg)
    if "v_max" in tweaks:
        env.robot.p = replace(env.robot.p, v_max=tweaks["v_max"])
    if tweaks.get("lidar_noise", 0) > 0:
        sigma, orig, rng = tweaks["lidar_noise"], env.lidar.scan, np.random.default_rng(seeds[0])
        env.lidar.scan = lambda d, y: np.clip(orig(d, y) + rng.normal(0, sigma, cfg.n_rays), 0, cfg.lidar_range)
    res = run_episodes(fn, cfg, seeds, env)
    env.close()
    return policy_name, sweep, value, [(r["outcome"], r["time_s"], r["min_clearance"]) for r in res]


# main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=[], help=".pt checkpoints to compare")
    ap.add_argument("--baseline", action="store_true", help="also run the obstacle-blind go-to-goal controller")
    ap.add_argument("--sweeps", nargs="+", default=list(SWEEPS), choices=list(SWEEPS))
    ap.add_argument("--episodes", type=int, default=100, help="episodes per sweep point (max 1000)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    policies = list(args.models) + (["gotogoal"] if args.baseline else [])
    if not policies:
        ap.error("give --models and/or --baseline")
    assert args.episodes <= 1000

    def short(p):
        return "gotogoal" if p == "gotogoal" else f"{Path(p).parent.name}/{Path(p).stem}"

    name = args.name or "_vs_".join(short(p).replace("/", "-") for p in policies)
    out = ROOT / "results" / "robust" / name
    out.mkdir(parents=True, exist_ok=True)

    chunk = 25
    tasks = []
    for si, sweep in enumerate(SWEEPS):
        if sweep not in args.sweeps:
            continue
        for pi, val in enumerate(SWEEPS[sweep][1]):
            s0 = TEST_SEED0 + si * 10_000 + pi * 1000       # same seeds for every policy
            for c in range(0, args.episodes, chunk):
                seeds = list(range(s0 + c, s0 + min(c + chunk, args.episodes)))
                for pol in policies:
                    tasks.append((pol, sweep, val, seeds))
    print(f"{len(policies)} policies x {len(args.sweeps)} sweeps, {args.episodes} episodes/point, "
          f"{len(tasks)} tasks on {args.workers} workers -> {out}", flush=True)

    raw: dict = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for i, (pol, sweep, val, eps) in enumerate(pool.imap_unordered(_run_task, tasks), 1):
            raw.setdefault((pol, sweep, val), []).extend(eps)
            if i % max(1, len(tasks) // 20) == 0 or i == len(tasks):
                el = time.time() - t0
                print(f"  {i}/{len(tasks)} tasks  {el / 60:4.1f} min  ETA {el / i * (len(tasks) - i) / 60:4.1f} min",
                      flush=True)

    rows = []
    for sweep in args.sweeps:
        for val in SWEEPS[sweep][1]:
            for pol in policies:
                eps = raw[(pol, sweep, val)]
                n = len(eps)
                k_s = sum(o == "success" for o, _, _ in eps)
                k_c = sum(o == "collision" for o, _, _ in eps)
                t_s = [t for o, t, _ in eps if o == "success"]
                lo, hi = wilson(k_s, n)
                clo, chi = wilson(k_c, n)
                rows.append(dict(sweep=sweep, value=val, policy=short(pol), n=n,
                                 success=k_s / n, success_lo=lo, success_hi=hi,
                                 collision=k_c / n, collision_lo=clo, collision_hi=chi,
                                 timeout=1 - (k_s + k_c) / n,
                                 time_to_goal=float(np.mean(t_s)) if t_s else float("nan"),
                                 min_clearance=float(np.mean([c for o, _, c in eps if o != "collision"] or [np.nan]))))

    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    (out / "summary.json").write_text(json.dumps(dict(policies=policies, episodes=args.episodes, rows=rows), indent=1))

    # console tables
    for sweep in args.sweeps:
        print(f"\n{sweep}: {SWEEPS[sweep][0]}   (success [95% CI] / collision)")
        print(f"{'value':>8} " + " ".join(f"{short(p):>30}" for p in policies))
        for val in SWEEPS[sweep][1]:
            cells = []
            for pol in policies:
                r = next(r for r in rows if r["sweep"] == sweep and r["value"] == val and r["policy"] == short(pol))
                cells.append(f"{r['success']:.2f} [{r['success_lo']:.2f}-{r['success_hi']:.2f}] / {r['collision']:.2f}")
            print(f"{val:>8} " + " ".join(f"{c:>30}" for c in cells))

    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in ("success", "collision"):
        n = len(args.sweeps)
        cols = min(4, n)
        rws = (n + cols - 1) // cols
        fig, axes = plt.subplots(rws, cols, figsize=(4.2 * cols, 3.4 * rws), squeeze=False)
        for ax, sweep in zip(axes.flat, args.sweeps):
            for pol in policies:
                rs = [r for r in rows if r["sweep"] == sweep and r["policy"] == short(pol)]
                x = [r["value"] for r in rs]
                ax.plot(x, [r[metric] for r in rs], "o-", label=short(pol))
                ax.fill_between(x, [r[f"{metric}_lo"] for r in rs], [r[f"{metric}_hi"] for r in rs], alpha=0.15)
            ax.set_title(sweep, fontsize=10); ax.set_xlabel(SWEEPS[sweep][0], fontsize=8)
            ax.set_ylim(-0.02, 1.02); ax.grid(alpha=0.3)
        for ax in list(axes.flat)[n:]:
            ax.axis("off")
        axes.flat[0].set_ylabel(f"{metric} rate (95% CI)")
        axes.flat[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"{metric}.png", dpi=110)
        plt.close(fig)
    print(f"\nwrote {out}/summary.csv, summary.json, success.png, collision.png  ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
