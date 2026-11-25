"""Evaluate a trained PPO model (or the GoToGoal baseline) on fixed held-out seeds.

Examples:
  python scripts/evaluate.py --model logs/navrl/curriculum/s3/best_model.pt
  python scripts/evaluate.py --controller gotogoal
  python scripts/evaluate.py --model logs/navrl/curriculum/s2/best_model.pt --suites empty static --episodes 50

Test seeds start at 100000 (training uses seeds < 10000, in-training validation 50000+),
so test episodes are never seen during training. Every policy sees identical episodes.
Writes results/<name>/summary.json, episodes.json and trajectory plots.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from navrl.callbacks import run_episodes, summarize  # noqa: E402
from navrl.config import make_config  # noqa: E402
from navrl.controllers import GoToGoal  # noqa: E402
from navrl.viz import plot_trajectories  # noqa: E402

# Small core suite (docs/DESIGN.md 13.1, reduced)
SUITES = {
    "empty":      ("s1_empty",   {}),
    "static":     ("s2_static",  dict(n_static=(6, 6))),
    "dyn_slow":   ("s3_dynamic", dict(n_static=(3, 3), n_dynamic=(3, 3), dyn_speed=(0.15, 0.15), p_crossing=0.0)),
    "dyn_medium": ("s3_dynamic", dict(n_static=(3, 3), n_dynamic=(3, 3), dyn_speed=(0.30, 0.30), p_crossing=0.0)),
    "dyn_fast":   ("s3_dynamic", dict(n_static=(3, 3), n_dynamic=(3, 3), dyn_speed=(0.45, 0.45), p_crossing=0.0)),
    "crossing":   ("s3_dynamic", dict(n_static=(2, 2), n_dynamic=(2, 2), dyn_speed=(0.30, 0.30), p_crossing=1.0)),
    "walls":      ("s4_complex", dict(n_walls=(4, 4), n_static=(3, 3), n_dynamic=(0, 0))),
    "complex":    ("s4_complex", dict(n_walls=(4, 4), n_static=(3, 3), n_dynamic=(4, 4), dyn_speed=(0.4, 0.4))),
    "crowd":      ("s4_complex", dict(n_walls=(2, 2), n_static=(2, 2), n_dynamic=(6, 6), dyn_speed=(0.5, 0.5))),
}
TEST_SEED0 = 100_000


def load_policy(args):
    if args.controller == "gotogoal":
        ctrl = GoToGoal()
        return ctrl.act, {"n_stack": 1}, args.name or "gotogoal"
    from navrl.checkpoint import load_policy as load_pt, n_stack_of, obs_kwargs_of
    policy, ckpt = load_pt(args.model)
    name = args.name or Path(args.model).parent.name + "_" + Path(args.model).stem
    return (lambda o, e: policy.predict(o, deterministic=True)[0]), obs_kwargs_of(ckpt), name


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=str)
    g.add_argument("--controller", choices=["gotogoal"])
    ap.add_argument("--suites", nargs="+", default=list(SUITES), choices=list(SUITES))
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--name", type=str, default=None)
    args = ap.parse_args()

    policy, n_stack, name = load_policy(args)
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    seeds = range(TEST_SEED0, TEST_SEED0 + args.episodes)

    summary, episodes = {}, {}
    hdr = f"{'suite':<11}{'success':>9}{'collide':>9}{'timeout':>9}{'t_goal[s]':>11}{'path[m]':>9}{'ratio':>7}{'min_clr[m]':>11}"
    print(f"policy: {name}  (n_stack={n_stack}, {args.episodes} episodes/suite)\n{hdr}")
    for suite in args.suites:
        stage, over = SUITES[suite]
        cfg = make_config(stage, **n_stack, **over)
        res = run_episodes(policy, cfg, seeds)
        s = summarize(res)
        summary[suite], episodes[suite] = s, res
        print(f"{suite:<11}{s['success']:>9.2f}{s['collision']:>9.2f}{s['timeout']:>9.2f}"
              f"{s['time_to_goal']:>11.1f}{s['path_len']:>9.2f}{s['path_ratio']:>7.2f}{s['min_clearance']:>11.2f}")
        # show a mix of outcomes in the plot
        picks = sorted(res, key=lambda r: r["outcome"] != "collision")[:2] + \
            [r for r in res if r["outcome"] == "success"][:4]
        plot_trajectories(picks or res, out / f"traj_{suite}.png", f"{name} - {suite}")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "episodes.json").write_text(json.dumps(episodes))
    print(f"\nwrote {out}/summary.json, episodes.json, traj_*.png")


if __name__ == "__main__":
    main()
