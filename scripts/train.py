#!/usr/bin/env python3
"""Train a NavRL PPO policy (Stable-Baselines3), rsl_rl-style iterations CLI.

One iteration = num_envs x num_steps_per_env env steps (one rollout) + one PPO update.

    python scripts/train.py --task s1_empty --num_envs 4 --max_iterations 100
    python scripts/train.py --task s2_static --max_iterations 300 \
        --warm_start logs/navrl/curriculum/s1/best_model.pt
    python scripts/train.py --task s2_static --max_iterations 400 --resume true \
        --load_run logs/navrl/curriculum/s2 --checkpoint model_150.pt

--warm_start  copies actor + critic weights only (fresh optimizer, iteration counter 0):
              use it to move to the next curriculum stage.
--resume      continues the SAME run (weights, optimizer, iteration count).
Both require the same observation width (same --n_stack); loading is strict.

Output: logs/navrl/<experiment_name>/<run_name>/
    model_<iter>.pt (every --save_interval its + final), best_model.pt (best validation
    success), tb/ (TensorBoard), config.json. Ctrl+C saves model_<iter>.pt before exiting.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from navrl.callbacks import IterationCallback  # noqa: E402
from navrl.checkpoint import POLICY_KWARGS, load_checkpoint, save_checkpoint  # noqa: E402
from navrl.config import STAGES, make_config  # noqa: E402
from navrl.env import NavRLEnv  # noqa: E402


def _str2bool(v: str) -> bool:
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "y", "t"):
        return True
    if s in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"expected true/false, got {v!r}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a NavRL PPO policy.")
    p.add_argument("--task", choices=list(STAGES), required=True, help="curriculum stage")
    p.add_argument("--num_envs", type=int, default=4,
                   help="parallel envs (one CPU process each; 4 is the benchmarked sweet spot)")
    p.add_argument("--max_iterations", type=int, required=True)
    p.add_argument("--num_steps_per_env", type=int, default=1024, help="rollout length per env")
    p.add_argument("--num_mini_batches", type=int, default=16)
    p.add_argument("--num_learning_epochs", type=int, default=10)
    p.add_argument("--learning_rate", type=float, default=3e-4)
    p.add_argument("--entropy_coef", type=float, default=0.0)
    p.add_argument("--clip_param", type=float, default=0.2)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--n_stack", type=int, default=3, help="LiDAR scans in the observation (1 or 3)")
    p.add_argument("--physics_dt", type=float, default=0.01,
                   help="physics substep [s]; 0.02 = ~1.8x faster env, <1 cm trajectory difference over 15 s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--experiment_name", type=str, default="curriculum")
    p.add_argument("--run_name", type=str, default=None)
    p.add_argument("--save_interval", type=int, default=25, help="iterations between model_<iter>.pt")
    p.add_argument("--eval_interval", type=int, default=10, help="iterations between validation evals")
    p.add_argument("--eval_episodes", type=int, default=30)
    p.add_argument("--warm_start", type=str, default=None, metavar="model.pt")
    p.add_argument("--resume", type=_str2bool, default=False, metavar="true|false")
    p.add_argument("--load_run", type=str, default=None, help="run dir to resume (with --resume true)")
    p.add_argument("--checkpoint", type=str, default=None,
                   help="checkpoint file in --load_run (default: highest model_<iter>.pt)")
    p.add_argument("--device", type=str, default="cpu", help="cpu recommended for small MLPs")
    return p


def latest_checkpoint(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda f: int(f.stem.split("_")[1]))
    if not ckpts:
        raise SystemExit(f"[ERROR] no model_<iter>.pt in {run_dir}")
    return ckpts[-1]


def main() -> None:
    args = build_parser().parse_args()
    if args.resume and not args.load_run:
        raise SystemExit("[ERROR] --resume true needs --load_run <run dir>")
    if args.resume and args.warm_start:
        raise SystemExit("[ERROR] use either --resume or --warm_start, not both")

    if args.resume:
        run_dir = Path(args.load_run).resolve()
    else:
        name = args.run_name or f"{args.task}_{datetime.now():%Y%m%d_%H%M%S}"
        run_dir = ROOT / "logs" / "navrl" / args.experiment_name / name
    run_dir.mkdir(parents=True, exist_ok=True)

    steps_per_iter = args.num_envs * args.num_steps_per_env
    batch_size = steps_per_iter // args.num_mini_batches
    cfg = make_config(args.task, n_stack=args.n_stack, physics_dt=args.physics_dt)
    (run_dir / ("config_resume.json" if args.resume else "config.json")).write_text(
        json.dumps({"args": vars(args), "steps_per_iter": steps_per_iter, "batch_size": batch_size,
                    "env": asdict(cfg)}, indent=2))

    set_random_seed(args.seed)
    torch.set_num_threads(2)
    vec_cls = SubprocVecEnv if args.num_envs > 1 else DummyVecEnv
    venv = make_vec_env(NavRLEnv, n_envs=args.num_envs, seed=args.seed,
                        env_kwargs={"config": cfg}, vec_env_cls=vec_cls)

    model = PPO("MlpPolicy", venv, device=args.device, seed=args.seed, verbose=0,
                learning_rate=args.learning_rate, n_steps=args.num_steps_per_env,
                batch_size=batch_size, n_epochs=args.num_learning_epochs, gamma=args.gamma,
                gae_lambda=args.lam, clip_range=args.clip_param, ent_coef=args.entropy_coef,
                vf_coef=0.5, max_grad_norm=args.max_grad_norm, policy_kwargs=POLICY_KWARGS,
                tensorboard_log=str(run_dir / "tb"))
    done = 0
    if args.resume:
        ckpt_path = run_dir / args.checkpoint if args.checkpoint else latest_checkpoint(run_dir)
        ckpt = load_checkpoint(ckpt_path, args.device)
        model.policy.load_state_dict(ckpt["model_state_dict"], strict=True)
        model.policy.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        model.num_timesteps = ckpt["num_timesteps"]
        done = ckpt["iteration"]
        print(f"[resume] {ckpt_path} ({done} iterations done)")
    elif args.warm_start:
        ckpt = load_checkpoint(args.warm_start, args.device)
        # strict: fails immediately if observation/action widths differ
        model.policy.load_state_dict(ckpt["model_state_dict"], strict=True)
        print(f"[warm_start] actor+critic weights from {args.warm_start} "
              f"(fresh optimizer, iteration 0)")

    meta = {"task": args.task, "n_stack": args.n_stack, "stack_gap": cfg.stack_gap, "train_args": vars(args)}
    remaining = max(args.max_iterations - done, 0)
    print(f"[train] task={args.task} n_stack={args.n_stack} num_envs={args.num_envs} "
          f"steps/iter={steps_per_iter} batch={batch_size} iterations {done}->{args.max_iterations}\n"
          f"[train] logs: {run_dir}", flush=True)

    cb = IterationCallback(cfg, run_dir, steps_per_iter, args.max_iterations,
                           save_interval=args.save_interval, eval_interval=args.eval_interval,
                           eval_episodes=args.eval_episodes, meta=meta)
    if args.resume and (run_dir / "best_model.pt").exists():
        ev = load_checkpoint(run_dir / "best_model.pt").get("eval")
        if ev:   # keep the previous best so a worse model can't overwrite best_model.pt
            t = ev["time_to_goal"] if ev["time_to_goal"] == ev["time_to_goal"] else 1e9
            cb.best = (ev["success"], -ev["collision"], -t)
    try:
        model.learn(total_timesteps=remaining * steps_per_iter, callback=cb,
                    reset_num_timesteps=not args.resume, tb_log_name="tb", progress_bar=False)
    except KeyboardInterrupt:
        it = model.num_timesteps // steps_per_iter
        save_checkpoint(model, run_dir / f"model_{it}.pt", it, meta)
        print(f"\n[interrupted] saved {run_dir / f'model_{it}.pt'}")
    finally:
        venv.close()
        best = f"{cb.best[0]:.2f}" if cb.best[0] >= 0 else "n/a (no eval ran)"
        print(f"[done] best validation success {best} -> {run_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
