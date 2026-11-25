"""Quick env-throughput benchmark (no learning).

Usage:  python scripts/benchmark.py
Measures single-process env.step()/s per stage, and SubprocVecEnv scaling on s3.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from navrl.config import make_config  # noqa: E402
from navrl.env import NavRLEnv  # noqa: E402


def single(stage: str, n_steps: int = 3000, **kw) -> float:
    env = NavRLEnv(make_config(stage, **kw))
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    t = time.perf_counter()
    for _ in range(n_steps):
        _, _, term, trunc, _ = env.step(rng.uniform(-1, 1, 2))
        if term or trunc:
            env.reset()
    return n_steps / (time.perf_counter() - t)


def make_env(stage):
    def _f():
        return NavRLEnv(make_config(stage))
    return _f


def vec(n_envs: int, stage: str = "s3_dynamic", n_steps: int = 2000) -> float:
    from stable_baselines3.common.vec_env import SubprocVecEnv
    venv = SubprocVecEnv([make_env(stage) for _ in range(n_envs)])
    venv.seed(0); venv.reset()
    acts = np.random.default_rng(0).uniform(-1, 1, (n_steps, n_envs, 2))
    t = time.perf_counter()
    for i in range(n_steps):
        venv.step(acts[i])
    sps = n_steps * n_envs / (time.perf_counter() - t)
    venv.close()
    return sps


if __name__ == "__main__":
    print(f"CPU logical cores: {os.cpu_count()}")
    for st in ("s1_empty", "s2_static", "s3_dynamic"):
        print(f"single env {st:<11} {single(st):8.0f} steps/s")
    print(f"single env s3 n_stack=3 {single('s3_dynamic', n_stack=3):8.0f} steps/s")
    for n in (2, 4, 6, 8):
        print(f"SubprocVecEnv x{n} (s3)  {vec(n):8.0f} steps/s")
