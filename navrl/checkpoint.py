"""Plain-PyTorch checkpoints (.pt) for SB3 PPO policies, rsl_rl style.

A checkpoint is a dict saved with torch.save:
    model_state_dict      actor-critic weights (incl. log_std)
    optimizer_state_dict  Adam state (needed for --resume)
    iteration, num_timesteps
    obs_dim, act_dim, n_stack, task, policy_kwargs, train_args
Loading never needs the original zip or SB3's save format, so a .pt file can also be
used directly from plain torch (policy weights are under model_state_dict).
"""
from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy

# Single source of truth for the network shape (train + load must agree).
POLICY_KWARGS = dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), activation_fn=torch.nn.Tanh,
                     log_std_init=-0.5)


def save_checkpoint(model, path: Path, iteration: int, meta: dict) -> None:
    torch.save({
        "model_state_dict": model.policy.state_dict(),
        "optimizer_state_dict": model.policy.optimizer.state_dict(),
        "iteration": int(iteration),
        "num_timesteps": int(model.num_timesteps),
        "obs_dim": int(model.observation_space.shape[0]),
        "act_dim": int(model.action_space.shape[0]),
        "policy_kwargs": {"net_arch": POLICY_KWARGS["net_arch"], "activation": "tanh",
                          "log_std_init": POLICY_KWARGS["log_std_init"]},
        **meta,
    }, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=False)


def load_policy(path: str | Path, device: str = "cpu") -> tuple[ActorCriticPolicy, dict]:
    """Rebuild the actor-critic from a .pt checkpoint for inference."""
    ckpt = load_checkpoint(path, device)
    obs_space = gym.spaces.Box(-1.0, 1.0, shape=(ckpt["obs_dim"],), dtype=np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, shape=(ckpt["act_dim"],), dtype=np.float32)
    policy = ActorCriticPolicy(obs_space, act_space, lr_schedule=lambda _: 0.0, **POLICY_KWARGS)
    policy.load_state_dict(ckpt["model_state_dict"], strict=True)
    policy.to(device).eval()
    return policy, ckpt


def n_stack_of(ckpt: dict, n_rays: int = 60) -> int:
    return int(ckpt.get("n_stack", (ckpt["obs_dim"] - 7) // n_rays))


def obs_kwargs_of(ckpt: dict) -> dict:
    """EnvConfig fields that must match training for the observation to mean the same thing."""
    return {"n_stack": n_stack_of(ckpt), "stack_gap": int(ckpt.get("stack_gap", 1))}
