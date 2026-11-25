"""Core sanity tests for NavRL. Run:  python -m pytest -q tests/"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from navrl.config import make_config  # noqa: E402
from navrl.controllers import GoToGoal  # noqa: E402
from navrl.env import NavRLEnv  # noqa: E402
from navrl.robot import RobotParams, forward_kinematics, inverse_kinematics  # noqa: E402


def test_kinematics_roundtrip():
    p = RobotParams()
    for v, w in [(0.3, 0.0), (0.0, 1.0), (0.4, -1.2), (-0.2, 0.5)]:
        assert np.allclose(forward_kinematics(*inverse_kinematics(v, w, p), p), (v, w))


@pytest.mark.parametrize("stage,n_stack", [("s1_empty", 1), ("s2_static", 1), ("s3_dynamic", 3)])
def test_gymnasium_check_env(stage, n_stack):
    env = NavRLEnv(make_config(stage, n_stack=n_stack))
    check_env(env, skip_render_check=True)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (60 * n_stack + 7,)


def test_lidar_matches_analytic_walls():
    """Robot at the arena centre, empty arena: wall at 3.0 m along axes; range clipped at 3.0."""
    env = NavRLEnv(make_config("s1_empty", lidar_range=5.0))
    env.reset(seed=0)
    env.robot.set_pose(env.data, 0.0, 0.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    r = env.lidar.scan(env.data, 0.0)
    ang = env.lidar.angles
    # distance from centre to a square of half-size 3 along each ray
    expected = 3.0 / np.maximum(np.abs(np.cos(ang)), np.abs(np.sin(ang)))
    expected = np.minimum(expected, 5.0)
    assert np.max(np.abs(r - expected)) < 1e-3


def test_lidar_sees_obstacle_not_self():
    env = NavRLEnv(make_config("s1_empty"))
    env.reset(seed=0)
    env.robot.set_pose(env.data, 0.0, 0.0, 0.0)
    o = env.obstacles.static[0]                 # cylinder r = 0.2
    o.pos[:] = (1.0, 0.0); o.active = True
    env.obstacles.write(env.data)
    mujoco.mj_forward(env.model, env.data)
    r = env.lidar.scan(env.data, 0.0)
    assert abs(r[0] - 0.8) < 1e-3               # ray 0 points forward, hits surface at 0.8 m
    assert r.min() > 0.5                        # never hits the robot itself


def test_collision_terminates():
    env = NavRLEnv(make_config("s1_empty"))
    env.reset(seed=0)
    env.robot.set_pose(env.data, 2.3, 0.0, 0.0)     # 0.55 m from east wall, facing it
    env.goal[:] = (-2.0, 0.0)
    mujoco.mj_forward(env.model, env.data)
    env._prev_dist, _ = env._goal_polar()
    for _ in range(50):
        _, r, term, trunc, info = env.step(np.array([1.0, 0.0]))
        if term:
            break
    assert term and info["outcome"] == "collision" and r < -5


def test_determinism():
    def rollout(seed):
        env = NavRLEnv(make_config("s3_dynamic"))
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(1)
        out = [obs]
        for _ in range(30):
            obs, *_ = env.step(rng.uniform(-1, 1, 2))
            out.append(obs)
        return np.array(out)
    assert np.array_equal(rollout(7), rollout(7))


def test_timeout_is_truncation():
    env = NavRLEnv(make_config("s1_empty", max_steps=5))
    env.reset(seed=0)
    for _ in range(5):
        _, _, term, trunc, info = env.step(np.zeros(2))
    assert trunc and not term and info["outcome"] == "timeout"


def test_go_to_goal_empty_arena():
    env = NavRLEnv(make_config("s1_empty"))
    ctrl = GoToGoal()
    n, wins, progress_err = 50, 0, []
    for ep in range(n):
        obs, info = env.reset(seed=1000 + ep)
        total_prog = 0.0
        while True:
            obs, r, term, trunc, inf = env.step(ctrl.act(obs, env))
            total_prog += inf["reward_terms"]["progress"]
            if term or trunc:
                break
        wins += inf["outcome"] == "success"
        if inf["outcome"] == "success":
            progress_err.append(abs(total_prog - (info["start_dist"] - inf["goal_dist"])))
    assert wins / n >= 0.98
    assert max(progress_err) < 1e-6
