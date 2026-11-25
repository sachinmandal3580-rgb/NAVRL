"""NavRLEnv: Gymnasium environment for goal navigation among obstacles.

Observation (float32), all components normalised to [-1, 1] or [0, 1]:
    [ scan_t, scan_{t-1}, ..., scan_{t-k+1}   (n_rays * n_stack, range / max_range)
      goal_dist / arena_diag, sin(goal_bearing), cos(goal_bearing),
      v / v_max, w / w_max,
      a_{t-1} (2) ]
Action: a in [-1, 1]^2 -> (v, w) -> accel limit -> wheel IK/saturation -> base servos.

Episode end (checked after each 0.1 s control step, in this priority order):
    collision (any chassis contact during any substep)  -> terminated, outcome 'collision'
    goal distance < goal_radius                         -> terminated, outcome 'success'
    step count == max_steps                             -> truncated,  outcome 'timeout'
'terminated' means a true MDP end (no bootstrapping); 'truncated' means a time limit,
so PPO bootstraps the value of the last state (SB3 handles this via TimeLimit.truncated).
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import mujoco
import numpy as np

from navrl import ARENA_XML
from navrl.config import EnvConfig
from navrl.lidar import Lidar
from navrl.obstacles import ObstacleManager
from navrl.robot import RobotHandle, denormalize_action


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2 * np.pi) - np.pi


class NavRLEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, config: EnvConfig | None = None, render_mode: str | None = None):
        self.cfg = config or EnvConfig()
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(str(ARENA_XML))
        self.model.opt.timestep = self.cfg.physics_dt
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(self.cfg.ctrl_dt / self.cfg.physics_dt))

        self.robot = RobotHandle(self.model)
        self.lidar = Lidar(self.model, self.cfg.n_rays, self.cfg.lidar_range)
        self.obstacles = ObstacleManager(self.model, self.cfg.arena_half, self.robot.p.body_radius)
        self.goal_mocap = int(self.model.body_mocapid[self.model.body("goal").id])
        self.chassis_geom = self.model.geom("chassis").id
        self.arena_diag = 2 * np.sqrt(2) * self.cfg.arena_half

        obs_dim = self.cfg.n_rays * self.cfg.n_stack + 3 + 2 + 2
        low = np.full(obs_dim, -1.0, dtype=np.float32)
        low[: self.cfg.n_rays * self.cfg.n_stack] = 0.0
        self.observation_space = gym.spaces.Box(low, np.ones(obs_dim, dtype=np.float32), dtype=np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

        self._scans: deque[np.ndarray] = deque(maxlen=(self.cfg.n_stack - 1) * self.cfg.stack_gap + 1)
        # per-episode obstacle settings when training on a stage mix
        from navrl.config import MIX_KEYS, STAGES, EnvConfig as _E
        self._mix = [{k: STAGES[st].get(k, getattr(_E(), k)) for k in MIX_KEYS} for st in self.cfg.mix]
        self._field = None
        self._renderer = None
        self.goal = np.zeros(2)

    # helpers
    def _goal_polar(self) -> tuple[float, float]:
        x, y, yaw = self.robot.pose(self.data)
        dx, dy = self.goal[0] - x, self.goal[1] - y
        return float(np.hypot(dx, dy)), wrap_angle(np.arctan2(dy, dx) - yaw)

    def _observe(self) -> np.ndarray:
        dist, bearing = self._goal_polar()
        v, w = self.robot.measured_velocity(self.data)
        p = self.robot.p
        hist = list(self._scans)[::-1]                                   # newest first
        scans = np.concatenate(hist[:: self.cfg.stack_gap][: self.cfg.n_stack]) / self.cfg.lidar_range
        tail = np.array([min(dist / self.arena_diag, 1.0), np.sin(bearing), np.cos(bearing),
                         np.clip(v / p.v_max, -1, 1), np.clip(w / p.w_max, -1, 1),
                         *self._prev_action])
        return np.concatenate([scans, tail]).astype(np.float32)

    def _collided(self) -> bool:
        n = self.data.ncon
        if n == 0:
            return False
        g1 = self.data.contact.geom1[:n]
        g2 = self.data.contact.geom2[:n]
        return bool(np.any((g1 == self.chassis_geom) | (g2 == self.chassis_geom)))

    def _sample_start_goal(self) -> tuple[np.ndarray, np.ndarray]:
        h = self.cfg.arena_half - 0.4
        lo, hi = self.cfg.start_goal_dist
        while True:
            s = self.np_random.uniform(-h, h, 2)
            g = self.np_random.uniform(-h, h, 2)
            if lo <= np.linalg.norm(g - s) <= hi:
                return s, g

    # gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        cfg = self.cfg
        start, self.goal = self._sample_start_goal()
        yaw = self.np_random.uniform(-np.pi, np.pi)
        self.robot.set_pose(self.data, start[0], start[1], yaw)
        o = self._mix[int(self.np_random.integers(len(self._mix)))] if self._mix else \
            {k: getattr(cfg, k) for k in ("n_static", "n_dynamic", "dyn_speed", "p_crossing", "n_walls", "dyn_wander")}
        n_s = int(self.np_random.integers(o["n_static"][0], o["n_static"][1] + 1))
        n_d = int(self.np_random.integers(o["n_dynamic"][0], o["n_dynamic"][1] + 1))
        n_w = int(self.np_random.integers(o["n_walls"][0], o["n_walls"][1] + 1))
        self.obstacles.reset(self.np_random, start, self.goal, n_s, n_d, o["dyn_speed"], o["p_crossing"],
                             n_walls=n_w, wander=o["dyn_wander"])
        self.obstacles.write(self.data)
        self.data.mocap_pos[self.goal_mocap] = (self.goal[0], self.goal[1], 0.005)
        mujoco.mj_forward(self.model, self.data)

        self._steps = 0
        self._prev_action = np.zeros(2)
        scan = self.lidar.scan(self.data, yaw)
        self._scans.clear()
        for _ in range(self._scans.maxlen):
            self._scans.append(scan)
        self._last_scan = scan
        self._field = self.obstacles.geodesic_field(self.goal) if cfg.geodesic_progress else None
        self._prev_dist = self._potential()
        self.start = start.copy()
        info = {"start": start.copy(), "goal": self.goal.copy(), "start_dist": self._prev_dist}
        return self._observe(), info

    def set_goal(self, x: float, y: float) -> np.ndarray:
        """Move the goal mid-episode (manual goal tool). Clamped inside the arena; returns new obs."""
        h = self.cfg.arena_half - 0.35
        self.goal = np.clip(np.array([x, y], dtype=float), -h, h)
        self.data.mocap_pos[self.goal_mocap] = (self.goal[0], self.goal[1], 0.005)
        self._field = self.obstacles.geodesic_field(self.goal) if self.cfg.geodesic_progress else None
        self._prev_dist = self._potential()
        return self._observe()

    def _potential(self) -> float:
        """Distance-to-goal used by the progress reward: Euclidean, or geodesic (bilinear lookup)."""
        if self._field is None:
            return self._goal_polar()[0]
        x, y, _ = self.robot.pose(self.data)
        F, res, h = self._field, 0.05, self.cfg.arena_half
        n = F.shape[0]
        u = np.clip((x + h) / res - 0.5, 0, n - 1.001)
        v = np.clip((y + h) / res - 0.5, 0, n - 1.001)
        i, j = int(u), int(v)
        fu, fv = u - i, v - j
        return float((1 - fu) * (1 - fv) * F[i, j] + fu * (1 - fv) * F[i + 1, j]
                     + (1 - fu) * fv * F[i, j + 1] + fu * fv * F[i + 1, j + 1])

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        v, w = denormalize_action(a, self.robot.p)
        self.robot.set_command(v, w, self.cfg.ctrl_dt)

        collided = False
        for _ in range(self.n_sub):
            self.obstacles.step(self.cfg.physics_dt, self.np_random)
            self.obstacles.write_dynamic(self.data)
            self.robot.apply_command(self.data)
            mujoco.mj_step(self.model, self.data)
            collided = collided or self._collided()
        self._steps += 1

        x, y, yaw = self.robot.pose(self.data)
        scan = self.lidar.scan(self.data, yaw)
        self._scans.append(scan)
        self._last_scan = scan
        dist, _ = self._goal_polar()
        pot = self._potential()
        clearance = float(scan.min()) - self.robot.p.body_radius

        rc = self.cfg.reward
        success = (not collided) and dist < self.cfg.goal_radius
        terms = {
            "progress": rc.progress * (self._prev_dist - pot),
            "goal": rc.goal if success else 0.0,
            "collision": -rc.collision if collided else 0.0,
            "time": -rc.time,
            "smooth": -rc.smooth * float(np.sum((a - self._prev_action) ** 2)),
            "proximity": -rc.proximity * max(0.0, 1.0 - clearance / rc.d_safe),
        }
        reward = float(sum(terms.values()))

        sim_error = bool(np.isnan(self.data.qpos).any())
        terminated = collided or success or sim_error
        truncated = (not terminated) and self._steps >= self.cfg.max_steps
        outcome = ("sim_error" if sim_error else "collision" if collided else "success" if success
                   else "timeout" if truncated else "running")

        self._prev_dist = pot
        self._prev_action = a
        info = {"reward_terms": terms, "outcome": outcome, "is_success": success,
                "clearance": clearance, "pos": (x, y), "yaw": yaw, "goal_dist": dist}
        return self._observe(), reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, 480, 480)
        self._renderer.update_scene(self.data, camera="top")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
