"""Non-learned controllers sharing the policy's interface: act(obs, env) -> action in [-1, 1]^2.

GoToGoal: proportional heading controller that IGNORES obstacles. Used to
  (1) validate the env (should reach ~100% of goals in an empty arena), and
  (2) give a lower bound for collision avoidance.
"""
from __future__ import annotations

import numpy as np


class GoToGoal:
    def __init__(self, k_w: float = 2.0, v_frac: float = 0.8):
        self.k_w = k_w
        self.v_frac = v_frac

    def act(self, obs: np.ndarray, env) -> np.ndarray:
        n = env.cfg.n_rays * env.cfg.n_stack
        sin_b, cos_b = obs[n + 1], obs[n + 2]
        bearing = np.arctan2(sin_b, cos_b)
        a_w = np.clip(self.k_w * bearing / env.robot.p.w_max, -1, 1)
        # slow down while facing away from the goal; turn in place if behind
        a_v = self.v_frac * max(0.0, np.cos(bearing))
        return np.array([a_v, a_w], dtype=np.float32)
