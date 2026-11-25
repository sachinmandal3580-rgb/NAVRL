"""Simulated 2D LiDAR using MuJoCo ray casting.

All rays are cast in one C call (mujoco.mj_multiRay) from the 'lidar' site on the
robot, horizontally, evenly spaced over 360 degrees in the robot frame
(ray 0 points straight ahead, angles increase counter-clockwise).

  * geomgroup mask: only group 1 (walls + obstacles) is visible, so the floor and
    the goal marker are never hit.
  * bodyexclude: the robot body, so the robot never sees itself.
  * rays with no hit within `max_range` return `max_range`.
"""
from __future__ import annotations

import mujoco
import numpy as np


class Lidar:
    def __init__(self, model: mujoco.MjModel, n_rays: int = 60, max_range: float = 3.0,
                 site: str = "lidar", exclude_body: str = "base"):
        self.model = model
        self.n_rays = n_rays
        self.max_range = max_range
        self.site_id = model.site(site).id
        self.exclude_body = model.body(exclude_body).id
        self.angles = np.linspace(0.0, 2 * np.pi, n_rays, endpoint=False)
        self.geomgroup = np.array([0, 1, 0, 0, 0, 0], dtype=np.uint8)
        self._vec = np.zeros(3 * n_rays)
        self._geomid = np.zeros(n_rays, dtype=np.int32)
        self._dist = np.zeros(n_rays)

    def scan(self, data: mujoco.MjData, yaw: float) -> np.ndarray:
        """Return ranges [m], shape (n_rays,), clipped to [0, max_range]."""
        a = self.angles + yaw
        self._vec[0::3] = np.cos(a)
        self._vec[1::3] = np.sin(a)
        # z component stays 0: horizontal rays
        pnt = data.site_xpos[self.site_id].copy()
        mujoco.mj_multiRay(self.model, data, pnt, self._vec, self.geomgroup, 1,
                           self.exclude_body, self._geomid, self._dist, None,
                           self.n_rays, self.max_range)
        d = self._dist.copy()
        d[(self._geomid < 0) | (d < 0)] = self.max_range
        return np.minimum(d, self.max_range)

    def endpoints(self, data: mujoco.MjData, yaw: float, ranges: np.ndarray) -> np.ndarray:
        """World-frame hit points (n_rays, 3), for visualisation."""
        a = self.angles + yaw
        p = data.site_xpos[self.site_id]
        return np.stack([p[0] + ranges * np.cos(a), p[1] + ranges * np.sin(a),
                         np.full_like(ranges, p[2])], axis=1)
