"""Differential-drive robot: action mapping, acceleration limits and kinematics.

Pipeline (see docs/DESIGN.md section 8):
    policy action a in [-1, 1]^2
      -> (v, w) via piecewise-linear denormalisation (a = 0 means "stop")
      -> acceleration limit (dynamic-feasibility filter, DWA-style)
      -> wheel angular speeds via inverse kinematics
      -> MuJoCo velocity servos on the wheel hinges
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class RobotParams:
    wheel_radius: float = 0.04       # r [m]
    track_width: float = 0.28        # L [m], distance between wheel contact points
    body_radius: float = 0.15        # footprint radius [m]
    v_max: float = 0.5               # forward speed limit [m/s]
    v_min: float = -0.2              # reverse speed limit [m/s]
    w_max: float = 1.5               # yaw-rate limit [rad/s]
    acc_v: float = 1.0               # linear acceleration limit [m/s^2]
    acc_w: float = 3.0               # angular acceleration limit [rad/s^2]
    wheel_speed_max: float = 20.0    # actuator ctrlrange [rad/s]


def denormalize_action(a: np.ndarray, p: RobotParams) -> tuple[float, float]:
    """Map a in [-1,1]^2 to (v, w). Piecewise-linear in v so that a_v = 0 -> v = 0."""
    av, aw = float(np.clip(a[0], -1, 1)), float(np.clip(a[1], -1, 1))
    v = av * p.v_max if av >= 0 else av * (-p.v_min)
    return v, aw * p.w_max


def accel_limit(v: float, w: float, v_prev: float, w_prev: float, dt: float,
                p: RobotParams) -> tuple[float, float]:
    """Clip the command change to what the base can physically follow in one control step."""
    v = float(np.clip(v, v_prev - p.acc_v * dt, v_prev + p.acc_v * dt))
    w = float(np.clip(w, w_prev - p.acc_w * dt, w_prev + p.acc_w * dt))
    return v, w


def inverse_kinematics(v: float, w: float, p: RobotParams) -> tuple[float, float]:
    """(v, w) -> (omega_left, omega_right) wheel speeds [rad/s].

    If a wheel would exceed its speed limit, both are scaled by the same factor, which
    preserves the path curvature w/v (the robot slows down instead of changing course).
    """
    wl = (v - w * p.track_width / 2) / p.wheel_radius
    wr = (v + w * p.track_width / 2) / p.wheel_radius
    peak = max(abs(wl), abs(wr))
    if peak > p.wheel_speed_max:
        s = p.wheel_speed_max / peak
        wl, wr = wl * s, wr * s
    return wl, wr


def forward_kinematics(wl: float, wr: float, p: RobotParams) -> tuple[float, float]:
    """(omega_left, omega_right) -> (v, w)."""
    v = p.wheel_radius * (wr + wl) / 2
    w = p.wheel_radius * (wr - wl) / p.track_width
    return v, w


class RobotHandle:
    """Accessor for the planar differential-drive base inside an MjModel/MjData pair.

    The base has planar joints (x, y, yaw) driven by velocity servos. `apply_command`
    must be called every physics substep: it converts the body-frame command (v, w)
    into world-frame servo targets using the *current* yaw, which enforces the
    non-holonomic constraint (zero commanded sideways velocity).
    """

    def __init__(self, model: mujoco.MjModel, params: RobotParams | None = None):
        self.p = params or RobotParams()
        self.body_id = model.body("base").id
        self.jx = model.joint("base_x").qposadr[0]
        self.jy = model.joint("base_y").qposadr[0]
        self.jyaw = model.joint("base_yaw").qposadr[0]
        self.dx = model.joint("base_x").dofadr[0]
        self.dy = model.joint("base_y").dofadr[0]
        self.dyaw = model.joint("base_yaw").dofadr[0]
        self.act_x = model.actuator("drive_x").id
        self.act_y = model.actuator("drive_y").id
        self.act_yaw = model.actuator("drive_yaw").id
        self.vel_adr = model.sensor("vel").adr[0]
        self.gyro_adr = model.sensor("gyro").adr[0]
        # collidable geoms of the robot (chassis only; wheels/casters are visual)
        self.geom_ids = np.array([g for g in range(model.ngeom)
                                  if model.geom_bodyid[g] == self.body_id
                                  and (model.geom_contype[g] or model.geom_conaffinity[g])])
        # last command actually sent (after accel limit + wheel saturation)
        self.v_cmd = 0.0
        self.w_cmd = 0.0

    # state
    def pose(self, data: mujoco.MjData) -> tuple[float, float, float]:
        """(x, y, yaw) of the chassis in the world frame."""
        return float(data.qpos[self.jx]), float(data.qpos[self.jy]), float(data.qpos[self.jyaw])

    def measured_velocity(self, data: mujoco.MjData) -> tuple[float, float]:
        """Body-frame forward speed (velocimeter x) and yaw rate (gyro z)."""
        return float(data.sensordata[self.vel_adr]), float(data.sensordata[self.gyro_adr + 2])

    def wheel_speeds(self) -> tuple[float, float]:
        """Wheel angular speeds implied by the current command (for logging/visuals)."""
        return inverse_kinematics(self.v_cmd, self.w_cmd, self.p)

    # commands
    def set_command(self, v: float, w: float, dt: float) -> None:
        """Accel-limit (v, w), saturate through the wheels, and store as the active command."""
        v, w = accel_limit(v, w, self.v_cmd, self.w_cmd, dt, self.p)
        wl, wr = inverse_kinematics(v, w, self.p)        # wheel-speed limits
        self.v_cmd, self.w_cmd = forward_kinematics(wl, wr, self.p)

    def apply_command(self, data: mujoco.MjData) -> None:
        """Write servo targets for the active command; call every physics substep."""
        yaw = data.qpos[self.jyaw]
        data.ctrl[self.act_x] = self.v_cmd * np.cos(yaw)
        data.ctrl[self.act_y] = self.v_cmd * np.sin(yaw)
        data.ctrl[self.act_yaw] = self.w_cmd

    def set_pose(self, data: mujoco.MjData, x: float, y: float, yaw: float) -> None:
        data.qpos[self.jx], data.qpos[self.jy], data.qpos[self.jyaw] = x, y, yaw
        data.qvel[[self.dx, self.dy, self.dyaw]] = 0
        self.v_cmd = self.w_cmd = 0.0
