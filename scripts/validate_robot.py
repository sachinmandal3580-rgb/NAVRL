"""Physics validation battery for the differential-drive robot (run before any RL).

Usage:  python scripts/validate_robot.py
Prints a PASS/FAIL table. Each test drives the robot open-loop in an empty arena
through the same pipeline the RL env uses (accel limit -> wheel IK/saturation ->
non-holonomic velocity servos, re-applied every physics substep).
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from navrl import ARENA_XML  # noqa: E402
from navrl.robot import RobotHandle  # noqa: E402

CTRL_DT = 0.1


def make(timestep: float | None = None):
    m = mujoco.MjModel.from_xml_path(str(ARENA_XML))
    if timestep:
        m.opt.timestep = timestep
    d = mujoco.MjData(m)
    r = RobotHandle(m)
    r.set_pose(d, -2.0, 0.0, 0.0)
    mujoco.mj_forward(m, d)
    return m, d, r


def drive(m, d, r, profile):
    """profile: list of (duration_s, v, w). Returns a per-control-step log."""
    n_sub = int(round(CTRL_DT / m.opt.timestep))
    log = []
    for dur, v_t, w_t in profile:
        for _ in range(int(round(dur / CTRL_DT))):
            r.set_command(v_t, w_t, CTRL_DT)
            for _ in range(n_sub):
                r.apply_command(d)
                mujoco.mj_step(m, d)
            x, y, yaw = r.pose(d)
            v, w = r.measured_velocity(d)
            lat = float(d.sensordata[r.vel_adr + 1])      # body-frame sideways speed
            log.append(dict(x=x, y=y, yaw=yaw, v=v, w=w, lat=lat,
                            v_cmd=r.v_cmd, w_cmd=r.w_cmd,
                            nan=bool(np.isnan(d.qpos).any())))
    return log


def report(name, ok, detail):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<26} {detail}")
    return ok


def main() -> int:
    results = []
    print("NavRL robot validation (planar diff-drive model)")

    m, d, r = make()
    log = drive(m, d, r, [(5.0, 0, 0)])
    drift = np.hypot(log[-1]["x"] + 2.0, log[-1]["y"])
    results.append(report("rest stability", drift < 0.01, f"drift={drift*100:.3f} cm"))

    m, d, r = make()
    log = drive(m, d, r, [(5.0, 0.3, 0)])
    vs = np.array([s["v"] for s in log])
    ss = vs[-10:].mean(); lat = abs(log[-1]["y"])
    rise = (np.argmax(vs > 0.27) - np.argmax(vs > 0.03)) * CTRL_DT
    results.append(report("forward v=0.3", abs(ss - 0.3) / 0.3 < 0.05 and lat < 0.02,
                          f"v_ss={ss:.3f} rise={rise:.1f}s lateral={lat*100:.2f} cm"))

    m, d, r = make()
    log = drive(m, d, r, [(4.0, -0.2, 0)])
    ss = np.mean([s["v"] for s in log[-10:]])
    results.append(report("reverse v=-0.2", abs(ss + 0.2) / 0.2 < 0.05, f"v_ss={ss:.3f}"))

    m, d, r = make()
    log = drive(m, d, r, [(2 * np.pi + 0.5, 0, 1.0)])
    ws = np.mean([s["w"] for s in log[10:]])
    centre = np.hypot(log[-1]["x"] + 2.0, log[-1]["y"])
    results.append(report("rotate in place w=1.0", abs(ws - 1.0) < 0.05 and centre < 0.03,
                          f"w_ss={ws:.3f} centre_drift={centre*100:.2f} cm"))

    m, d, r = make()
    log = drive(m, d, r, [(8.0, 0.3, 0.5)])
    R = np.mean([s["v"] for s in log[20:]]) / np.mean([s["w"] for s in log[20:]])
    results.append(report("curve R=0.6 m", abs(R - 0.6) / 0.6 < 0.05, f"R={R:.3f} m"))

    m, d, r = make()
    log = drive(m, d, r, [(2.0, 0.5, 0), (1.5, 0, 0)])
    stop = log[-1]["x"] - log[19]["x"]; v_top = log[19]["v"]
    expected = v_top ** 2 / (2 * 1.0)
    results.append(report("accel + brake", abs(stop - expected) / expected < 0.2,
                          f"v_top={v_top:.3f} stop_dist={stop:.3f} m (ideal {expected:.3f})"))

    m, d, r = make()
    log = drive(m, d, r, [(3.0, 0.5, 0.0), (3.0, 0.4, 1.0), (2.0, 0.5, -1.5)])
    lat = max(abs(s["lat"]) for s in log)
    results.append(report("no sideways slip", lat < 0.02, f"max lateral speed={lat*100:.2f} cm/s"))

    m, d, r = make()
    log = drive(m, d, r, [(1.0, 0.5, 1.5), (1.0, 0.5, -1.5)] * 3)
    nan = any(s["nan"] for s in log); n_warn = sum(int(w.number) for w in d.warning)
    results.append(report("aggressive manoeuvre", not nan and n_warn == 0, f"nan={nan} warnings={n_warn}"))

    # Wall impact: drive into the east wall at full speed; must stop, not tunnel or explode.
    m, d, r = make()
    r.set_pose(d, 2.0, 0.0, 0.0)
    log = drive(m, d, r, [(4.0, 0.5, 0)])
    x_max = max(s["x"] for s in log)
    wall_ok = x_max < 3.0 - 0.15 + 0.02 and not any(s["nan"] for s in log)
    results.append(report("wall impact", wall_ok, f"max x={x_max:.3f} (wall face at 2.85 for robot centre)"))

    finals = {}
    for ts in (0.005, 0.0025, 0.01):
        m, d, r = make(ts)
        log = drive(m, d, r, [(3.0, 0.3, 0), (3.0, 0.3, 0.5)])
        finals[ts] = np.array([log[-1]["x"], log[-1]["y"]])
    dev = np.linalg.norm(finals[0.005] - finals[0.0025])
    dev01 = np.linalg.norm(finals[0.01] - finals[0.0025])
    results.append(report("timestep sensitivity", dev < 0.03, f"dt 0.005 vs 0.0025: {dev*100:.2f} cm"))
    report("(info) dt=0.01 viable", dev01 < 0.03, f"dt 0.01 vs 0.0025: {dev01*100:.2f} cm")

    print(f"\n{sum(results)}/{len(results)} tests passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
