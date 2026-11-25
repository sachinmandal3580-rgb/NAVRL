"""Drive the robot yourself in the live MuJoCo viewer (keyboard teleop).

Usage:
  python scripts/teleop.py                       # stage s3_dynamic (static + moving obstacles)
  python scripts/teleop.py --stage s2_static
  python scripts/teleop.py --stage s1_empty --seed 5

Keys (click the viewer window first so it has keyboard focus):
  UP / DOWN      increase / decrease forward speed v   (steps of 0.1 m/s, range -0.2 .. 0.5)
  LEFT / RIGHT   turn left / right: change yaw rate w  (steps of 0.3 rad/s, range -1.5 .. 1.5)
  SPACE          stop (v = 0, w = 0)
  ENTER          new episode (next seed)
  R              toggle LiDAR rays

Driving goes through exactly the same env.step() as the trained policy: same acceleration
limits, wheel kinematics, collision/goal/timeout detection and reward terms. The terminal
shows a live status line (command, measured speed, goal distance, clearance, reward) and the
episode outcome. After a collision / success / timeout the next episode starts automatically.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from navrl.config import STAGES, make_config  # noqa: E402
from navrl.env import NavRLEnv  # noqa: E402
from navrl.viz import add_overlays  # noqa: E402

# GLFW key codes
KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_SPACE, KEY_ENTER, KEY_KP_ENTER = 32, 257, 335
KEY_R = (82, 114)

V_STEP, W_STEP = 0.1, 0.3


def to_action(v: float, w: float, p) -> np.ndarray:
    """Inverse of robot.denormalize_action: (v, w) in physical units -> action in [-1, 1]^2."""
    a_v = v / p.v_max if v >= 0 else v / (-p.v_min)
    return np.array([np.clip(a_v, -1, 1), np.clip(w / p.w_max, -1, 1)], dtype=np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=list(STAGES), default="s3_dynamic")
    ap.add_argument("--seed", type=int, default=300_000)
    ap.add_argument("--max_steps", type=int, default=600, help="episode limit (60 s) for manual driving")
    args = ap.parse_args()

    import mujoco.viewer

    env = NavRLEnv(make_config(args.stage, max_steps=args.max_steps))
    p = env.robot.p
    cmd = {"v": 0.0, "w": 0.0, "reset": False, "rays": True}

    def on_key(key: int) -> None:
        if key == KEY_UP:
            cmd["v"] = min(cmd["v"] + V_STEP, p.v_max)
        elif key == KEY_DOWN:
            cmd["v"] = max(cmd["v"] - V_STEP, p.v_min)
        elif key == KEY_LEFT:
            cmd["w"] = min(cmd["w"] + W_STEP, p.w_max)
        elif key == KEY_RIGHT:
            cmd["w"] = max(cmd["w"] - W_STEP, -p.w_max)
        elif key == KEY_SPACE:
            cmd["v"] = cmd["w"] = 0.0
        elif key in (KEY_ENTER, KEY_KP_ENTER):
            cmd["reset"] = True
        elif key in KEY_R:
            cmd["rays"] = not cmd["rays"]
        cmd["v"], cmd["w"] = round(cmd["v"], 2), round(cmd["w"], 2)

    print(__doc__.split("Keys")[1].split("Driving")[0].strip("\n :"))
    seed = args.seed
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = (0, 0, 0); viewer.cam.distance = 9.0
        viewer.cam.elevation = -70; viewer.cam.azimuth = 90
        while viewer.is_running():
            obs, info = env.reset(seed=seed)
            trail = [info["start"].copy()]
            cmd.update(v=0.0, w=0.0, reset=False)
            ep_ret, outcome = 0.0, "running"
            print(f"\n--- episode seed {seed}: goal {info['goal'].round(2)}, distance {info['start_dist']:.2f} m ---")
            while viewer.is_running() and not cmd["reset"]:
                t0 = time.time()
                obs, r, term, trunc, inf = env.step(to_action(cmd["v"], cmd["w"], p))
                ep_ret += r
                trail.append(np.array(inf["pos"]))
                v_meas, w_meas = env.robot.measured_velocity(env.data)
                print(f"\rcmd v={cmd['v']:+.2f} w={cmd['w']:+.2f} | meas v={v_meas:+.2f} w={w_meas:+.2f} | "
                      f"goal {inf['goal_dist']:4.2f} m | clearance {max(inf['clearance'], 0):4.2f} m | "
                      f"return {ep_ret:+6.2f}   ", end="", flush=True)
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    add_overlays(viewer.user_scn, env, trail[-400:], cmd["rays"])
                viewer.sync()
                if term or trunc:
                    outcome = inf["outcome"]
                    print(f"\n{outcome.upper()} after {len(trail) - 1} steps, return {ep_ret:+.2f}")
                    end = time.time()
                    while viewer.is_running() and time.time() - end < 1.5:   # hold the final frame
                        viewer.sync(); time.sleep(0.05)
                    break
                time.sleep(max(0.0, env.cfg.ctrl_dt - (time.time() - t0)))
            seed += 1
    # MuJoCo's passive viewer can segfault during interpreter teardown on Linux; exit directly.
    print()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
