"""Watch a policy drive in the MuJoCo viewer (real time), or record an MP4.

Examples:
  python scripts/demo.py --model logs/navrl/curriculum/s3/best_model.pt --stage s3_dynamic
  python scripts/demo.py --controller gotogoal --stage s2_static        # obstacle-blind baseline
  python scripts/demo.py --model logs/navrl/curriculum/s3/best_model.pt --stage s3_dynamic --record demo.mp4

Viewer keys: SPACE pause/resume, N next episode, R toggle LiDAR rays.
Overlays: LiDAR rays (green = far, red = close), blue path trail, small green goal marker,
grey static obstacles, orange moving obstacles.
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
from navrl.controllers import GoToGoal  # noqa: E402
from navrl.env import NavRLEnv  # noqa: E402
from navrl.viz import add_overlays  # noqa: E402


def load_policy(args):
    if args.controller == "gotogoal":
        return GoToGoal().act, {"n_stack": 1}
    from navrl.checkpoint import load_policy as load_pt, n_stack_of, obs_kwargs_of
    policy, ckpt = load_pt(args.model)
    print(f"loaded {args.model}: task={ckpt.get('task')} iteration={ckpt.get('iteration')} "
          f"n_stack={n_stack_of(ckpt)}")
    return (lambda o, e: policy.predict(o, deterministic=True)[0]), obs_kwargs_of(ckpt)


def run_viewer(env, policy, args):
    import mujoco.viewer
    state = {"paused": False, "next": False, "rays": True}

    def on_key(key):
        if key == 32: state["paused"] = not state["paused"]          # space
        elif key in (78, 110): state["next"] = True                    # N
        elif key in (82, 114): state["rays"] = not state["rays"]       # R

    seed = args.seed
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = (0, 0, 0); viewer.cam.distance = 9.0
        viewer.cam.elevation = -70; viewer.cam.azimuth = 90
        for ep in range(args.episodes):
            obs, info = env.reset(seed=seed + ep)
            trail = [info["start"].copy()]
            state["next"] = False
            outcome = "running"
            while viewer.is_running() and not state["next"]:
                t0 = time.time()
                if not state["paused"] and outcome == "running":
                    obs, r, term, trunc, inf = env.step(policy(obs, env))
                    trail.append(np.array(inf["pos"]))
                    if term or trunc:
                        outcome = inf["outcome"]
                        print(f"episode {ep} (seed {seed + ep}): {outcome} after {len(trail) - 1} steps")
                        end_time = time.time()
                with viewer.lock():
                    viewer.user_scn.ngeom = 0
                    add_overlays(viewer.user_scn, env, trail[-400:], state["rays"])
                viewer.sync()
                if outcome != "running" and time.time() - end_time > 1.5:
                    break
                time.sleep(max(0.0, env.cfg.ctrl_dt / args.speed - (time.time() - t0)))
            if not viewer.is_running():
                break
        # keep the last frame up until the user closes the window
        if viewer.is_running():
            print("all episodes done - close the viewer window to exit")
        while viewer.is_running():
            time.sleep(0.1)


def record(env, policy, args):
    import imageio.v2 as imageio
    import mujoco
    renderer = mujoco.Renderer(env.model, 720, 720)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = (0, 0, 0); cam.distance = 8.5; cam.elevation = -70; cam.azimuth = 90
    frames = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        trail = [info["start"].copy()]
        while True:
            obs, r, term, trunc, inf = env.step(policy(obs, env))
            trail.append(np.array(inf["pos"]))
            renderer.update_scene(env.data, cam)
            add_overlays(renderer.scene, env, trail, True)
            frames.append(renderer.render().copy())
            if term or trunc:
                print(f"episode {ep}: {inf['outcome']}")
                frames += [frames[-1]] * 10
                break
    imageio.mimsave(args.record, frames, fps=int(10 * args.speed))
    print(f"wrote {args.record} ({len(frames)} frames)")


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=str)
    g.add_argument("--controller", choices=["gotogoal"])
    ap.add_argument("--stage", choices=list(STAGES), default="s3_dynamic")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--seed", type=int, default=200_000)
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    ap.add_argument("--record", type=str, default=None, help="write MP4 instead of opening viewer")
    args = ap.parse_args()

    policy, n_stack = load_policy(args)
    env = NavRLEnv(make_config(args.stage, **n_stack))
    if args.record:
        record(env, policy, args)
    else:
        run_viewer(env, policy, args)
        # MuJoCo's passive viewer can segfault during normal interpreter teardown on Linux
        # (GLFW/OpenGL cleanup racing the render thread). Everything is finished here,
        # so exit immediately and skip that teardown.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
