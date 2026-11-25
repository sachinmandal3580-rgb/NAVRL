"""Give the trained policy goals by hand and watch it drive there (live MuJoCo viewer).

Usage:
  python scripts/goal_nav.py --model logs/navrl/curriculum/s3/best_model.pt
  python scripts/goal_nav.py --model logs/navrl/curriculum/s4/best_model.pt --stage s4_complex
  python scripts/goal_nav.py --controller gotogoal --stage s2_static     # obstacle-blind baseline

Three ways to set the goal (green disc):
  1. TERMINAL  type  "x y"  and press Enter, e.g.  1.5 -2   (metres, arena is -3..3)
               "r" = random goal, "q" = quit
  2. KEYS      (click the viewer first) arrow keys move the goal 0.25 m
               SPACE pause/resume, ENTER new obstacle layout (same goal), R toggle LiDAR rays
  3. MOUSE     double-click the small green goal marker to select it, then Ctrl + right-drag to move it

The robot drives to the goal and waits there. Move the goal again and it drives on.
Goal reached = status line says REACHED; on a collision the arena is re-generated
(new obstacle layout) and the robot restarts, keeping your goal. No time limit.
"""
from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from navrl.config import STAGES, make_config  # noqa: E402
from navrl.controllers import GoToGoal  # noqa: E402
from navrl.env import NavRLEnv  # noqa: E402
from navrl.viz import add_overlays  # noqa: E402

KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT = 265, 264, 263, 262
KEY_SPACE, KEY_ENTER, KEY_KP_ENTER = 32, 257, 335
KEY_R = (82, 114)
NUDGE = 0.25


def load_policy(args):
    if args.controller == "gotogoal":
        return GoToGoal().act, {"n_stack": 1}
    from navrl.checkpoint import load_policy as load_pt, n_stack_of, obs_kwargs_of
    policy, ckpt = load_pt(args.model)
    print(f"loaded {args.model}: task={ckpt.get('task')} iteration={ckpt.get('iteration')} "
          f"n_stack={n_stack_of(ckpt)}")
    return (lambda o, e: policy.predict(o, deterministic=True)[0]), obs_kwargs_of(ckpt)


def goal_blocked(env, g) -> str | None:
    """Warn if the goal centre lies inside/too close to an obstacle or wall."""
    r = env.robot.p.body_radius
    for pos, rad, dyn in env.obstacles.active_circles():
        if not dyn and np.linalg.norm(g - pos) < rad + r:
            return "static obstacle"
    walls = [w for w in env.obstacles.walls if w.active]
    if walls and env.obstacles.wall_dist(g, walls) < r:
        return "wall"
    return None


def stdin_reader(q: queue.Queue) -> None:
    """Background thread: parse terminal lines into goal commands."""
    for line in sys.stdin:
        t = line.strip().lower().replace(",", " ").split()
        if not t:
            continue
        if t[0] in ("q", "quit", "exit"):
            q.put(("quit",))
        elif t[0] in ("r", "random"):
            q.put(("random",))
        elif len(t) == 2:
            try:
                q.put(("goal", float(t[0]), float(t[1])))
            except ValueError:
                print(f"\n  ? could not parse {line.strip()!r}; type e.g.  1.5 -2", flush=True)
        else:
            print(f"\n  ? type 'x y' (e.g. 1.5 -2), 'r' or 'q'", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=str)
    g.add_argument("--controller", choices=["gotogoal"])
    ap.add_argument("--stage", choices=list(STAGES), default="s3_dynamic")
    ap.add_argument("--seed", type=int, default=400_000)
    args = ap.parse_args()

    import mujoco.viewer

    policy, n_stack = load_policy(args)
    env = NavRLEnv(make_config(args.stage, **n_stack, max_steps=10**9))
    rng = np.random.default_rng(args.seed)
    cmds: queue.Queue = queue.Queue()
    state = {"paused": False, "rays": True}

    def on_key(key: int) -> None:          # viewer thread: only enqueue, main loop applies
        moves = {KEY_UP: (0, NUDGE), KEY_DOWN: (0, -NUDGE), KEY_LEFT: (-NUDGE, 0), KEY_RIGHT: (NUDGE, 0)}
        if key in moves:
            cmds.put(("nudge", *moves[key]))
        elif key == KEY_SPACE:
            state["paused"] = not state["paused"]
        elif key in (KEY_ENTER, KEY_KP_ENTER):
            cmds.put(("layout",))
        elif key in KEY_R:
            state["rays"] = not state["rays"]

    threading.Thread(target=stdin_reader, args=(cmds,), daemon=True).start()
    print(__doc__.split("Three ways")[1].split("The robot")[0].join(["Three ways", ""]).rstrip())

    seed = args.seed
    obs, info = env.reset(seed=seed)
    goal = info["goal"].copy()
    trail = [info["start"].copy()]
    reached, last_goal_written = False, goal.copy()

    def apply_goal(x, y, why):
        nonlocal obs, goal, reached, last_goal_written
        obs = env.set_goal(x, y)
        goal, last_goal_written, reached = env.goal.copy(), env.goal.copy(), False
        warn = goal_blocked(env, goal)
        print(f"\nnew goal ({goal[0]:+.2f}, {goal[1]:+.2f}) [{why}]"
              + (f"  WARNING: inside/too close to a {warn}, robot cannot reach it" if warn else ""),
              flush=True)

    print(f"\nstart goal ({goal[0]:+.2f}, {goal[1]:+.2f}); type a new one any time")
    with mujoco.viewer.launch_passive(env.model, env.data, key_callback=on_key) as viewer:
        viewer.cam.lookat[:] = (0, 0, 0); viewer.cam.distance = 9.0
        viewer.cam.elevation = -70; viewer.cam.azimuth = 90
        while viewer.is_running():
            t0 = time.time()
            # goal moved by mouse drag in the viewer (mocap perturbation)?
            with viewer.lock():
                g_view = np.array(env.data.mocap_pos[env.goal_mocap][:2])
            if np.linalg.norm(g_view - last_goal_written) > 1e-4:
                apply_goal(*g_view, "mouse")
            # terminal / key commands
            while not cmds.empty():
                c = cmds.get()
                if c[0] == "quit":
                    viewer.close()
                    break
                if c[0] == "goal":
                    apply_goal(c[1], c[2], "typed")
                elif c[0] == "nudge":
                    apply_goal(goal[0] + c[1], goal[1] + c[2], "keys")
                elif c[0] == "random":
                    h = env.cfg.arena_half - 0.4
                    apply_goal(*rng.uniform(-h, h, 2), "random")
                elif c[0] == "layout":
                    seed += 1
                    with viewer.lock():
                        obs, info = env.reset(seed=seed)
                        apply_goal(*goal, "new layout")
                    trail = [info["start"].copy()]
            if not viewer.is_running():
                break

            if not state["paused"]:
                action = np.zeros(2) if reached else policy(obs, env)
                with viewer.lock():
                    obs, r, term, trunc, inf = env.step(action)
                trail.append(np.array(inf["pos"]))
                if inf["outcome"] == "success" and not reached:
                    reached = True
                    print(f"\nREACHED ({goal[0]:+.2f}, {goal[1]:+.2f}) after {len(trail) - 1} steps. "
                          f"Give a new goal.", flush=True)
                elif inf["outcome"] == "collision":
                    print("\nCOLLISION - new obstacle layout, robot restarts, goal kept", flush=True)
                    seed += 1
                    with viewer.lock():
                        obs, info = env.reset(seed=seed)
                        apply_goal(*goal, "after collision")
                    trail = [info["start"].copy()]
                elif reached and inf["goal_dist"] > env.cfg.goal_radius * 1.5:
                    reached = False      # pushed away / goal moved: drive again
                v, w = env.robot.measured_velocity(env.data)
                status = "REACHED" if reached else "driving"
                print(f"\r[{status:7s}] goal ({goal[0]:+.2f},{goal[1]:+.2f}) dist {inf['goal_dist']:4.2f} m | "
                      f"v={v:+.2f} w={w:+.2f} | clearance {max(inf['clearance'], 0):4.2f} m   ",
                      end="", flush=True)
            with viewer.lock():
                viewer.user_scn.ngeom = 0
                add_overlays(viewer.user_scn, env, trail[-400:], state["rays"])
            viewer.sync()
            time.sleep(max(0.0, env.cfg.ctrl_dt - (time.time() - t0)))
    # MuJoCo's passive viewer can segfault during interpreter teardown on Linux; exit directly.
    print()
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
