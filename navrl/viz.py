"""Visualisation helpers: scene overlays (LiDAR rays, path trail) and trajectory plots."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

_EYE = np.eye(3).flatten()


def _line(scene: mujoco.MjvScene, a, b, rgba, width=0.01) -> None:
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), _EYE,
                        np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
    scene.ngeom += 1


def add_overlays(scene: mujoco.MjvScene, env, trail: list, show_rays: bool = True) -> None:
    """Draw LiDAR rays (green -> red as range shrinks) and the robot's path trail."""
    if show_rays and getattr(env, "_last_scan", None) is not None:
        x, y, yaw = env.robot.pose(env.data)
        ranges = env._last_scan
        origin = env.data.site_xpos[env.lidar.site_id]
        ends = env.lidar.endpoints(env.data, yaw, ranges)
        for r, e in zip(ranges, ends):
            t = float(np.clip(r / env.cfg.lidar_range, 0, 1))
            _line(scene, origin, e, (1 - t, t, 0.1, 0.35), 0.004)
    for a, b in zip(trail[:-1], trail[1:]):
        _line(scene, (a[0], a[1], 0.01), (b[0], b[1], 0.01), (0.1, 0.3, 0.9, 0.8), 0.012)


def plot_trajectories(results: list[dict], out_path: Path, title: str, max_eps: int = 6,
                      arena_half: float = 3.0) -> None:
    """Top-down plots of a few episodes: path, start, goal, obstacles (final positions)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = results[:max_eps]
    cols = min(3, len(eps)); rows = int(np.ceil(len(eps) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    colors = {"success": "tab:green", "collision": "tab:red", "timeout": "tab:orange"}
    for ax, ep in zip(axes.flat, eps):
        tr = np.array(ep["traj"])
        for (a, b, ht) in ep.get("walls", []):
            ax.plot([a[0], b[0]], [a[1], b[1]], color="0.3", lw=4, solid_capstyle="butt")
        for (pos, rad, dyn) in ep["obstacles"]:
            ax.add_patch(plt.Circle(pos, rad, color="tab:orange" if dyn else "0.45", alpha=0.7))
        ax.plot(tr[:, 0], tr[:, 1], color=colors.get(ep["outcome"], "k"), lw=2)
        ax.plot(*tr[0], "ko", ms=6)
        ax.add_patch(plt.Circle(ep["goal"], 0.3, color="tab:green", alpha=0.3))
        ax.set_xlim(-arena_half, arena_half); ax.set_ylim(-arena_half, arena_half)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"seed {ep['seed']}: {ep['outcome']} ({ep['time_s']:.1f}s)", fontsize=9)
    for ax in list(axes.flat)[len(eps):]:
        ax.axis("off")
    fig.suptitle(title + "  (moving obstacles shown at final position)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
