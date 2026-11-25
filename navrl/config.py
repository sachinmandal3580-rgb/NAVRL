"""Environment configuration and curriculum stage presets.

All tunable numbers live here (no magic constants in the env). Reward coefficients
are the provisional values from docs/DESIGN.md section 9.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass
class RewardConfig:
    progress: float = 1.0        # per metre of goal-distance reduction (potential-based shaping)
    goal: float = 10.0           # terminal bonus on success
    collision: float = 10.0      # terminal penalty on collision
    time: float = 0.005          # per-step cost
    smooth: float = 0.001        # * ||a_t - a_{t-1}||^2
    proximity: float = 0.01      # * max(0, 1 - clearance / d_safe)
    d_safe: float = 0.4          # [m] proximity band


@dataclass
class EnvConfig:
    # timing
    ctrl_dt: float = 0.1             # policy rate 10 Hz
    physics_dt: float = 0.01         # validated: <0.3 cm deviation vs 0.0025
    max_steps: int = 300             # 30 s episode limit (truncation)
    # sensing / observation
    n_rays: int = 60
    lidar_range: float = 3.0
    n_stack: int = 1                 # 1 = single scan (Variant A), 3 = stacked (Variant B)
    stack_gap: int = 1               # control steps between stacked scans (3 -> scans 0.3 s apart)
    # task
    goal_radius: float = 0.3
    start_goal_dist: tuple[float, float] = (1.5, 5.0)
    arena_half: float = 3.0
    # obstacles
    n_static: tuple[int, int] = (0, 0)       # inclusive range sampled per episode
    n_dynamic: tuple[int, int] = (0, 0)
    dyn_speed: tuple[float, float] = (0.1, 0.45)
    p_crossing: float = 0.0
    n_walls: tuple[int, int] = (0, 0)        # interior wall segments (stage s4)
    dyn_wander: float = 0.0                  # heading random-walk [rad/sqrt(s)] for moving obstacles
    # training-signal options (v2)
    geodesic_progress: bool = False          # progress = decrease of shortest-path distance around
                                             # walls/static obstacles (no local minima behind walls)
    mix: tuple = ()                          # stage names sampled per episode (anti-forgetting)
    reward: RewardConfig = field(default_factory=RewardConfig)


STAGES: dict[str, dict] = {
    "s1_empty": dict(n_static=(0, 0), n_dynamic=(0, 0)),
    "s2_static": dict(n_static=(3, 8), n_dynamic=(0, 0)),
    "s3_dynamic": dict(n_static=(2, 5), n_dynamic=(1, 4), dyn_speed=(0.1, 0.45), p_crossing=0.5),
    # harder: interior walls (corridors / L-shapes), more + faster + wandering movers,
    # longer episodes because detours around walls are needed
    "s4_complex": dict(n_walls=(2, 6), n_static=(2, 6), n_dynamic=(2, 6), dyn_speed=(0.1, 0.6),
                       p_crossing=0.5, dyn_wander=0.6, max_steps=450),
}


# v2 "better policy" recipe: mixed stages, geodesic progress, scans 0.3 s apart,
# stronger/earlier proximity penalty. Same observation size as v1 (n_stack=3).
STAGES["s5_mix"] = dict(
    mix=("s1_empty", "s2_static", "s3_dynamic", "s3_dynamic", "s4_complex", "s4_complex"),
    geodesic_progress=True, stack_gap=3, max_steps=450,
    # time cost doubled: s4 learned to hover near the goal (17% timeouts even in an empty arena)
    reward=RewardConfig(proximity=0.03, d_safe=0.5, time=0.01),
)

MIX_KEYS = ("n_static", "n_dynamic", "dyn_speed", "p_crossing", "n_walls", "dyn_wander")


def make_config(stage: str = "s2_static", **overrides) -> EnvConfig:
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {list(STAGES)}")
    return replace(EnvConfig(), **{**STAGES[stage], **overrides})
