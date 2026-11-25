"""Obstacle placement and motion (static + constant-velocity moving obstacles).

All obstacles are mocap bodies declared in assets/arena.xml (pools 'sobs*' and
'dobs*'). Unused ones are parked below the floor where LiDAR/contacts never see them.

Moving obstacles: constant velocity, reflecting off the arena walls, off the
bounding circles of static obstacles, and off each other (equal-mass elastic
bounce). Positions are advanced every physics substep so they never teleport.
Optionally one moving obstacle is a scripted "crossing" obstacle aimed at the
robot's straight-line path to the goal. With `wander` > 0 moving obstacles also
random-walk their heading (pedestrian-like, harder to predict).

Interior walls (pool 'iwall*', stage s4): thin box segments, mostly axis-aligned,
placed at random. Obstacles avoid them and moving obstacles bounce off them. After
placement a grid flood-fill (robot radius inflated) checks that the goal is still
reachable from the start; layouts without a path are resampled.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

PARK = np.array([0.0, 0.0, -5.0])


@dataclass
class Obstacle:
    mocap_id: int
    radius: float          # bounding-circle radius [m]
    height: float          # half-height [m] (for z placement)
    pos: np.ndarray        # (2,) xy
    vel: np.ndarray        # (2,) xy velocity (zero for static)
    yaw: float = 0.0
    active: bool = False


@dataclass
class Wall:
    mocap_id: int
    half_len: float
    half_thick: float
    height: float
    pos: np.ndarray
    yaw: float = 0.0
    active: bool = False

    @property
    def ends(self) -> tuple[np.ndarray, np.ndarray]:
        d = self.half_len * np.array([np.cos(self.yaw), np.sin(self.yaw)])
        return self.pos - d, self.pos + d


def seg_closest(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    t = np.clip((p - a) @ ab / (ab @ ab + 1e-12), 0.0, 1.0)
    return a + t * ab


def _pool(model: mujoco.MjModel, prefix: str) -> list[Obstacle]:
    obs = []
    i = 0
    while True:
        try:
            b = model.body(f"{prefix}{i}")
        except KeyError:
            break
        g = b.geomadr[0]
        size = model.geom_size[g]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX:
            radius, h = float(np.hypot(size[0], size[1])), float(size[2])
        else:  # cylinder
            radius, h = float(size[0]), float(size[1])
        obs.append(Obstacle(int(model.body_mocapid[b.id]), radius, h, np.zeros(2), np.zeros(2)))
        i += 1
    return obs


class ObstacleManager:
    def __init__(self, model: mujoco.MjModel, half_size: float = 3.0, robot_radius: float = 0.15):
        self.half = half_size
        self.robot_radius = robot_radius
        self.static = _pool(model, "sobs")
        self.dynamic = _pool(model, "dobs")
        self.walls: list[Wall] = []
        i = 0
        while True:
            try:
                b = model.body(f"iwall{i}")
            except KeyError:
                break
            sz = model.geom_size[b.geomadr[0]]
            self.walls.append(Wall(int(model.body_mocapid[b.id]), float(sz[0]), float(sz[1]),
                                   float(sz[2]), np.zeros(2)))
            i += 1
        self.wander = 0.0
        self.geom_ids = np.array([model.body(f"sobs{i}").geomadr[0] for i in range(len(self.static))]
                                 + [model.body(f"dobs{i}").geomadr[0] for i in range(len(self.dynamic))])

    # reset
    def reset(self, rng: np.random.Generator, start: np.ndarray, goal: np.ndarray,
              n_static: int, n_dynamic: int, speed_range: tuple[float, float],
              p_crossing: float = 0.0, robot_clearance: float = 0.6,
              dyn_robot_clearance: float = 1.0, gap: float = 0.45, n_walls: int = 0,
              wander: float = 0.0) -> None:
        self.wander = wander
        for tries in range(30):
            self._reset_once(rng, start, goal, n_static, n_dynamic, speed_range, p_crossing,
                             robot_clearance, dyn_robot_clearance, gap, n_walls)
            if self.path_exists(start, goal):
                self._cache()
                return
            if tries >= 20:   # give up on walls; static obstacles alone almost never block
                n_walls = 0
        # last resort: nothing that blocks
        self._reset_once(rng, start, goal, 0, n_dynamic, speed_range, p_crossing,
                         robot_clearance, dyn_robot_clearance, gap, 0)
        self._cache()

    def _reset_once(self, rng, start, goal, n_static, n_dynamic, speed_range, p_crossing,
                    robot_clearance, dyn_robot_clearance, gap, n_walls) -> None:
        for o in self.static + self.dynamic:
            o.active, o.vel[:] = False, 0.0
        for w in self.walls:
            w.active = False

        # interior walls first: random subset of the pool, mostly axis-aligned
        for idx in rng.permutation(len(self.walls))[:n_walls]:
            w = self.walls[idx]
            for _ in range(50):
                yaw = rng.choice([0.0, np.pi / 2]) if rng.random() < 0.8 else rng.uniform(0, np.pi)
                c = rng.uniform(-self.half + 0.3, self.half - 0.3, 2)
                w.pos[:], w.yaw = c, yaw
                a, b = w.ends
                if np.any(np.abs(a) > self.half) or np.any(np.abs(b) > self.half):
                    continue
                if min(self.wall_dist(start, [w]), self.wall_dist(goal, [w])) < robot_clearance:
                    continue
                w.active = True
                break

        placed: list[tuple[np.ndarray, float]] = []   # (pos, radius) of everything placed
        walls = [w for w in self.walls if w.active]

        def ok(p, r, min_robot):
            if np.any(np.abs(p) > self.half - r - 0.05):
                return False
            if np.linalg.norm(p - start) < r + min_robot:
                return False
            if np.linalg.norm(p - goal) < r + robot_clearance:
                return False
            if walls and self.wall_dist(p, walls) < r + 0.1:
                return False
            return all(np.linalg.norm(p - q) >= r + rq + gap for q, rq in placed)

        # static obstacles: random subset of the pool
        for idx in rng.permutation(len(self.static))[:n_static]:
            o = self.static[idx]
            for _ in range(100):
                p = rng.uniform(-self.half, self.half, 2)
                if ok(p, o.radius, robot_clearance):
                    o.pos[:], o.yaw, o.active = p, rng.uniform(0, np.pi), True
                    placed.append((o.pos.copy(), o.radius))
                    break

        # moving obstacles
        for k, o in enumerate(self.dynamic[:n_dynamic]):
            speed = rng.uniform(*speed_range)
            if k == 0 and rng.random() < p_crossing:
                p, v = self._crossing(rng, start, goal, speed, o.radius)
                if p is not None and ok(p, o.radius, dyn_robot_clearance):
                    o.pos[:], o.vel[:], o.active = p, v, True
                    placed.append((o.pos.copy(), o.radius))
                    continue
            for _ in range(100):
                p = rng.uniform(-self.half, self.half, 2)
                if ok(p, o.radius, dyn_robot_clearance):
                    th = rng.uniform(0, 2 * np.pi)
                    o.pos[:], o.vel[:], o.active = p, speed * np.array([np.cos(th), np.sin(th)]), True
                    placed.append((o.pos.copy(), o.radius))
                    break

    def _crossing(self, rng, start, goal, speed, radius, v_nom: float = 0.35):
        """Spawn so the obstacle reaches a point on the start-goal line when the robot does."""
        f = rng.uniform(0.4, 0.6)
        P = start + f * (goal - start)
        t_r = np.linalg.norm(P - start) / v_nom
        path = (goal - start) / (np.linalg.norm(goal - start) + 1e-9)
        for _ in range(10):
            ang = rng.uniform(np.pi / 3, 2 * np.pi / 3) * rng.choice([-1, 1])
            c, s = np.cos(ang), np.sin(ang)
            u = np.array([c * path[0] - s * path[1], s * path[0] + c * path[1]])
            spawn = P - speed * t_r * u
            if np.all(np.abs(spawn) < self.half - radius - 0.05):
                return spawn, speed * u
        return None, None

    # geometry
    @staticmethod
    def wall_dist(p: np.ndarray, walls: list[Wall]) -> float:
        """Distance from point p to the nearest wall surface (negative = inside)."""
        return min(np.linalg.norm(p - seg_closest(p, *w.ends)) - w.half_thick for w in walls)

    def path_exists(self, start: np.ndarray, goal: np.ndarray, res: float = 0.05) -> bool:
        """4-connected flood fill on a grid of free robot-centre positions."""
        walls = [w for w in self.walls if w.active]
        stat = [o for o in self.static if o.active]
        if not walls and not stat:
            return True
        n = int(round(2 * self.half / res))
        xs = -self.half + (np.arange(n) + 0.5) * res
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        P = np.stack([X, Y], -1)
        inf = self.robot_radius + 0.03
        free = (np.abs(X) < self.half - inf) & (np.abs(Y) < self.half - inf)
        for w in walls:
            a, b = w.ends
            ab = b - a
            t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)[..., None]
            free &= np.linalg.norm(P - (a + t * ab), axis=-1) > w.half_thick + inf
        for o in stat:
            free &= np.linalg.norm(P - o.pos, axis=-1) > o.radius + inf
        idx = lambda q: tuple(np.clip(((q + self.half) / res).astype(int), 0, n - 1))
        s, g = idx(start), idx(goal)
        free[s] = free[g] = True
        seen = np.zeros_like(free)
        seen[s] = True
        q = deque([s])
        while q:
            i, j = q.popleft()
            if (i, j) == g:
                return True
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                a2, b2 = i + di, j + dj
                if 0 <= a2 < n and 0 <= b2 < n and free[a2, b2] and not seen[a2, b2]:
                    seen[a2, b2] = True
                    q.append((a2, b2))
        return False

    def geodesic_field(self, goal: np.ndarray, res: float = 0.05) -> np.ndarray:
        """Shortest-path distance to the goal around walls + static obstacles, on a grid.

        Grid cell (i, j) has centre (-half + (i+.5)*res, -half + (j+.5)*res). Free = robot centre
        can be there (inflated by robot radius). 8-connected Dijkstra from the goal cell. Blocked
        cells get the value of the nearest free cell + the distance to it, so the field is finite
        everywhere (the robot can graze an inflated cell without the reward exploding).
        Used only for the reward (privileged info); the policy never sees it.
        """
        from scipy import ndimage
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        n = int(round(2 * self.half / res))
        xs = -self.half + (np.arange(n) + 0.5) * res
        X, Y = np.meshgrid(xs, xs, indexing="ij")
        P = np.stack([X, Y], -1)
        inf = self.robot_radius
        free = (np.abs(X) < self.half - inf) & (np.abs(Y) < self.half - inf)
        for w in (w for w in self.walls if w.active):
            a, b = w.ends
            ab = b - a
            t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)[..., None]
            free &= np.linalg.norm(P - (a + t * ab), axis=-1) > w.half_thick + inf
        for o in (o for o in self.static if o.active):
            free &= np.linalg.norm(P - o.pos, axis=-1) > o.radius + inf
        gi = tuple(np.clip(((goal + self.half) / res).astype(int), 0, n - 1))
        free[gi] = True
        idx = -np.ones((n, n), dtype=np.int64)
        idx[free] = np.arange(free.sum())
        rows, cols, wts = [], [], []
        for di, dj in ((1, 0), (0, 1), (1, 1), (1, -1)):
            a = idx[max(0, -di):n - max(0, di), max(0, -dj):n - max(0, dj)]
            b = idx[max(0, di):n - max(0, -di) or None, max(0, dj):n - max(0, -dj) or None]
            m = (a >= 0) & (b >= 0)
            rows.append(a[m]); cols.append(b[m]); wts.append(np.full(m.sum(), res * np.hypot(di, dj)))
        r, c, w = np.concatenate(rows), np.concatenate(cols), np.concatenate(wts)
        N = int(free.sum())
        g = coo_matrix((w, (r, c)), shape=(N, N)).tocsr()
        d = dijkstra(g, directed=False, indices=idx[gi])
        field = np.full((n, n), np.inf)
        field[free] = d
        # unreachable free cells (pockets): fall back to large value
        big = 4 * self.half * 2
        field[np.isinf(field) & free] = big
        dist_to_free, inds = ndimage.distance_transform_edt(~free, return_indices=True)
        field[~free] = field[inds[0][~free], inds[1][~free]] + dist_to_free[~free] * res
        return field

    # motion
    def _cache(self) -> None:
        """Pack active obstacles/walls into arrays once per reset (motion is vectorised)."""
        self._dyn = [o for o in self.dynamic if o.active]
        self._P = np.array([o.pos for o in self._dyn], dtype=float).reshape(-1, 2)
        self._V = np.array([o.vel for o in self._dyn], dtype=float).reshape(-1, 2)
        for i, o in enumerate(self._dyn):       # objects now view the packed arrays
            o.pos, o.vel = self._P[i], self._V[i]
        st = [o for o in self.static if o.active]
        wl = [w for w in self.walls if w.active]
        self._dr = np.array([o.radius for o in self._dyn])
        self._sp = np.array([o.pos for o in st]).reshape(-1, 2)
        self._sr = np.array([o.radius for o in st])
        ends = [w.ends for w in wl]
        self._wa = np.array([e[0] for e in ends]).reshape(-1, 2)
        self._wab = np.array([e[1] - e[0] for e in ends]).reshape(-1, 2)
        self._wt = np.array([w.half_thick for w in wl])
        self._wab2 = (self._wab ** 2).sum(1)
        self._Qs = np.broadcast_to(self._sp, (len(self._dyn),) + self._sp.shape)

    @staticmethod
    def _reflect(P, V, Q, min_d):
        """Push points P (k,2) out of contact with nearest points Q (k,m,2); reflect V."""
        d = P[:, None] - Q                          # (k, m, 2)
        dist = np.sqrt((d * d).sum(-1))
        hit = (dist < min_d) & (dist > 1e-9)
        if not hit.any():
            return
        for i, j in zip(*np.nonzero(hit)):
            n = d[i, j] / dist[i, j]
            if V[i] @ n < 0:
                V[i] -= 2 * (V[i] @ n) * n
            P[i] = Q[i, j] + n * min_d[i, j]

    def step(self, dt: float, rng: np.random.Generator | None = None) -> None:
        if not getattr(self, "_dyn", None):
            return
        dyn = self._dyn
        P, V = self._P, self._V                     # views shared with the Obstacle objects
        if self.wander > 0 and rng is not None:     # pedestrian-like heading random walk
            th = rng.normal(0.0, self.wander * np.sqrt(dt), len(dyn))
            c, s_ = np.cos(th), np.sin(th)
            V[:] = np.stack([c * V[:, 0] - s_ * V[:, 1], s_ * V[:, 0] + c * V[:, 1]], 1)
        P += V * dt
        R = self._dr
        if len(self._wt):                           # interior walls
            t = np.clip(np.einsum("kmd,md->km", P[:, None] - self._wa, self._wab) / self._wab2, 0, 1)
            Q = self._wa + t[..., None] * self._wab
            self._reflect(P, V, Q, R[:, None] + self._wt)
        lim = self.half - R                         # outer walls
        for ax in range(2):
            hi, lo = P[:, ax] > lim, P[:, ax] < -lim
            P[hi, ax], V[hi, ax] = lim[hi], -np.abs(V[hi, ax])
            P[lo, ax], V[lo, ax] = -lim[lo], np.abs(V[lo, ax])
        if len(self._sr):                           # static bounding circles
            self._reflect(P, V, self._Qs, R[:, None] + self._sr)
        k = len(dyn)                                # each other: elastic swap along normal
        if k > 1:
            d = P[:, None] - P
            dist = np.sqrt((d * d).sum(-1))
            for i, j in zip(*np.nonzero(np.triu(dist < R[:, None] + R, 1) & (dist > 1e-9))):
                n = d[i, j] / dist[i, j]
                rel = (V[i] - V[j]) @ n
                if rel < 0:
                    V[i] -= rel * n
                    V[j] += rel * n

    def write(self, data: mujoco.MjData) -> None:
        """Copy obstacle poses into MuJoCo mocap arrays."""
        for o in self.static + self.dynamic:
            if o.active:
                data.mocap_pos[o.mocap_id] = (o.pos[0], o.pos[1], o.height)
                data.mocap_quat[o.mocap_id] = (np.cos(o.yaw / 2), 0, 0, np.sin(o.yaw / 2))
            else:
                data.mocap_pos[o.mocap_id] = PARK
        for w in self.walls:
            if w.active:
                data.mocap_pos[w.mocap_id] = (w.pos[0], w.pos[1], w.height)
                data.mocap_quat[w.mocap_id] = (np.cos(w.yaw / 2), 0, 0, np.sin(w.yaw / 2))
            else:
                data.mocap_pos[w.mocap_id] = PARK

    def write_dynamic(self, data: mujoco.MjData) -> None:
        """Per-substep update: only moving obstacles change."""
        for o in self._dyn:
            data.mocap_pos[o.mocap_id, :2] = o.pos

    def active_walls(self) -> list[tuple[list, list, float]]:
        """(end_a, end_b, half_thickness) for active interior walls."""
        return [(w.ends[0].tolist(), w.ends[1].tolist(), w.half_thick) for w in self.walls if w.active]

    def active_circles(self) -> list[tuple[np.ndarray, float, bool]]:
        """(pos, radius, is_dynamic) for active obstacles (used by plots/baselines)."""
        return ([(o.pos.copy(), o.radius, False) for o in self.static if o.active]
                + [(o.pos.copy(), o.radius, True) for o in self.dynamic if o.active])
