"""Training callback (rsl_rl-style iterations) and episode metrics.

One *iteration* = one PPO rollout of num_envs x num_steps_per_env env steps followed by
one PPO update. IterationCallback:
  * prints one console line per iteration (return, episode length, outcome rates, fps, ETA)
  * logs every reward term separately + behaviour diagnostics to TensorBoard
  * saves model_<iter>.pt every save_interval iterations
  * runs a deterministic validation eval every eval_interval iterations and keeps
    best_model.pt by success rate (tie-breaks: fewer collisions, then faster time-to-goal)
"""
from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from navrl.checkpoint import save_checkpoint
from navrl.config import EnvConfig
from navrl.env import NavRLEnv


def run_episodes(policy_fn, cfg: EnvConfig, seeds, env: NavRLEnv | None = None) -> list[dict]:
    """Roll out `policy_fn(obs, env) -> action` on the given seeds; return per-episode metrics."""
    env = env or NavRLEnv(cfg)
    results = []
    for seed in seeds:
        obs, info = env.reset(seed=int(seed))
        path, prev, min_clear, steps = 0.0, np.array(info["start"]), np.inf, 0
        traj = [prev.copy()]
        while True:
            obs, r, term, trunc, inf = env.step(policy_fn(obs, env))
            p = np.array(inf["pos"]); path += float(np.linalg.norm(p - prev)); prev = p
            traj.append(p.copy())
            min_clear = min(min_clear, inf["clearance"]); steps += 1
            if term or trunc:
                break
        results.append(dict(seed=int(seed), outcome=inf["outcome"], steps=steps,
                            time_s=steps * cfg.ctrl_dt, path_len=path,
                            # straight-line distance to the edge of the goal disc
                            straight=float(np.linalg.norm(info["goal"] - info["start"]) - cfg.goal_radius),
                            min_clearance=float(max(min_clear, 0.0)),
                            traj=np.array(traj).tolist(), goal=info["goal"].tolist(),
                            obstacles=[(c[0].tolist(), c[1], c[2]) for c in env.obstacles.active_circles()],
                            walls=env.obstacles.active_walls()))
    return results


def summarize(results: list[dict]) -> dict:
    n = len(results)
    succ = [r for r in results if r["outcome"] == "success"]
    nonc = [r for r in results if r["outcome"] != "collision"]
    return dict(
        n=n,
        success=sum(r["outcome"] == "success" for r in results) / n,
        collision=sum(r["outcome"] == "collision" for r in results) / n,
        timeout=sum(r["outcome"] == "timeout" for r in results) / n,
        time_to_goal=float(np.mean([r["time_s"] for r in succ])) if succ else float("nan"),
        path_len=float(np.mean([r["path_len"] for r in succ])) if succ else float("nan"),
        path_ratio=float(np.mean([r["path_len"] / r["straight"] for r in succ])) if succ else float("nan"),
        min_clearance=float(np.mean([r["min_clearance"] for r in nonc])) if nonc else float("nan"),
    )


class IterationCallback(BaseCallback):
    def __init__(self, cfg: EnvConfig, run_dir: Path, steps_per_iter: int, max_iterations: int,
                 save_interval: int = 50, eval_interval: int = 10, eval_episodes: int = 30,
                 eval_seed0: int = 50_000, meta: dict | None = None):
        super().__init__()
        self.meta = meta or {}
        self.cfg, self.run_dir = cfg, Path(run_dir)
        self.steps_per_iter, self.max_iterations = steps_per_iter, max_iterations
        self.save_interval, self.eval_interval = save_interval, eval_interval
        self.eval_seeds = list(range(eval_seed0, eval_seed0 + eval_episodes))
        self.eval_env: NavRLEnv | None = None
        self.best = (-1.0, 0.0, -1e9)
        self._ep_terms: list[dict] = []
        self._episodes: list[dict] = []
        self._beh = defaultdict(list)
        self._t_iter = None
        self._t_start = None
        self._iters_this_session = 0

    # helpers
    @property
    def iteration(self) -> int:
        """Completed PPO updates (also correct after --resume)."""
        return self.model.num_timesteps // self.steps_per_iter

    def _evaluate(self) -> dict:
        if self.eval_env is None:
            self.eval_env = NavRLEnv(self.cfg)
        model = self.model
        s = summarize(run_episodes(lambda o, e: model.predict(o, deterministic=True)[0],
                                   self.cfg, self.eval_seeds, self.eval_env))
        for k in ("success", "collision", "timeout", "time_to_goal", "min_clearance"):
            self.logger.record(f"eval/{k}", s[k])
        # rank: success, then fewer collisions, then faster time-to-goal
        t_goal = s["time_to_goal"] if s["time_to_goal"] == s["time_to_goal"] else 1e9   # nan -> worst
        score = (s["success"], -s["collision"], -t_goal)
        if score > self.best:
            self.best = score
            save_checkpoint(model, self.run_dir / "best_model.pt", self.iteration,
                            {**self.meta, "eval": s})
        return s

    # SB3 hooks
    def _on_training_start(self) -> None:
        self._ep_terms = [defaultdict(float) for _ in range(self.training_env.num_envs)]
        self._t_start = self._t_iter = time.time()

    def _on_rollout_start(self) -> None:
        # the previous iteration's PPO update has finished here
        it = self.iteration
        if it > 0 and self._iters_this_session > 0:
            if it % self.save_interval == 0:
                save_checkpoint(self.model, self.run_dir / f"model_{it}.pt", it, self.meta)
            if it % self.eval_interval == 0:
                s = self._evaluate()
                self.logger.dump(self.num_timesteps)
                print(f"  [eval it {it}] success={s['success']:.2f} collision={s['collision']:.2f} "
                      f"timeout={s['timeout']:.2f} t_goal={s['time_to_goal']:.1f} s  "
                      f"(best success {self.best[0]:.2f})", flush=True)

    def _on_step(self) -> bool:
        infos, dones = self.locals["infos"], self.locals["dones"]
        actions = self.locals.get("clipped_actions", self.locals.get("actions"))
        for i, info in enumerate(infos):
            for k, v in info.get("reward_terms", {}).items():
                self._ep_terms[i][k] += v
            if actions is not None:
                a = np.clip(actions[i], -1, 1)
                self._beh["v"].append(a[0]); self._beh["w"].append(abs(a[1]))
                self._beh["still"].append(float(abs(a[0]) < 0.1 and abs(a[1]) < 0.1))
            if dones[i]:
                ep = dict(self._ep_terms[i]); ep["outcome"] = info.get("outcome", "?")
                self._episodes.append(ep)
                self._ep_terms[i] = defaultdict(float)
        return True

    def _on_rollout_end(self) -> None:
        self._iters_this_session += 1
        it = self.iteration              # rollouts collected so far (this one included)
        now = time.time()
        dt_iter = now - self._t_iter
        self._t_iter = now
        rates = {}
        if self._episodes:
            for k in [k for k in self._episodes[0] if k != "outcome"]:
                self.logger.record(f"reward/{k}", float(np.mean([e.get(k, 0.0) for e in self._episodes])))
            outs = [e["outcome"] for e in self._episodes]
            for o in ("success", "collision", "timeout"):
                rates[o] = outs.count(o) / len(outs)
                self.logger.record(f"train/{o}_rate", rates[o])
            n_eps = len(self._episodes)
            self._episodes = []
        else:
            n_eps = 0
        if self._beh["v"]:
            self.logger.record("behaviour/mean_v_action", float(np.mean(self._beh["v"])))
            self.logger.record("behaviour/mean_abs_w_action", float(np.mean(self._beh["w"])))
            self.logger.record("behaviour/stationary_frac", float(np.mean(self._beh["still"])))
            self._beh.clear()
        self.logger.record("train/iteration", it)

        buf = self.model.ep_info_buffer
        ret = np.mean([e["r"] for e in buf]) if buf else float("nan")
        ep_len = np.mean([e["l"] for e in buf]) if buf else float("nan")
        elapsed = now - self._t_start
        eta = elapsed / self._iters_this_session * max(self.max_iterations - it, 0)
        print(f"it {it:>5}/{self.max_iterations} | {self.steps_per_iter / dt_iter:6.0f} sps "
              f"{dt_iter:5.1f}s/it | ret {ret:7.2f} len {ep_len:5.0f} | eps {n_eps:3d} "
              f"succ {rates.get('success', 0):.2f} col {rates.get('collision', 0):.2f} "
              f"to {rates.get('timeout', 0):.2f} | ETA {eta / 60:5.1f} min", flush=True)

    def _on_training_end(self) -> None:
        it = self.iteration
        save_checkpoint(self.model, self.run_dir / f"model_{it}.pt", it, self.meta)
        if self.eval_env is not None:
            self.eval_env.close()
