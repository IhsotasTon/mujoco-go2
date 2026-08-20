"""无窗口检查前进步态：别靠眼睛，先看数字。柱子不参与评估。

  python eval_spider_walk.py
  python eval_spider_walk.py --zip spider_quad/spider_walk_ppo.zip

步态优先门槛（不是保证，是这一阶段的闸门）：
  frac_upright > 0.85
  mean_vx      > 0.15   （明显高于爬行；目标巡航约 0.18，封顶 0.35）
  mean_abs_vy  < 0.12   （别蟹行）
  mean_abs_wz  < 0.45   （别原地甩头 / 南绕）
  mean_abs_y   < 0.35   （别整场偏出 x 轴）
  mean_abs_wy  < 1.2
  mean_steps   > 450
狂翻 / 一两秒就倒 / 横着走完：停训，别继续砸时间。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from spider_walk_env import SpiderWalkEnv

ZIP = Path(__file__).resolve().parent / "spider_quad" / "spider_walk_ppo.zip"

# 步态优先闸门。比「几乎能爬」更严，但仍是 200k 短跑该看的数，不是 1.5M 目标。
GAIT_GATES = {
    "frac_upright": 0.85,
    "mean_vx": 0.15,
    "mean_abs_vy": 0.12,
    "mean_abs_wz": 0.45,
    "mean_abs_y": 0.35,
    "mean_abs_wy": 1.2,
    "steps": 450.0,
}

GAIT_STAT_KEYS = (
    "dx",
    "mean_vx",
    "mean_abs_vy",
    "mean_abs_wz",
    "mean_abs_y",
    "mean_abs_wy",
    "frac_upright",
    "hit_rate",
    "steps",
)


def make_eval_env(*, walk_only: bool = True) -> SpiderWalkEnv:
    return SpiderWalkEnv(render_mode=None, walk_only=walk_only)


def gait_passed(stats: dict) -> bool:
    return (
        stats["frac_upright"] > GAIT_GATES["frac_upright"]
        and stats["mean_vx"] > GAIT_GATES["mean_vx"]
        and stats["mean_abs_vy"] < GAIT_GATES["mean_abs_vy"]
        and stats["mean_abs_wz"] < GAIT_GATES["mean_abs_wz"]
        and stats["mean_abs_y"] < GAIT_GATES["mean_abs_y"]
        and stats["mean_abs_wy"] < GAIT_GATES["mean_abs_wy"]
        and stats["steps"] > GAIT_GATES["steps"]
    )


def print_gait_stats(stats: dict) -> None:
    for k in GAIT_STAT_KEYS:
        if k in stats:
            print(f"  {k:16s} {stats[k]:8.3f}")


def evaluate(zip_path: Path, n_ep: int = 8, max_steps: int = 600) -> dict:
    from stable_baselines3 import PPO

    env = make_eval_env(walk_only=True)
    model = PPO.load(str(zip_path.with_suffix("")), env=env, device="cpu")
    rows = []
    for _ in range(n_ep):
        obs, _ = env.reset()
        ups, vxs, vys, wzs, ys, wys, hits = [], [], [], [], [], [], []
        x0 = float(env.data.qpos[0])
        steps = 0
        for _t in range(max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(act)
            ups.append(info["up"])
            vxs.append(info["vx"])
            vys.append(info["abs_vy"])
            wzs.append(info["abs_wz"])
            ys.append(info["abs_y"])
            wys.append(info["abs_wy"])
            hits.append(info["hit"])
            steps += 1
            if term or trunc:
                break
        rows.append(
            dict(
                dx=float(env.data.qpos[0]) - x0,
                mean_vx=float(np.mean(vxs)),
                mean_abs_vy=float(np.mean(vys)),
                mean_abs_wz=float(np.mean(wzs)),
                mean_abs_y=float(np.mean(ys)),
                mean_abs_wy=float(np.mean(wys)),
                frac_upright=float(np.mean(np.array(ups) > 0.70)),
                hit_rate=float(np.mean(hits)),
                steps=steps,
            )
        )
    env.close()
    out = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, default=ZIP)
    parser.add_argument("--episodes", type=int, default=8)
    args = parser.parse_args()
    if not args.zip.exists():
        raise SystemExit(f"没有权重：{args.zip}")
    stats = evaluate(args.zip, n_ep=args.episodes)
    print("=== 步态评估（无柱） ===")
    print_gait_stats(stats)
    ok = gait_passed(stats)
    print("门槛：", "过了，可以看窗口" if ok else "没过，还在摔/转/横移，先别当成功")


if __name__ == "__main__":
    main()
