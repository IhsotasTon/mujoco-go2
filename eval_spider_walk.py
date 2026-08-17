"""无窗口检查行走策略：别靠眼睛，先看数字。

  python eval_spider_walk.py
  python eval_spider_walk.py --zip spider_quad/spider_walk_ppo.zip

成功大概长这样（不是保证，是门槛）：
  frac_upright > 0.80
  mean_vx      > 0.12
  mean_abs_wy  < 1.5
  mean_steps   > 400
狂翻 / 一两秒就倒：停训，别继续砸时间。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from spider_walk_env import SpiderWalkEnv

ZIP = Path(__file__).resolve().parent / "spider_quad" / "spider_walk_ppo.zip"


def evaluate(zip_path: Path, n_ep: int = 8, max_steps: int = 600) -> dict:
    env = SpiderWalkEnv(render_mode=None)
    model = PPO.load(str(zip_path.with_suffix("")), env=env, device="cpu")
    rows = []
    for _ in range(n_ep):
        obs, _ = env.reset()
        ups, vxs, wys, hits = [], [], [], []
        x0 = float(env.data.qpos[0])
        steps = 0
        for _t in range(max_steps):
            act, _ = model.predict(obs, deterministic=True)
            obs, _r, term, trunc, info = env.step(act)
            ups.append(info["up"])
            vxs.append(info["vx"])
            wys.append(info["abs_wy"])
            hits.append(info["hit"])
            steps += 1
            if term or trunc:
                break
        rows.append(
            dict(
                dx=float(env.data.qpos[0]) - x0,
                mean_vx=float(np.mean(vxs)),
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
    print("=== 行走评估 ===")
    for k, v in stats.items():
        print(f"  {k:16s} {v:8.3f}")
    ok = (
        stats["frac_upright"] > 0.80
        and stats["mean_vx"] > 0.12
        and stats["mean_abs_wy"] < 1.5
        and stats["steps"] > 400
    )
    print("门槛：", "过了，可以看窗口" if ok else "没过，还在摔/翻，先别当成功")


if __name__ == "__main__":
    main()
