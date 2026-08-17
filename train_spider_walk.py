"""只训蜘蛛行走 + 避障。

先短跑看数字，别一上来几百万步：
  python train_spider_walk.py --timesteps 200000

云上（AutoDL，先装 libegl1 等）：
  export MUJOCO_GL=egl
  python train_spider_walk.py --timesteps 1500000 --n-envs 8

权重：spider_quad/spider_walk_ppo.zip
评估：python eval_spider_walk.py
播放：python play_spider_walk.py

这次不保证一次成功。200k 后若直立率低、角速度大，停下来改，别空转 4 小时。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "spider_quad" / "spider_walk_ppo.zip"


def make_env():
    from spider_walk_env import SpiderWalkEnv

    return SpiderWalkEnv()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=0, help="0=Linux 8 / Windows 4")
    parser.add_argument("--dummy", action="store_true", help="单进程，EGL 出问题时用")
    args = parser.parse_args()

    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    n_envs = args.n_envs or (8 if sys.platform.startswith("linux") else 4)
    if args.dummy or n_envs == 1:
        env = DummyVecEnv([make_env] * max(1, n_envs))
        kind = "DummyVecEnv"
    else:
        env = SubprocVecEnv([make_env] * n_envs)
        kind = "SubprocVecEnv"

    n_steps = max(128, 2048 // n_envs)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=n_steps,
        batch_size=256,
        learning_rate=3e-4,
        ent_coef=0.01,
        gamma=0.99,
        device="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    print(f"行走+避障  {kind} x{n_envs}  {args.timesteps} 步 → {SAVE}")
    print("先看评估数字：直立、前进、角速度。过不了门槛就停。")
    model.learn(total_timesteps=args.timesteps)
    SAVE.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(SAVE.with_suffix("")))
    print(f"已保存 {SAVE}")
    env.close()

    from eval_spider_walk import evaluate

    stats = evaluate(SAVE, n_ep=6)
    print("=== 训练结束评估 ===")
    for k, v in stats.items():
        print(f"  {k:16s} {v:8.3f}")


if __name__ == "__main__":
    main()
