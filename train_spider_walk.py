"""只训蜘蛛稳定前进步态。避障和后空翻都不进训练。

先短跑看数字，别一上来几百万步：
  python train_spider_walk.py --timesteps 200000

云上（AutoDL，先装 libegl1 等）：
  export MUJOCO_GL=egl
  python train_spider_walk.py --timesteps 200000 --n-envs 8

权重：spider_quad/spider_walk_ppo.zip
评估：python eval_spider_walk.py
播放：python play_spider_walk.py

200k 后先看步态数字：直立、vx、|vy|、|wz|、|y|。过不了门槛就停，别空转去 150 万。
不要用 train_spider.py 混训空翻。
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

    return SpiderWalkEnv(walk_only=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=0, help="0=Linux 8 / Windows 4")
    parser.add_argument("--dummy", action="store_true", help="单进程，EGL 出问题时用")
    parser.add_argument("--resume", action="store_true", help="从已有 zip 接着训")
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
    if args.resume and SAVE.exists():
        model = PPO.load(str(SAVE.with_suffix("")), env=env, device="auto")
        print(f"续训 {SAVE}")
    else:
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
    print(f"只训步态（柱子停远处）  {kind} x{n_envs}  {args.timesteps} 步 → {SAVE}")
    print("先看评估数字：直立、vx、|vy|、|wz|、|y|。过不了门槛就停。")
    model.verbose = 1
    model.learn(total_timesteps=args.timesteps, reset_num_timesteps=not args.resume)
    SAVE.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(SAVE.with_suffix("")))
    print(f"已保存 {SAVE}")
    env.close()

    from eval_spider_walk import evaluate, print_gait_stats

    stats = evaluate(SAVE, n_ep=6)
    print("=== 训练结束评估 ===")
    print_gait_stats(stats)


if __name__ == "__main__":
    main()
