"""训练蜘蛛：往前爬 + 空翻 + 倒了自动翻面。

本地试通：
  python train_spider.py --timesteps 4000

云上（AutoDL / RunPod）：
  export MUJOCO_GL=egl
  python train_spider.py --timesteps 1500000

权重：spider_quad/spider_ppo.zip
播放：python play_spider_rl.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SAVE = ROOT / "spider_quad" / "spider_ppo.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_200_000)
    args = parser.parse_args()

    if sys.platform.startswith("linux"):
        os.environ.setdefault("MUJOCO_GL", "egl")

    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from spider_env import SpiderEnv

    env = DummyVecEnv([SpiderEnv])
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        device="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    print(f"开始训练 {args.timesteps} 步（爬 / 空翻 / 翻面）→ {SAVE}")
    model.learn(total_timesteps=args.timesteps)
    SAVE.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(SAVE.with_suffix("")))
    print(f"已保存 {SAVE}")
    env.close()


if __name__ == "__main__":
    main()
