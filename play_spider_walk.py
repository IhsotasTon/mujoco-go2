"""播放行走+避障策略。先点 MuJoCo 窗口。

  小键盘 0  重置
关闭窗口退出。
"""

from __future__ import annotations

import time
from pathlib import Path

from stable_baselines3 import PPO

from spider_input import any_down
from spider_walk_env import SpiderWalkEnv

ZIP = Path(__file__).resolve().parent / "spider_quad" / "spider_walk_ppo.zip"


def main() -> None:
    if not ZIP.exists():
        raise SystemExit(f"还没有权重：{ZIP}\n先跑 python train_spider_walk.py")

    env = SpiderWalkEnv(render_mode="human")
    model = PPO.load(str(ZIP.with_suffix("")), env=env)
    obs, _ = env.reset()
    print(__doc__)
    print("只走路绕柱，没有空翻键。")

    last_poll = 0.0
    reset = False
    while True:
        now = time.time()
        if now - last_poll > 0.05:
            if any_down("NUM0", "INSERT"):
                reset = True
            last_poll = now
        if reset:
            reset = False
            obs, _ = env.reset()
            print("已重置")
            continue
        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()


if __name__ == "__main__":
    main()
