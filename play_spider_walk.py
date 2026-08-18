"""播放行走+避障策略。先点 MuJoCo 窗口。

  小键盘 0  重置
倒了不会自动刷新，避免柱子和蜘蛛一起闪。
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
    # 不要把带窗口的 env 交给 PPO.load，否则会再包一层并偷偷 reset
    model = PPO.load(str(ZIP.with_suffix("")), device="cpu")
    obs, _ = env.reset()
    print(__doc__)
    print("只走路绕柱。摔倒后按 0 再来。")

    last_poll = 0.0
    paused = False
    while True:
        t0 = time.time()
        if t0 - last_poll > 0.05:
            if any_down("NUM0", "INSERT"):
                obs, _ = env.reset()
                paused = False
                print("已重置")
            last_poll = t0

        if paused:
            env.render()
            time.sleep(0.02)
            continue

        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            paused = True
            why = "摔倒" if terminated else "到时"
            print(f"{why}（x={info['x']:.2f}  直立={info['up']:.2f}）。按小键盘 0 重置。")

        elapsed = time.time() - t0
        time.sleep(max(0.0, env.dt - elapsed))


if __name__ == "__main__":
    main()
