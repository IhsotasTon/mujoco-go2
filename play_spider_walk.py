"""播放行走+避障策略。

macOS 必须用：
  mjpython play_spider_walk.py

不要用 view_spider.py：那个没有电机，蜘蛛会自己瘫倒。
点格子地板聚焦窗口，不要点蜘蛛（点到身体等于推它）。
倒了或停住后，窗口内按 0 重置。
"""

from __future__ import annotations

import time
from pathlib import Path

import glfw
from stable_baselines3 import PPO

from spider_walk_env import STALL_STEPS, SpiderWalkEnv

ZIP = Path(__file__).resolve().parent / "spider_quad" / "spider_walk_ppo.zip"


def main() -> None:
    if not ZIP.exists():
        raise SystemExit(f"还没有权重：{ZIP}\n先跑 python train_spider_walk.py")

    want_reset = {"v": False}

    def on_key(keycode: int) -> None:
        if keycode in (glfw.KEY_KP_0, glfw.KEY_0, glfw.KEY_INSERT):
            want_reset["v"] = True

    env = SpiderWalkEnv(render_mode="human")
    env.key_callback = on_key
    model = PPO.load(str(ZIP.with_suffix("")), device="cpu")
    obs, _ = env.reset()
    print(__doc__)
    print("策略自己走。终端会写摔倒 / 停住 / 到时。")

    paused = False
    while True:
        t0 = time.time()
        if want_reset["v"]:
            want_reset["v"] = False
            obs, _ = env.reset()
            paused = False
            print("已重置")

        if paused:
            env.render()
            time.sleep(0.02)
            continue

        action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            paused = True
            z = float(env.data.qpos[2])
            if truncated:
                why = "到时"
            elif env.still_steps >= STALL_STEPS:
                why = "停住"
            else:
                why = "摔倒"
            print(
                f"{why}  步数={env.steps}  x={info['x']:.2f}  "
                f"直立={info['up']:.2f}  高度={z:.3f}。按 0 重置。"
            )

        elapsed = time.time() - t0
        time.sleep(max(0.0, env.dt - elapsed))


if __name__ == "__main__":
    main()
