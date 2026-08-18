"""播放行走+避障。倒下会后空翻回正。

macOS 必须用：
  mjpython play_spider_walk.py

  小键盘 7 / 顶排 7     后空翻（倒了也会自动翻）
  小键盘 0             重置
点格子地板聚焦窗口，不要点蜘蛛。
"""

from __future__ import annotations

import time
from pathlib import Path

import glfw
import numpy as np
from stable_baselines3 import PPO

from spider_walk_env import STALL_STEPS, SpiderWalkEnv

ZIP = Path(__file__).resolve().parent / "spider_quad" / "spider_walk_ppo.zip"


def main() -> None:
    if not ZIP.exists():
        raise SystemExit(f"还没有权重：{ZIP}\n先跑 python train_spider_walk.py")

    keys = {"reset": False, "flip": False}

    def on_key(keycode: int) -> None:
        if keycode in (glfw.KEY_KP_0, glfw.KEY_0, glfw.KEY_INSERT):
            keys["reset"] = True
        elif keycode in (glfw.KEY_KP_7, glfw.KEY_7):
            keys["flip"] = True

    env = SpiderWalkEnv(render_mode="human", recover=True)
    env.key_callback = on_key
    model = PPO.load(str(ZIP.with_suffix("")), device="cpu")
    obs, _ = env.reset()
    print(__doc__)
    print("自己走。倒了后空翻回正。7 = 手动后空翻。")

    paused = False
    while True:
        t0 = time.time()
        if keys["reset"]:
            keys["reset"] = False
            obs, _ = env.reset()
            paused = False
            print("已重置")
        if keys["flip"]:
            keys["flip"] = False
            env.trigger_backflip()
            paused = False
            print("后空翻")

        if paused:
            env.render()
            time.sleep(0.02)
            continue

        if env.flip_steps > 0:
            action = np.zeros(env.model.nu, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _r, terminated, truncated, info = env.step(action)
        if truncated:
            paused = True
            print(f"到时  x={info['x']:.2f}。按 0 重置。")
        elif terminated:
            paused = True
            z = float(env.data.qpos[2])
            why = "停住" if env.still_steps >= STALL_STEPS else "掉出范围"
            print(
                f"{why}  步数={env.steps}  x={info['x']:.2f}  "
                f"直立={info['up']:.2f}  高度={z:.3f}。按 0 重置。"
            )

        elapsed = time.time() - t0
        time.sleep(max(0.0, env.dt - elapsed))


if __name__ == "__main__":
    main()
