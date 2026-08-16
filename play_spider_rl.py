"""播放训好的多技能蜘蛛。

  小键盘 8 / 方向上 / 顶排 8     爬
  小键盘 7 / 顶排 7             空翻
  倒了会自动切到翻面（不用按键）
  小键盘 0                      重置
关闭窗口退出。
"""

from __future__ import annotations

import time
from pathlib import Path

from stable_baselines3 import PPO

from spider_env import CMD_FLIP, CMD_WALK, SpiderEnv
from spider_input import any_down

ZIP = Path(__file__).resolve().parent / "spider_quad" / "spider_ppo.zip"
_cmd = {"v": CMD_WALK, "reset": False}


def main() -> None:
    if not ZIP.exists():
        raise SystemExit(f"还没有权重：{ZIP}\n先在云上跑 python train_spider.py")

    env = SpiderEnv(render_mode="human")
    model = PPO.load(str(ZIP.with_suffix("")), env=env)
    obs, info = env.reset(options={"command": CMD_WALK})
    print(__doc__)
    print("当前任务：爬。倒地会自己翻面。")

    last_poll = 0.0
    while True:
        now = time.time()
        if now - last_poll > 0.05:
            if any_down("NUM8", "UP", "D8"):
                _cmd["v"] = CMD_WALK
            if any_down("NUM7", "HOME", "D7"):
                _cmd["v"] = CMD_FLIP
            if any_down("NUM0", "INSERT"):
                _cmd["reset"] = True
            last_poll = now

        if _cmd["reset"]:
            _cmd["reset"] = False
            obs, info = env.reset(options={"command": CMD_WALK})
            _cmd["v"] = CMD_WALK
            print("已重置")
            continue

        env.set_command(_cmd["v"])
        action, _ = model.predict(obs, deterministic=True)
        obs, _reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset(options={"command": _cmd["v"]})


if __name__ == "__main__":
    main()
