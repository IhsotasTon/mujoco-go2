"""用训练好的 PPO 策略让 G1 走路（纯电机，无外挂力）.

说明：
  这个现成策略并不完美，会往前走一段然后摔倒，属于正常。
  修好后应能看到明显前进，而不是一上来就扭成麻花。

运行：
  python walk_g1.py
"""

import time
from pathlib import Path

from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from g1_env import G1Env

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "g1_walk_ref" / "pretrained" / "g1_ppo_final.zip"
NORM_PATH = ROOT / "g1_walk_ref" / "pretrained" / "vec_normalize.pkl"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到策略文件: {MODEL_PATH}")

    print(__doc__)
    print(f"加载策略: {MODEL_PATH.name}")

    env = TimeLimit(G1Env(render_mode="human"), max_episode_steps=10000)
    vec_env = DummyVecEnv([lambda: env])
    if NORM_PATH.exists():
        vec_env = VecNormalize.load(str(NORM_PATH), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    model = PPO.load(str(MODEL_PATH), env=vec_env, device="cpu")
    print("开始。应能往前走一段；摔倒会重置再来。\n")

    obs = vec_env.reset()
    ep = 1
    best_x = 0.0
    try:
        while True:
            if env.unwrapped.viewer is not None and not env.unwrapped.viewer.is_running():
                break

            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)
            x = float(env.unwrapped.data.qpos[0])
            best_x = max(best_x, x)
            time.sleep(env.unwrapped.dt)

            if dones[0]:
                print(f"第 {ep} 次摔倒，最远大约走到 x={best_x:.2f} m，重置…")
                ep += 1
                best_x = 0.0
                obs = vec_env.reset()
    finally:
        env.close()


if __name__ == "__main__":
    main()
