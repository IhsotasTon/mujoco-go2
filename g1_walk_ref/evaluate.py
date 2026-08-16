"""Render a trained G1 walking policy.

Loads the policy and observation normalizer from ./pretrained/ (override the
paths in main() to inspect an intermediate ./models/ checkpoint) and runs it in
the MuJoCo viewer with deterministic actions.
"""

import time
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from env import G1Env

def main():
    # Path to the trained model and normalizer.
    # The repo ships a pretrained policy in pretrained/ so this runs out of the box.
    # To watch an intermediate checkpoint instead, point model_path at ./models/.
    model_path = "pretrained/g1_ppo_final.zip"
    norm_path = "pretrained/vec_normalize.pkl"

    if not os.path.exists(model_path):
        print(f"Model {model_path} not found. Ensure you have run train.py first.")
        return

    from gymnasium.wrappers import TimeLimit
    # Create the environment with render mode human
    env = TimeLimit(G1Env(render_mode="human"), max_episode_steps=10000)
    
    # Wrap and load normalizer
    vec_env = DummyVecEnv([lambda: env])
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        # Disable updating running stats during evaluation
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print(f"Warning: Normalizer {norm_path} not found. Performance might be degraded.")

    # Load the trained model ("auto" picks CUDA when available, else CPU)
    model = PPO.load(model_path, env=vec_env, device="auto")

    print("Evaluating trained model...")
    obs = vec_env.reset()
    for _ in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        
        # We don't need a hard time.sleep here as MuJoCo passive viewer syncs well 
        # when we call env.render() from within step(), but we can add a small delay 
        # if the simulation runs too fast.
        time.sleep(env.unwrapped.dt)

        if dones[0]:
            print("Episode finished.")
            obs = vec_env.reset()

    env.close()

if __name__ == "__main__":
    main()
