"""Train a PPO walking policy for the Unitree G1 with Stable-Baselines3.

Runs 8 parallel MuJoCo environments under VecNormalize, logs custom reward
terms to TensorBoard, checkpoints to ./models/, and saves the final policy and
observation normalizer to ./pretrained/.
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from env import G1Env

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # Check if 'infos' is available
        if "infos" in self.locals:
            forward_rewards = []
            ctrl_rewards = []
            survive_rewards = []
            z_positions = []
            
            for info in self.locals["infos"]:
                if "reward_forward" in info:
                    forward_rewards.append(info["reward_forward"])
                    ctrl_rewards.append(info["reward_ctrl"])
                    survive_rewards.append(info["reward_survive"])
                    z_positions.append(info["z_pos"])
            
            # Log the mean of the parallel environments
            if len(forward_rewards) > 0:
                import numpy as np
                self.logger.record("custom/reward_forward", np.mean(forward_rewards))
                self.logger.record("custom/reward_ctrl", np.mean(ctrl_rewards))
                self.logger.record("custom/reward_survive", np.mean(survive_rewards))
                self.logger.record("custom/z_pos", np.mean(z_positions))
        return True

from gymnasium.wrappers import TimeLimit

def make_env(rank, seed=0):
    """
    Utility function for multiprocessed env.
    """
    def _init():
        env = G1Env(render_mode=None)
        env = TimeLimit(env, max_episode_steps=10000)
        env.reset(seed=seed + rank)
        return env
    return _init

def main():
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Number of parallel environments
    num_envs = 8
    
    # Create the vectorized environment using SubprocVecEnv for true parallelism
    vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Setup the PPO model
    # We use hyperparameters generally suitable for continuous control / robotics
    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1,
        tensorboard_log=log_dir,
        device="auto"  # CUDA when available, else CPU
    )

    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path='./models/',
        name_prefix='g1_ppo'
    )
    tensorboard_callback = TensorboardCallback()

    print("Starting training...")
    # Train for a small number of timesteps as a test. 
    # For a real policy, this would be in the millions.
    total_timesteps = 1e7
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, tensorboard_callback])
    
    print("Training finished. Saving final model and normalizer...")
    # The final policy + observation normalizer go to pretrained/ (what evaluate.py loads).
    # Intermediate checkpoints are written to ./models/ by the CheckpointCallback above.
    os.makedirs("pretrained", exist_ok=True)
    model.save(os.path.join("pretrained", "g1_ppo_final"))
    vec_env.save(os.path.join("pretrained", "vec_normalize.pkl"))

if __name__ == "__main__":
    main()
