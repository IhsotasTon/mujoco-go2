"""Sanity-check the G1 environment: open the viewer and step with zero actions.

Useful for verifying the model loads and the simulation runs before training.
The robot simply falls under gravity since no policy is applied.
"""

import time
import mujoco
import mujoco.viewer
from env import G1Env

def main():
    print("Initializing environment...")
    env = G1Env(render_mode="human")
    obs, _ = env.reset()
    
    print("Environment reset. Stepping through simulation...")
    # Loop to just let the robot fall
    for i in range(1000):
        # Apply zero action
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample() * 0)
        
        if terminated or truncated:
            print("Episode ended. Resetting...")
            env.reset()
            
    env.close()
    print("Done.")

if __name__ == "__main__":
    main()
