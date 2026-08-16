"""Gymnasium environment for Unitree G1 locomotion in MuJoCo.

`G1Env` loads the G1 scene from the MuJoCo Menagerie and exposes a continuous
control task: walk forward while staying upright, balanced, and smooth. See the
README for the full reward breakdown and observation/action layout.
"""

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np
import time

class G1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path("mujoco_menagerie/unitree_g1/scene.xml")
        self.data = mujoco.MjData(self.model)
        
        self.render_mode = render_mode
        self.viewer = None
        
        # Action space: 29 actuators (or depending on the exact model version, we use model.nu)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        
        # Observation space: 
        # qpos (without x,y of root) + qvel
        obs_size = self.model.nq - 2 + self.model.nv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32)
        
        # Simulation parameters
        self.frame_skip = 10
        self.model.opt.timestep = 0.002
        self.dt = self.model.opt.timestep * self.frame_skip
        
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        
    def step(self, action):
        # Scale action to actuator limits
        ctrlrange = self.model.actuator_ctrlrange
        actuation_center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        actuation_range = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        
        # Clip action to ensure it stays within [-1, 1] bounds before scaling
        action = np.clip(action, -1.0, 1.0)
        self.data.ctrl[:] = actuation_center + action * actuation_range
        
        # Save previous state for velocity calculation if needed
        # xy_position_before = self.data.qpos[:2].copy()
        
        # Step physics
        mujoco.mj_step(self.model, self.data, self.frame_skip)
        
        # Observations
        obs = self._get_obs()
        
        # Reward calculation
        # 1. Forward velocity reward (x-axis)
        x_vel = self.data.qvel[0]
        forward_reward = 1.0 * x_vel
        
        # 2. Healthy reward (not fallen)
        z_pos = self.data.qpos[2] # Pelvis height
        torso_z = self.data.body("torso_link").xpos[2]
        l_hand_z = self.data.body("left_wrist_roll_link").xpos[2]
        r_hand_z = self.data.body("right_wrist_roll_link").xpos[2]
        
        # Ensure it doesn't fall, bend over too much, or touch the ground with hands
        is_healthy = bool(
            z_pos > 0.4 and z_pos < 1.2 and 
            torso_z > 0.5 and 
            l_hand_z > 0.2 and r_hand_z > 0.2
        )
        healthy_reward = 1.0 if is_healthy else 0.0
        
        # 3. Posture reward: penalize pitch and roll to keep body straight
        qw, qx, qy, qz = self.data.qpos[3:7]
        upright_penalty = 3.0 * (qx**2 + qy**2)
        
        # 4. "Head held high" reward
        posture_reward = 1.0 * torso_z
        
        # 5. Control cost
        ctrl_cost = 0.1 * np.sum(np.square(action))
        
        # 6. Smoothness penalty (action rate)
        action_rate_penalty = 0.5 * np.sum(np.square(action - self.prev_action))
        self.prev_action = action.copy()
        
        # 7. Joint velocity penalty
        joint_vel_penalty = 0.001 * np.sum(np.square(self.data.qvel[6:]))
        
        reward = forward_reward + healthy_reward + posture_reward - ctrl_cost - upright_penalty - action_rate_penalty - joint_vel_penalty
        
        # Termination
        terminated = not is_healthy
        truncated = False
        
        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "reward_posture": posture_reward,
            "penalty_upright": -upright_penalty,
            "penalty_action_rate": -action_rate_penalty,
            "penalty_joint_vel": -joint_vel_penalty,
            "z_pos": z_pos,
            "torso_z": torso_z,
        }
        
        if self.render_mode == "human":
            self.render()
            
        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        mujoco.mj_resetData(self.model, self.data)
        
        # Add slight noise to initial state
        qpos_noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel_noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        
        self.data.qpos[:] = self.model.qpos0 + qpos_noise
        self.data.qvel[:] = qvel_noise
        
        # Ensure quaternion is valid for the free joint (indices 3 to 6)
        mujoco.mju_normalize4(self.data.qpos[3:7])
        
        mujoco.mj_forward(self.model, self.data)
        
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        
        obs = self._get_obs()
        return obs, {}

    def _get_obs(self):
        # Exclude x and y position for translation invariance (indices 0 and 1)
        qpos = self.data.qpos[2:].copy()
        qvel = self.data.qvel.copy()
        return np.concatenate([qpos, qvel])

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            # Add a small delay to allow the viewer thread to process events and render
            time.sleep(self.dt)
            
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
