"""G1 行走环境 —— 必须和训练时尽量一致，否则策略会扭成麻花。"""

from pathlib import Path

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
from gymnasium import spaces

SCENE = Path(__file__).resolve().parent / "unitree_g1" / "scene.xml"


class G1Env(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(str(SCENE))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32
        )
        obs_size = self.model.nq - 2 + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float32
        )

        self.frame_skip = 10
        self.model.opt.timestep = 0.002
        self.dt = self.model.opt.timestep * self.frame_skip
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)

    def step(self, action):
        # 与训练代码相同的动作缩放方式
        ctrlrange = self.model.actuator_ctrlrange
        center = (ctrlrange[:, 1] + ctrlrange[:, 0]) / 2.0
        half = (ctrlrange[:, 1] - ctrlrange[:, 0]) / 2.0
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.data.ctrl[:] = center + action * half
        mujoco.mj_step(self.model, self.data, self.frame_skip)

        obs = self._get_obs()
        z_pos = float(self.data.qpos[2])
        torso_z = float(self.data.body("torso_link").xpos[2])
        l_hand_z = float(self.data.body("left_wrist_roll_link").xpos[2])
        r_hand_z = float(self.data.body("right_wrist_roll_link").xpos[2])
        is_healthy = bool(
            0.4 < z_pos < 1.2
            and torso_z > 0.5
            and l_hand_z > 0.2
            and r_hand_z > 0.2
        )

        if self.render_mode == "human":
            self.render()

        x_vel = float(self.data.qvel[0])
        reward = (1.0 * x_vel) + (1.0 if is_healthy else 0.0)
        self.prev_action = action.copy()
        info = {
            "z_pos": z_pos,
            "x_pos": float(self.data.qpos[0]),
            "x_vel": x_vel,
        }
        return obs, reward, not is_healthy, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        # 关键：必须和训练时一样从 qpos0 + 小噪声起步
        # （用 stand 关键帧会让策略“看不懂”，动作就会扭曲）
        qpos_noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nq)
        qvel_noise = self.np_random.uniform(low=-0.01, high=0.01, size=self.model.nv)
        self.data.qpos[:] = self.model.qpos0 + qpos_noise
        self.data.qvel[:] = qvel_noise
        mujoco.mju_normalize4(self.data.qpos[3:7])
        mujoco.mj_forward(self.model, self.data)

        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        return self._get_obs(), {}

    def _get_obs(self):
        # 去掉世界坐标 x,y，和训练一致
        return np.concatenate([self.data.qpos[2:].copy(), self.data.qvel.copy()])

    def render(self):
        if self.render_mode != "human":
            return
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        if self.viewer.is_running():
            self.viewer.sync()
        import time

        time.sleep(self.dt)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
