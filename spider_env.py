"""蜘蛛多技能环境：往前爬、空翻、倒了自动翻面。

观测里有一个 3 维指令（爬 / 空翻 / 翻面），策略根据指令做事。
直立爬的时候如果翻过去了，指令会自动切成「翻面」。
"""

from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

XML = Path(__file__).resolve().parent / "spider_quad" / "spider_quad.xml"

KP = 28.0
KD = 0.9
ACTION_SCALE = 0.45
CMD_WALK, CMD_FLIP, CMD_RIGHT = 0, 1, 2
MAX_STEPS = 500


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """机体坐标里的世界 +Z。站立约 [0, 0, 1]，肚子朝天约 [0, 0, -1]。"""
    w, x, y, z = (float(v) for v in quat)
    return np.array(
        [
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dtype=np.float32,
    )


class SpiderEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        self.frame_skip = 10
        self.model.opt.timestep = 0.002
        self.dt = self.model.opt.timestep * self.frame_skip

        if render_mode is None:
            for name in ("crate_a", "crate_b"):
                gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                if gid >= 0:
                    self.model.geom_pos[gid, 0] = 80.0

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.q_stand = self.data.qpos[7:].astype(np.float32).copy()
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self.command = CMD_WALK
        self.steps = 0
        self.pitch_acc = 0.0
        self.flip_bonus_given = False
        self.right_bonus_given = False

        n = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        obs_dim = 3 + 3 + n * 3 + 3
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _cmd_onehot(self) -> np.ndarray:
        v = np.zeros(3, dtype=np.float32)
        v[int(self.command)] = 1.0
        return v

    def _get_obs(self) -> np.ndarray:
        grav = projected_gravity(self.data.qpos[3:7])
        omega = self.data.qvel[3:6].astype(np.float32) * 0.2
        q = (self.data.qpos[7:] - self.q_stand).astype(np.float32)
        dq = (self.data.qvel[6:] * 0.05).astype(np.float32)
        return np.concatenate(
            [grav, omega, q, dq, self.prev_action, self._cmd_onehot()]
        ).astype(np.float32)

    def set_command(self, cmd: int) -> None:
        self.command = int(np.clip(cmd, 0, 2))

    def _maybe_auto_right(self) -> None:
        """爬的时候一旦肚子朝天，自动改成翻面任务；翻正后再回到爬。"""
        grav = projected_gravity(self.data.qpos[3:7])
        up = float(grav[2])
        if self.command == CMD_WALK and up < 0.15:
            self.command = CMD_RIGHT
            self.right_bonus_given = False
        elif self.command == CMD_RIGHT and self.right_bonus_given and up > 0.8:
            self.command = CMD_WALK

    def _reward(self, action: np.ndarray) -> float:
        grav = projected_gravity(self.data.qpos[3:7])
        up = float(grav[2])
        z = float(self.data.qpos[2])
        vx = float(self.data.qvel[0])
        wy = float(self.data.qvel[4])
        smooth = float(np.square(action - self.prev_action).mean())
        energy = float(np.square(action).mean())
        r = -0.02 * energy - 0.01 * smooth

        if self.command == CMD_WALK:
            # 只有站稳时前进才给分，否则会学会用空翻骗速度
            stand = 1.0 if up > 0.55 and 0.10 < z < 0.30 else 0.0
            r += 2.4 * vx * stand
            r += 1.0 * up
            r -= 0.25 * abs(wy)
            r -= 1.2 * max(0.0, 0.4 - up)
            r -= 0.05 * abs(float(self.data.qvel[5]))
            if stand:
                r += 0.4
        elif self.command == CMD_FLIP:
            # 空翻只奖「转一圈再落地」，转完继续转会扣分
            if not self.flip_bonus_given:
                r += 0.12 * float(np.clip(wy, 0.0, 8.0))
                self.pitch_acc += wy * self.dt
                if self.pitch_acc > 5.5 and up > 0.5 and 0.10 < z < 0.32:
                    r += 18.0
                    self.flip_bonus_given = True
            else:
                r += 1.4 * up + 1.0 * vx
                r -= 0.35 * abs(wy)
        else:
            r += 2.6 * up
            r -= 0.08 * abs(wy) if up > 0.45 else 0.0
            if 0.10 < z < 0.30:
                r += 0.4
            if (not self.right_bonus_given) and up > 0.75 and z > 0.11:
                r += 12.0
                self.right_bonus_given = True
            if self.right_bonus_given:
                r += 0.8 * vx
        return r

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target = self.q_stand + action * ACTION_SCALE
        for _ in range(self.frame_skip):
            q = self.data.qpos[7:]
            dq = self.data.qvel[6:]
            tau = KP * (target - q) - KD * dq
            lo = self.model.actuator_ctrlrange[:, 0]
            hi = self.model.actuator_ctrlrange[:, 1]
            self.data.ctrl[:] = np.clip(tau, lo, hi)
            mujoco.mj_step(self.model, self.data)

        self.steps += 1
        self._maybe_auto_right()
        obs = self._get_obs()
        reward = self._reward(action)
        self.prev_action = action.copy()

        z = float(self.data.qpos[2])
        terminated = z < 0.03 or z > 0.55
        truncated = self.steps >= MAX_STEPS
        if self.render_mode == "human":
            self.render()
        info = {
            "x": float(self.data.qpos[0]),
            "vx": float(self.data.qvel[0]),
            "z": z,
            "cmd": int(self.command),
        }
        return obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.steps = 0
        self.pitch_acc = 0.0
        self.flip_bonus_given = False
        self.right_bonus_given = False
        self.prev_action[:] = 0

        forced = None
        if options and "command" in options:
            forced = int(options["command"])

        roll = float(self.np_random.random()) if forced is None else -1.0
        if forced == CMD_RIGHT or (forced is None and roll < 0.15):
            self.command = CMD_RIGHT
            self.data.qpos[2] = 0.16
            self.data.qpos[3:7] = np.array([0.0, 1.0, 0.0, 0.0])
        elif forced == CMD_FLIP or (forced is None and roll < 0.30):
            self.command = CMD_FLIP
        else:
            self.command = CMD_WALK if forced is None else forced

        self.data.qpos[7:] += self.np_random.uniform(-0.04, 0.04, size=self.model.nu)
        self.data.qvel[:] = self.np_random.uniform(-0.03, 0.03, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"cmd": int(self.command)}

    def render(self):
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
