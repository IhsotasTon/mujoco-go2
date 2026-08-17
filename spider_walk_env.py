"""只训行走 + 避障。没有空翻、没有翻面。

倒了直接结束。前进分只在站稳时给。撞柱子扣分。
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
ACTION_SCALE = 0.40
MAX_STEPS = 600
RF_CUTOFF = 1.4
OBS_NAMES = ("obs_0", "obs_1", "obs_2", "obs_3")
RF_NAMES = ("rf_ll", "rf_l", "rf_c", "rf_r", "rf_rr")


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


class SpiderWalkEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 50}

    def __init__(self, render_mode=None):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.viewer = None
        self.frame_skip = 10
        self.model.opt.timestep = 0.002
        self.dt = self.model.opt.timestep * self.frame_skip

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.q_stand = self.data.qpos[7:].astype(np.float32).copy()
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self.steps = 0

        self.obs_gids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in OBS_NAMES
        ]
        self.floor_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.rf_adr = []
        for name in RF_NAMES:
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            self.rf_adr.append(int(self.model.sensor_adr[sid]))

        for name in ("crate_a", "crate_b"):
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.model.geom_pos[gid, 0] = 80.0

        n = self.model.nu
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)
        # grav3 + omega3 + q + dq + prev_a + 5 rangefinders
        obs_dim = 3 + 3 + n * 3 + 5
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    def _place_obstacles(self) -> None:
        rng = self.np_random
        used: list[tuple[float, float]] = [(0.0, 0.0)]
        # 第一个挡在正前方附近，逼它绕
        first_xy = (float(rng.uniform(0.75, 1.15)), float(rng.uniform(-0.22, 0.22)))
        used.append(first_xy)
        xys = [first_xy]
        for _ in range(len(self.obs_gids) - 1):
            for _try in range(40):
                x = float(rng.uniform(0.9, 2.8))
                y = float(rng.uniform(-0.75, 0.75))
                if all((x - ux) ** 2 + (y - uy) ** 2 > 0.28**2 for ux, uy in used):
                    used.append((x, y))
                    xys.append((x, y))
                    break
            else:
                xys.append((2.4, float(rng.uniform(-0.6, 0.6))))
        for gid, (x, y) in zip(self.obs_gids, xys):
            self.model.geom_pos[gid, 0] = x
            self.model.geom_pos[gid, 1] = y
            self.model.geom_pos[gid, 2] = 0.11

    def _rangefinders(self) -> np.ndarray:
        vals = np.zeros(len(self.rf_adr), dtype=np.float32)
        for i, adr in enumerate(self.rf_adr):
            d = float(self.data.sensordata[adr])
            if not np.isfinite(d) or d < 0:
                d = RF_CUTOFF
            vals[i] = np.clip(d / RF_CUTOFF, 0.0, 1.0)
        return vals

    def _get_obs(self) -> np.ndarray:
        grav = projected_gravity(self.data.qpos[3:7])
        omega = self.data.qvel[3:6].astype(np.float32) * 0.2
        q = (self.data.qpos[7:] - self.q_stand).astype(np.float32)
        dq = (self.data.qvel[6:] * 0.05).astype(np.float32)
        return np.concatenate(
            [grav, omega, q, dq, self.prev_action, self._rangefinders()]
        ).astype(np.float32)

    def _hit_obstacle(self) -> bool:
        obs_set = set(self.obs_gids)
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if g1 in obs_set or g2 in obs_set:
                other = g2 if g1 in obs_set else g1
                if other != self.floor_gid:
                    return True
        return False

    def _reward(self, action: np.ndarray, hit: bool) -> tuple[float, dict]:
        grav = projected_gravity(self.data.qpos[3:7])
        up = float(grav[2])
        z = float(self.data.qpos[2])
        vx = float(self.data.qvel[0])
        vy = float(self.data.qvel[1])
        wz = float(self.data.qvel[5])
        wx = float(self.data.qvel[3])
        wy = float(self.data.qvel[4])
        stand = up > 0.70 and 0.11 < z < 0.26
        energy = float(np.square(action).mean())
        smooth = float(np.square(action - self.prev_action).mean())

        r = 0.0
        r += 2.2 * vx if stand else 0.0
        r += 0.6 * up
        r -= 0.45 * abs(wy)
        r -= 0.25 * abs(wx)
        r -= 0.08 * abs(wz)
        r -= 0.08 * abs(vy)
        r -= 0.02 * energy
        r -= 0.01 * smooth
        if not stand:
            r -= 0.8
        if hit:
            r -= 1.5
        info = {
            "x": float(self.data.qpos[0]),
            "vx": vx,
            "up": up,
            "abs_wy": abs(wy),
            "hit": float(hit),
            "stand": float(stand),
        }
        return r, info

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
        hit = self._hit_obstacle()
        reward, info = self._reward(action, hit)
        self.prev_action = action.copy()
        obs = self._get_obs()

        grav = projected_gravity(self.data.qpos[3:7])
        up = float(grav[2])
        z = float(self.data.qpos[2])
        fallen = up < 0.40 or z < 0.07 or z > 0.38
        terminated = bool(fallen)
        truncated = self.steps >= MAX_STEPS
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.steps = 0
        self.prev_action[:] = 0
        self._place_obstacles()
        self.data.qpos[7:] += self.np_random.uniform(-0.03, 0.03, size=self.model.nu)
        self.data.qvel[:] = self.np_random.uniform(-0.02, 0.02, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        info = {"x": 0.0, "vx": 0.0, "up": 1.0, "abs_wy": 0.0, "hit": 0.0, "stand": 1.0}
        return self._get_obs(), info

    def render(self):
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
