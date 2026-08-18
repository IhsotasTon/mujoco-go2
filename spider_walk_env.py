"""只训行走 + 避障。播放时可后空翻回正。

倒了直接结束（训练）。播放 recover=True 时倒下会后空翻翻正。
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
ACTION_SCALE = 0.32
MAX_STEPS = 600
VX_CAP = 0.35
VX_COEF = 2.4
ALIVE_BONUS = 0.30
WY_COEF = 1.40
FALL_PENALTY = 3.0
BACKFLIP_UP = 2.3
BACKFLIP_PITCH = -12.5
BACKFLIP_BACK = -0.4
BACKFLIP_STEPS = 45
BACKFLIP_COOLDOWN = 20
STALL_VX = 0.10
STALL_VY = 0.10
STALL_WZ = 0.20
STALL_STEPS = 40
RF_CUTOFF = 1.4
PLAY_OBS_XY = ((1.20, 0.00), (1.70, 0.45), (2.10, -0.40), (2.60, 0.15))
OBS_NAMES = ("obs_0", "obs_1", "obs_2", "obs_3")
RF_NAMES = ("rf_ll", "rf_l", "rf_c", "rf_r", "rf_rr")


def is_idle_motion(vx: float, vy: float, wz: float) -> bool:
    """只有几乎完全不动才算停住。转弯/侧移要留给绕柱。"""
    return abs(vx) < STALL_VX and abs(vy) < STALL_VY and abs(wz) < STALL_WZ


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

    def __init__(self, render_mode=None, recover=False):
        self.model = mujoco.MjModel.from_xml_path(str(XML))
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.recover = recover
        self.viewer = None
        self.key_callback = None
        self.frame_skip = 10
        self.model.opt.timestep = 0.002
        self.dt = self.model.opt.timestep * self.frame_skip

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.q_stand = self.data.qpos[7:].astype(np.float32).copy()
        self.prev_action = np.zeros(self.model.nu, dtype=np.float32)
        self.steps = 0
        self.still_steps = 0
        self.flip_steps = 0
        self.flip_cooldown = 0

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
        if self.render_mode == "human":
            xys = list(PLAY_OBS_XY)
        else:
            rng = self.np_random
            used: list[tuple[float, float]] = [(0.0, 0.0)]
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
        for i, (x, y) in enumerate(xys):
            self.data.mocap_pos[i] = (x, y, 0.11)

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

    def trigger_backflip(self) -> None:
        """一次冲量后空翻。播放里用来回正，训练评估不会走这条。"""
        if self.flip_steps > 0:
            return
        self.flip_steps = BACKFLIP_STEPS
        self.flip_cooldown = BACKFLIP_COOLDOWN
        self.still_steps = 0
        self.data.qvel[0] += BACKFLIP_BACK
        self.data.qvel[2] += BACKFLIP_UP
        self.data.qvel[4] += BACKFLIP_PITCH

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
        rf = self._rangefinders()
        blocked = float(rf[2]) < 0.50
        moving = float(np.clip(max(vx, abs(vy)) / 0.10, 0.0, 1.0)) if blocked else float(
            np.clip(vx / 0.10, 0.0, 1.0)
        )
        if stand:
            # 速度封顶；直立/存活分只给正在走的，避免站桩刷满 600 步
            r += VX_COEF * float(np.clip(vx, 0.0, VX_CAP))
            r += (ALIVE_BONUS + 1.8 * up) * moving
            if vx < 0.08 and not blocked:
                r -= 0.80
            elif vx < 0.08:
                r -= 0.15
            if blocked:
                # 前方有柱：侧移/转弯比停住更赚
                r += 1.4 * abs(vy)
                r += 0.5 * abs(wz)
        else:
            r -= 1.6
        r += 0.15 * up
        r -= WY_COEF * abs(wy)
        r -= 0.35 * abs(wx)
        r -= 0.08 * abs(wz)
        r -= 0.08 * abs(vy)
        r -= 0.02 * energy
        r -= 0.01 * smooth
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
        grav0 = projected_gravity(self.data.qpos[3:7])
        up0 = float(grav0[2])
        if self.recover and self.flip_steps == 0 and self.flip_cooldown == 0 and up0 < 0.35:
            self.trigger_backflip()
        flipping = self.flip_steps > 0
        if flipping:
            action = np.zeros(self.model.nu, dtype=np.float32)

        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        target = self.q_stand + action * ACTION_SCALE
        self.data.xfrc_applied[:] = 0
        for _ in range(self.frame_skip):
            q = self.data.qpos[7:]
            dq = self.data.qvel[6:]
            tau = KP * (target - q) - KD * dq
            lo = self.model.actuator_ctrlrange[:, 0]
            hi = self.model.actuator_ctrlrange[:, 1]
            self.data.ctrl[:] = np.clip(tau, lo, hi)
            mujoco.mj_step(self.model, self.data)

        grav = projected_gravity(self.data.qpos[3:7])
        up = float(grav[2])
        z = float(self.data.qpos[2])
        if self.recover and self.flip_steps > 0 and up > 0.65 and 0.08 < z < 0.55:
            self.data.qvel[3:6] *= 0.12
            self.data.qvel[2] = min(float(self.data.qvel[2]), 0.15)
            self.flip_steps = 0

        self.steps += 1
        if self.flip_steps > 0:
            self.flip_steps -= 1
        if self.flip_cooldown > 0 and self.flip_steps == 0:
            self.flip_cooldown -= 1
        hit = self._hit_obstacle()
        reward, info = self._reward(action, hit)
        info["flip"] = float(self.flip_steps > 0)
        self.prev_action = action.copy()
        obs = self._get_obs()

        grav = projected_gravity(self.data.qpos[3:7])
        up = float(grav[2])
        z = float(self.data.qpos[2])
        if self.recover:
            fallen = z < 0.03 or z > 0.70
        else:
            fallen = up < 0.40 or z < 0.07 or z > 0.38
        if flipping:
            self.still_steps = 0
        elif is_idle_motion(float(self.data.qvel[0]), float(self.data.qvel[1]), float(self.data.qvel[5])):
            self.still_steps += 1
        else:
            self.still_steps = 0
        stalled = (not self.recover) and self.still_steps >= STALL_STEPS
        terminated = bool(fallen or stalled)
        truncated = self.steps >= MAX_STEPS
        if fallen:
            reward -= FALL_PENALTY
        elif stalled:
            reward -= FALL_PENALTY
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.steps = 0
        self.still_steps = 0
        self.flip_steps = 0
        self.flip_cooldown = 0
        self.prev_action[:] = 0
        self._place_obstacles()
        self.data.qpos[7:] += self.np_random.uniform(-0.03, 0.03, size=self.model.nu)
        self.data.qvel[:] = self.np_random.uniform(-0.02, 0.02, size=self.model.nv)
        mujoco.mj_forward(self.model, self.data)
        if self.render_mode == "human":
            self.render()
        info = {"x": 0.0, "vx": 0.0, "up": 1.0, "abs_wy": 0.0, "hit": 0.0, "stand": 1.0}
        return self._get_obs(), info

    def render(self):
        if self.viewer is None:
            import mujoco.viewer

            self.viewer = mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self.key_callback,
                show_left_ui=False,
                show_right_ui=False,
            )
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
