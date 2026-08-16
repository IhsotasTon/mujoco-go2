"""键盘控制 Unitree Go2 + 倾斜保护（歪太多就减速/停下）.

按住：
  W / ↑     前进
  S / ↓     后退
  A / ←     左转
  D / →     右转
  Space     强制停止
  R         摔倒后重置站立

保护逻辑（用身体倾斜角度）：
  有点歪  → 自动减速
  很歪    → 强制站立（不再踏步）
  几乎倒地 → 同样急停，按 R 重置
"""

import ctypes
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "unitree_go2" / "scene_with_sensors.xml"

KP = 110.0
KD = 3.0
GAIT_FREQ = 2.4
STEP_H = 0.28
STEP_L = 0.42
TURN_GAIN = 0.55
HIP_YAW = 0.10
CROUCH = 0.08

# 倾斜保护阈值（单位：度）——可以自己改着玩
TILT_SLOW_DEG = 15.0   # 超过就开始减速
TILT_STOP_DEG = 28.0   # 超过就强制停步
HEIGHT_FALLEN = 0.15   # 身体太低，当成摔倒了

VK = {
    "W": 0x57,
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
    "UP": 0x26,
    "DOWN": 0x28,
    "LEFT": 0x25,
    "RIGHT": 0x27,
    "SPACE": 0x20,
    "R": 0x52,
}

FL, FR, RL, RR = 0, 3, 6, 9
HIP, THIGH, CALF = 0, 1, 2

_cmd = {"vx": 0.0, "wz": 0.0}


def key_down(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def quat_to_rpy_deg(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr, cosr))

    sinp = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny, cosy))
    return float(roll), float(pitch), float(yaw)


def sensor_adr(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return int(model.sensor_adr[sid])


def tilt_from_imu(model: mujoco.MjModel, data: mujoco.MjData, adr_quat: int) -> tuple[float, float, float]:
    """从背上的 imu 传感器读姿态，算出歪了多少。

    真实机器上也是这样：程序不能“上帝视角”看自己，
    只能问传感器“我现在歪不歪？”
    """
    w, x, y, z = data.sensordata[adr_quat : adr_quat + 4]
    roll, pitch, _yaw = quat_to_rpy_deg(float(w), float(x), float(y), float(z))
    tilt = math.hypot(roll, pitch)
    return abs(roll), abs(pitch), tilt


def safety_scale(
    data: mujoco.MjData, roll: float, pitch: float, tilt: float
) -> tuple[float, str]:
    """根据传感器算出的倾斜/高度，返回速度倍率 0~1 和状态文字。"""
    height = float(data.qpos[2])  # 高度暂时仍用仿真位置；真机可用测距等

    if height < HEIGHT_FALLEN or tilt >= TILT_STOP_DEG:
        return 0.0, "急停(传感器说太歪/摔倒，按R重置)"
    if tilt >= TILT_SLOW_DEG:
        span = TILT_STOP_DEG - TILT_SLOW_DEG
        scale = 1.0 - (tilt - TILT_SLOW_DEG) / span
        scale = float(np.clip(scale, 0.15, 0.7))
        return scale, f"减速(传感器:歪{tilt:.0f}°)"
    return 1.0, "正常"


def read_command_raw() -> tuple[float, float]:
    if key_down(VK["SPACE"]):
        return 0.0, 0.0

    vx = 0.0
    wz = 0.0
    if key_down(VK["W"]) or key_down(VK["UP"]):
        vx -= 1.6
    if key_down(VK["S"]) or key_down(VK["DOWN"]):
        vx += 1.2
    if key_down(VK["A"]) or key_down(VK["LEFT"]):
        wz += 1.0
    if key_down(VK["D"]) or key_down(VK["RIGHT"]):
        wz -= 1.0

    if abs(wz) > 1e-3 and abs(vx) < 1e-3:
        vx = -0.5
    if abs(vx) > 0.8 and abs(wz) > 1e-3:
        wz *= 0.55
        vx *= 0.85
    return vx, wz


def smooth_command(dt: float, scale: float) -> tuple[float, float]:
    raw_vx, raw_wz = read_command_raw()
    raw_vx *= scale
    raw_wz *= scale

    # 急停时更快把指令收掉
    ax = 8.0 if scale < 0.2 else (4.0 if abs(raw_vx) >= abs(_cmd["vx"]) else 6.0)
    aw = 8.0 if scale < 0.2 else (2.5 if abs(raw_wz) >= abs(_cmd["wz"]) else 5.0)
    _cmd["vx"] += np.clip(raw_vx - _cmd["vx"], -ax * dt, ax * dt)
    _cmd["wz"] += np.clip(raw_wz - _cmd["wz"], -aw * dt, aw * dt)

    if raw_vx == 0.0 and raw_wz == 0.0:
        _cmd["vx"] *= 0.8
        _cmd["wz"] *= 0.8
        if abs(_cmd["vx"]) < 0.02:
            _cmd["vx"] = 0.0
        if abs(_cmd["wz"]) < 0.02:
            _cmd["wz"] = 0.0
    return _cmd["vx"], _cmd["wz"]


def body_forward_speed(data: mujoco.MjData) -> float:
    w, x, y, z = data.qpos[3:7]
    xx = 1 - 2 * (y * y + z * z)
    xy = 2 * (x * y + w * z)
    xz = 2 * (x * z - w * y)
    v = data.qvel[0:3]
    return float(v[0] * xx + v[1] * xy + v[2] * xz)


def trot_targets(q_stand: np.ndarray, t: float, vx: float, wz: float) -> np.ndarray:
    q_des = q_stand.copy()
    if abs(vx) < 1e-3 and abs(wz) < 1e-3:
        return q_des

    thigh_ids = [1, 4, 7, 10]
    calf_ids = [2, 5, 8, 11]
    q_des[thigh_ids] += CROUCH
    q_des[calf_ids] -= 1.6 * CROUCH

    phase = 2.0 * math.pi * GAIT_FREQ * t
    s1 = math.sin(phase)
    s2 = math.sin(phase + math.pi)
    step_scale = max(min(abs(vx) / 1.2, 1.0), 0.55 * min(abs(wz), 1.0), 0.4)

    turn_scale = 1.0 / (1.0 + 0.8 * abs(vx))
    turn_g = TURN_GAIN * turn_scale
    hip_g = HIP_YAW * turn_scale

    def apply_leg(base: int, swing: float, side: float) -> None:
        stride = STEP_L * (vx + turn_g * wz * side)
        lift = STEP_H * max(swing, 0.0) * step_scale
        stance_push = -stride * max(-swing, 0.0)
        q_des[base + THIGH] += lift + stance_push
        q_des[base + CALF] -= 1.7 * lift - 0.55 * stance_push
        q_des[base + HIP] += hip_g * wz * side

    apply_leg(FL, s1, +1.0)
    apply_leg(RR, s1, -1.0)
    apply_leg(FR, s2, -1.0)
    apply_leg(RL, s2, +1.0)
    return q_des


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    q_stand = data.qpos[7:].copy()
    adr_quat = sensor_adr(model, "imu_quat")  # 背上 imu 的“朝向”通道

    print(__doc__)
    print("倾斜保护读的是背上 imu 传感器（和 sensors_go2.py 同源）。")
    print(
        f"阈值：歪>{TILT_SLOW_DEG:.0f}°减速，"
        f"歪>{TILT_STOP_DEG:.0f}°或高度<{HEIGHT_FALLEN}m 急停\n"
    )

    last_print = -1.0
    r_was_down = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            r_down = key_down(VK["R"])
            if r_down and not r_was_down:
                mujoco.mj_resetDataKeyframe(model, data, 0)
                _cmd["vx"] = 0.0
                _cmd["wz"] = 0.0
                print("\n>>> 已重置回站立\n")
            r_was_down = r_down

            # 先走一步物理，传感器数据才会更新
            # （下面先用上一时刻的传感器做保护，再 step——对入门够用）
            roll, pitch, tilt = tilt_from_imu(model, data, adr_quat)
            scale, safety = safety_scale(data, roll, pitch, tilt)
            vx, wz = smooth_command(dt, scale)

            q_des = trot_targets(q_stand, data.time, vx, wz)
            q = data.qpos[7:]
            dq = data.qvel[6:]
            tau = KP * (q_des - q) - KD * dq

            for i in range(model.nu):
                lo, hi = model.actuator_ctrlrange[i]
                tau[i] = float(np.clip(tau[i], lo, hi))

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print > 0.35:
                last_print = data.time
                roll, pitch, tilt = tilt_from_imu(model, data, adr_quat)
                height = float(data.qpos[2])
                bv = body_forward_speed(data)
                print(
                    f"[{safety}] IMU倾斜{tilt:4.1f}° "
                    f"(左右{roll:4.1f}/前后{pitch:4.1f}) "
                    f"高{height:.2f}m  倍率{scale:.2f}  "
                    f"vx={vx:+.2f} body_vx={bv:+.2f}"
                )

            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
