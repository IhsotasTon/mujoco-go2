"""键盘控制机械蜘蛛爬行。

先点 MuJoCo 窗口。不要用 WASD / Space（那是 MuJoCo 的线框/暂停）。

按一次就会继续走，再按 5 停下：
  小键盘 8  或  方向上  或  顶排 8     前进
  小键盘 2  或  方向下                 后退
  小键盘 4  或  方向左                 左转
  小键盘 6  或  方向右                 右转
  小键盘 5                             停止
  小键盘 9  或  顶排 9                 跳跃
  小键盘 7  或  顶排 7                 前空翻
  小键盘 0                             重置

推：双击身体，按住 Ctrl，右键拖。
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import glfw
import mujoco
import mujoco.viewer
import numpy as np

from spider_input import any_down

ROOT = Path(__file__).resolve().parent
XML = ROOT / "spider_quad" / "spider_quad.xml"

KP = 32.0
KD = 1.0
GAIT_FREQ = 2.2
SWING_DUTY = 0.22
COXA_STRIDE = 0.42
FEMUR_LIFT = 0.32
TIBIA_LIFT = 0.12
TURN_COXA = 0.55
YAW_DAMP = 0.22

TILT_SLOW_DEG = 22.0
TILT_STOP_DEG = 40.0
HEIGHT_FALLEN = 0.07
JUMP_DV = 1.45
JUMP_COOLDOWN = 0.7
FLIP_UP = 2.3
FLIP_PITCH = 12.5  # 绕左右轴转，往前翻
FLIP_FWD = 0.5
FLIP_AIR = 0.9
FLIP_COOLDOWN = 1.2

FL, RL, RR, FR = 0, 3, 6, 9
COXA, FEMUR, TIBIA = 0, 1, 2
COXA_SIGN = {FL: -1.0, RL: -1.0, RR: 1.0, FR: 1.0}
SWING_ORDER = (FL, RR, FR, RL)

# 按一次就锁住，直到按停止
_hold = {"vx": 0.0, "wz": 0.0}


def sensor_index(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return int(model.sensor_adr[sid])


def quat_rpy_deg(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr, cosr))
    sinp = float(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    pitch = np.degrees(np.arcsin(sinp))
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny, cosy))
    return float(roll), float(pitch), float(yaw)


def imu_tilt_deg(data: mujoco.MjData, adr_quat: int) -> tuple[float, float, float]:
    w, x, y, z = data.sensordata[adr_quat : adr_quat + 4]
    return quat_rpy_deg(float(w), float(x), float(y), float(z))


def safety_scale(height: float, tilt: float) -> tuple[float, str]:
    if height < HEIGHT_FALLEN or tilt >= TILT_STOP_DEG:
        return 0.0, "急停(按0重置)"
    if tilt >= TILT_SLOW_DEG:
        span = TILT_STOP_DEG - TILT_SLOW_DEG
        scale = float(np.clip(1.0 - (tilt - TILT_SLOW_DEG) / span, 0.2, 0.7))
        return scale, f"减速(歪{tilt:.0f}°)"
    return 1.0, "正常"


def set_cmd(vx: float, wz: float, label: str) -> None:
    if _hold["vx"] == vx and _hold["wz"] == wz:
        return
    _hold["vx"] = vx
    _hold["wz"] = wz
    print(f"指令 → {label}")


def on_key(key: int) -> None:
    if key in (glfw.KEY_KP_8, glfw.KEY_UP, glfw.KEY_8):
        set_cmd(1.0, 0.0, "前进")
    elif key in (glfw.KEY_KP_2, glfw.KEY_DOWN):
        set_cmd(-0.8, 0.0, "后退")
    elif key in (glfw.KEY_KP_4, glfw.KEY_LEFT):
        set_cmd(0.35, -1.0, "左转")
    elif key in (glfw.KEY_KP_6, glfw.KEY_RIGHT):
        set_cmd(0.35, 1.0, "右转")
    elif key == glfw.KEY_KP_5:
        set_cmd(0.0, 0.0, "停止")
    elif key in (glfw.KEY_KP_9, glfw.KEY_9):
        _hold["jump"] = True
    elif key in (glfw.KEY_KP_7, glfw.KEY_7):
        _hold["flip"] = True
    elif key == glfw.KEY_KP_0:
        _hold["reset"] = True


def poll_keys() -> None:
    """窗口回调漏掉时，再用系统按键补一层。"""
    if any_down("NUM5", "CLEAR"):
        set_cmd(0.0, 0.0, "停止")
        return
    if any_down("NUM8", "UP", "D8"):
        set_cmd(1.0, 0.0, "前进")
    elif any_down("NUM2", "DOWN"):
        set_cmd(-0.8, 0.0, "后退")
    elif any_down("NUM4", "LEFT"):
        set_cmd(0.35, -1.0, "左转")
    elif any_down("NUM6", "RIGHT"):
        set_cmd(0.35, 1.0, "右转")
    if any_down("NUM9", "PRIOR", "D9"):
        _hold["jump"] = True
    if any_down("NUM7", "HOME", "D7"):
        _hold["flip"] = True


def crawl_targets(
    q_stand: np.ndarray, t: float, vx: float, wz: float, yaw_rate: float = 0.0
) -> np.ndarray:
    q_des = q_stand.copy()
    if abs(vx) < 1e-3 and abs(wz) < 1e-3:
        return q_des

    speed = max(min(abs(vx), 1.0), 0.45 * min(abs(wz), 1.0), 0.45)
    cycle = math.fmod(t * GAIT_FREQ, 1.0)
    if cycle < 0:
        cycle += 1.0
    damp = 0.15 if abs(wz) > 0.2 else 1.0
    wz_hold = wz - YAW_DAMP * damp * yaw_rate

    for i, base in enumerate(SWING_ORDER):
        local = (cycle - i * 0.25) % 1.0
        if local < SWING_DUTY:
            phase = local / SWING_DUTY
            lift = math.sin(math.pi * phase) * speed
            stride = math.sin(math.pi * (phase - 0.5)) * speed
            q_des[base + FEMUR] -= FEMUR_LIFT * lift
            q_des[base + TIBIA] -= TIBIA_LIFT * lift
        else:
            phase = (local - SWING_DUTY) / (1.0 - SWING_DUTY)
            stride = (0.5 - phase) * 2.0 * speed
        q_des[base + COXA] += COXA_SIGN[base] * COXA_STRIDE * stride * vx
        q_des[base + COXA] += TURN_COXA * stride * wz_hold
    return q_des


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    q_stand = data.qpos[7:].copy()
    adr_quat = sensor_index(model, "imu_quat")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
    _hold["reset"] = False
    _hold["jump"] = False
    _hold["flip"] = False
    last_jump = -10.0
    last_flip = -10.0
    flip_until = -10.0

    print(__doc__)
    print("点窗口后按 8。终端出现「指令 → 前进」就说明按到了。")

    last_print = -1.0
    last_poll = 0.0
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = torso_id
        viewer.cam.distance = 0.9
        viewer.cam.elevation = -25
        viewer.cam.azimuth = -140

        while viewer.is_running():
            step_start = time.time()
            dt = model.opt.timestep

            now = time.time()
            if now - last_poll > 0.05:
                poll_keys()
                last_poll = now

            if _hold.pop("reset", False) or any_down("NUM0", "INSERT"):
                if _hold.get("_r") != True:
                    mujoco.mj_resetDataKeyframe(model, data, 0)
                    set_cmd(0.0, 0.0, "重置")
                    _hold["_r"] = True
            else:
                _hold["_r"] = False

            height = float(data.qpos[2])
            roll, pitch, _yaw = imu_tilt_deg(data, adr_quat)
            tilt = math.hypot(roll, pitch)

            want_jump = _hold.pop("jump", False)
            if want_jump and (data.time - last_jump) > JUMP_COOLDOWN and height > HEIGHT_FALLEN:
                data.qvel[2] += JUMP_DV
                last_jump = data.time
                print("指令 → 跳跃")

            want_flip = _hold.pop("flip", False)
            if want_flip and (data.time - last_flip) > FLIP_COOLDOWN:
                data.qvel[0] += FLIP_FWD
                data.qvel[2] += FLIP_UP
                data.qvel[4] += FLIP_PITCH
                last_flip = data.time
                flip_until = data.time + FLIP_AIR
                print("指令 → 前空翻")

            if data.time < flip_until:
                scale, safety = 1.0, "空翻中"
            else:
                scale, safety = safety_scale(height, tilt)
            vx = _hold["vx"] * scale
            wz = _hold["wz"] * scale

            q_des = crawl_targets(q_stand, data.time, vx, wz, float(data.qvel[5]))
            q = data.qpos[7:]
            dq = data.qvel[6:]
            tau = KP * (q_des - q) - KD * dq
            for i in range(model.nu):
                lo, hi = model.actuator_ctrlrange[i]
                tau[i] = float(np.clip(tau[i], lo, hi))
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print > 0.4:
                last_print = data.time
                going = "前进" if vx > 0.2 else "后退" if vx < -0.2 else "转弯" if abs(wz) > 0.2 else "站住"
                print(
                    f"[{safety}] {going}  x={data.qpos[0]:+.2f}m  "
                    f"高{height:.2f}  歪{tilt:4.1f}°"
                )

            sleep = dt - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
