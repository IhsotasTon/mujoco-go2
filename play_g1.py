"""G1 下一课：站着玩几个动作 + 看传感器。

说明：
  人形完整走路很难，这一步先感受“发目标角度 → 机器人动”。
  G1 的电机是位置电机：ctrl = 希望关节转到的角度。

按键（先点一下窗口）：
  W / S     双手往上抬 / 放下
  A / D     腰往左扭 / 往右扭
  I/K/J/L   推身体（和 stand_g1 一样）
  Space     动作清零，只保持站立
  R         整个人重置回站立
  关闭窗口退出

终端会显示：高度、歪了多少、上身传感器、当前状态。
"""

import ctypes
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = Path(__file__).resolve().parent / "unitree_g1" / "scene.xml"
PUSH = 120.0
TILT_WARN = 12.0
TILT_BAD = 25.0

VK = {
    "W": 0x57,
    "A": 0x41,
    "S": 0x53,
    "D": 0x44,
    "I": 0x49,
    "K": 0x4B,
    "J": 0x4A,
    "L": 0x4C,
    "SPACE": 0x20,
    "R": 0x52,
}

# 这些名字在 g1.xml 的 <actuator> 里
ARM_L = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_elbow_joint",
]
ARM_R = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_elbow_joint",
]
WAIST_YAW = "waist_yaw_joint"


def key_down(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    return int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name))


def sensor_adr(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return int(model.sensor_adr[sid])


def quat_to_rpy_deg(w, x, y, z):
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return float(roll), float(pitch), float(yaw)


def read_push() -> np.ndarray:
    f = np.zeros(3)
    if key_down(VK["I"]):
        f[0] += PUSH
    if key_down(VK["K"]):
        f[0] -= PUSH
    if key_down(VK["J"]):
        f[1] += PUSH
    if key_down(VK["L"]):
        f[1] -= PUSH
    return f


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    stand = model.key_ctrl[0].copy()
    data.ctrl[:] = stand

    ids = {
        "l_pitch": actuator_id(model, ARM_L[0]),
        "l_roll": actuator_id(model, ARM_L[1]),
        "l_elbow": actuator_id(model, ARM_L[2]),
        "r_pitch": actuator_id(model, ARM_R[0]),
        "r_roll": actuator_id(model, ARM_R[1]),
        "r_elbow": actuator_id(model, ARM_R[2]),
        "waist": actuator_id(model, WAIST_YAW),
    }
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    adr_gyro = sensor_adr(model, "imu-torso-angular-velocity")
    adr_acc = sensor_adr(model, "imu-torso-linear-acceleration")

    # 动作叠加量（在站立姿势上慢慢加减）
    arm_up = 0.0
    waist = 0.0

    print(__doc__)

    last_print = -1.0
    r_was = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            dt = model.opt.timestep

            r_down = key_down(VK["R"])
            if r_down and not r_was:
                mujoco.mj_resetDataKeyframe(model, data, 0)
                arm_up = 0.0
                waist = 0.0
                data.ctrl[:] = stand
                print("\n>>> 已重置站立\n")
            r_was = r_down

            if key_down(VK["SPACE"]):
                arm_up *= 0.9
                waist *= 0.9

            # 按住慢慢变，松手保持当前姿势
            if key_down(VK["W"]):
                arm_up = min(1.0, arm_up + 0.8 * dt)
            if key_down(VK["S"]):
                arm_up = max(0.0, arm_up - 0.8 * dt)
            if key_down(VK["A"]):
                waist = min(0.6, waist + 0.8 * dt)
            if key_down(VK["D"]):
                waist = max(-0.6, waist - 0.8 * dt)

            ctrl = stand.copy()
            # 抬手：肩向前抬一点、肘弯一点（数值是弧度）
            ctrl[ids["l_pitch"]] = stand[ids["l_pitch"]] - 1.0 * arm_up
            ctrl[ids["r_pitch"]] = stand[ids["r_pitch"]] - 1.0 * arm_up
            ctrl[ids["l_elbow"]] = stand[ids["l_elbow"]] + 0.8 * arm_up
            ctrl[ids["r_elbow"]] = stand[ids["r_elbow"]] + 0.8 * arm_up
            ctrl[ids["l_roll"]] = stand[ids["l_roll"]] + 0.3 * arm_up
            ctrl[ids["r_roll"]] = stand[ids["r_roll"]] - 0.3 * arm_up
            ctrl[ids["waist"]] = stand[ids["waist"]] + waist

            data.ctrl[:] = ctrl
            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[pelvis_id, :3] = read_push()

            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print >= 0.4:
                last_print = data.time
                roll, pitch, _yaw = quat_to_rpy_deg(*data.qpos[3:7])
                tilt = float(np.hypot(roll, pitch))
                if tilt >= TILT_BAD:
                    state = "很歪（快倒了，按R重置）"
                elif tilt >= TILT_WARN:
                    state = "有点歪"
                else:
                    state = "站得还行"
                gx, gy, gz = data.sensordata[adr_gyro : adr_gyro + 3]
                ax, ay, az = data.sensordata[adr_acc : adr_acc + 3]
                print(
                    f"[{state}] 倾斜{tilt:4.1f}° 高{data.qpos[2]:.2f}m  "
                    f"抬手{arm_up:.2f} 扭腰{waist:+.2f}  "
                    f"陀螺({gx:+.1f},{gy:+.1f},{gz:+.1f})  "
                    f"加速({ax:+.1f},{ay:+.1f},{az:+.1f})"
                )

            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
