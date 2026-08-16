"""让 Unitree G1 保持站立，并读取身上传感器（大白话版）.

G1 = 宇树人形机器人（你说的“人形”；型号是 G1，不是 G2）。

和 Go2 的差别（先记住这一条就行）：
  - Go2 电机要我们自己算力气（力矩）
  - G1 模型自带“转到某个角度”的电机，我们只要告诉它目标角度

操作：
  I/K  前推/后推
  J/L  左推/右推
  U/O  上抬/下压
  R    重置站立
  关闭窗口退出
"""

import ctypes
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = Path(__file__).resolve().parent / "unitree_g1" / "scene.xml"
PUSH = 120.0

VK = {
    "I": 0x49,
    "K": 0x4B,
    "J": 0x4A,
    "L": 0x4C,
    "U": 0x55,
    "O": 0x4F,
    "R": 0x52,
}


def key_down(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


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
    if key_down(VK["U"]):
        f[2] += PUSH
    if key_down(VK["O"]):
        f[2] -= PUSH * 0.5
    return f


def quat_to_rpy_deg(w, x, y, z):
    roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = np.degrees(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return float(roll), float(pitch), float(yaw)


def sensor_adr(model, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return int(model.sensor_adr[sid])


def main():
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)

    # 站立目标角度：来自模型自带的 stand 关键帧
    stand_ctrl = model.key_ctrl[0].copy()
    data.ctrl[:] = stand_ctrl

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    adr_gyro = sensor_adr(model, "imu-torso-angular-velocity")
    adr_acc = sensor_adr(model, "imu-torso-linear-acceleration")

    print(__doc__)
    print("G1 已站立。按 I/J/K/L 推它，看倾斜和传感器变化。\n")

    last_print = -1.0
    r_was = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()

            r_down = key_down(VK["R"])
            if r_down and not r_was:
                mujoco.mj_resetDataKeyframe(model, data, 0)
                data.ctrl[:] = stand_ctrl
                print("\n>>> 已重置站立\n")
            r_was = r_down

            data.xfrc_applied[:] = 0.0
            force = read_push()
            data.xfrc_applied[pelvis_id, :3] = force

            # 位置电机：一直保持站立角度
            data.ctrl[:] = stand_ctrl
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print >= 0.4:
                last_print = data.time
                roll, pitch, yaw = quat_to_rpy_deg(*data.qpos[3:7])
                tilt = float(np.hypot(roll, pitch))
                gx, gy, gz = data.sensordata[adr_gyro : adr_gyro + 3]
                ax, ay, az = data.sensordata[adr_acc : adr_acc + 3]
                push_txt = (
                    f"推力({force[0]:+.0f},{force[1]:+.0f},{force[2]:+.0f})"
                    if np.any(force)
                    else "推力(无，按I/J/K/L推)"
                )
                print("=" * 56)
                print(f"时间 {data.time:5.1f}s   {push_txt}")
                print(f"骨盆高度   {data.qpos[2]:.3f} m")
                print(f"左右歪     {roll:+6.1f}°")
                print(f"前后仰     {pitch:+6.1f}°")
                print(f"总倾斜     {tilt:6.1f}°")
                print(f"朝向       {yaw:+6.1f}°")
                print(f"上身陀螺仪 {gx:+.2f} {gy:+.2f} {gz:+.2f}  （转多快）")
                print(f"上身加速度 {ax:+.2f} {ay:+.2f} {az:+.2f}  （晃多猛）")

            sleep = model.opt.timestep - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
