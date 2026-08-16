"""G1：歪了自动撑住（已修好，不会自己倒下）.

上一版问题：
  脚踝回正方向反了，一点点歪会被越扳越倒。

这一版做法（仍然用传感器闭环）：
  1) 腿保持站立姿势
  2) 读骨盆 IMU：歪了多少、倒得有多快
  3) 程序立刻施加“扶正”的力和力矩，把人撑回来

说明（人话）：
  真机上扶正力来自腿部肌肉/电机；
  入门仿真里先把“扶正”作用在骨盆上，保证你能看清：
      传感器发现歪了 → 程序出力 → 身体回来。

操作：
  I/K/J/L   推一把（别按太久）
  R         若真倒了，重置
  关闭窗口退出
"""

import ctypes
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

SCENE = Path(__file__).resolve().parent / "unitree_g1" / "scene.xml"

# 键盘推力：太大必倒，适中才能看出“撑住”
PUSH = 90.0

# 扶正强度（已在无窗口测试：站着不倒，推一下能回正）
KP_TILT = 500.0
KD_GYRO = 50.0
KF_LEAN = 200.0

VK = {
    "I": 0x49,
    "K": 0x4B,
    "J": 0x4A,
    "L": 0x4C,
    "R": 0x52,
}


def key_down(vk: int) -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(vk) & 0x8000)


def sensor_adr(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return int(model.sensor_adr[sid])


def quat_to_rpy(w, x, y, z):
    roll = float(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)))
    yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
    return roll, pitch, yaw


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

    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    adr_gyro = sensor_adr(model, "imu-pelvis-angular-velocity")

    print(__doc__)
    print("站稳中。短按 J/L 推一下，应显示“正在撑住”然后回到“站稳”。\n")

    last_print = -1.0
    r_was = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            t0 = time.time()
            dt = model.opt.timestep

            r_down = key_down(VK["R"])
            if r_down and not r_was:
                mujoco.mj_resetDataKeyframe(model, data, 0)
                data.ctrl[:] = stand
                print("\n>>> 已重置站立\n")
            r_was = r_down

            # 腿：始终保持站立目标角
            data.ctrl[:] = stand

            # 传感器：倾斜 + 角速度
            roll, pitch, _yaw = quat_to_rpy(*data.qpos[3:7])
            wx, wy, wz = data.sensordata[adr_gyro : adr_gyro + 3]
            tilt_deg = float(np.degrees(np.hypot(roll, pitch)))

            # 扶正：力矩抵抗倾斜，力抵抗重心偏移
            data.xfrc_applied[:] = 0.0
            data.xfrc_applied[pelvis_id, 3] = -KP_TILT * roll - KD_GYRO * wx
            data.xfrc_applied[pelvis_id, 4] = -KP_TILT * pitch - KD_GYRO * wy
            data.xfrc_applied[pelvis_id, 0] += -KF_LEAN * pitch
            data.xfrc_applied[pelvis_id, 1] += -KF_LEAN * roll

            # 你的推力叠加上去
            data.xfrc_applied[pelvis_id, :3] += read_push()

            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print >= 0.35:
                last_print = data.time
                if data.qpos[2] < 0.5 or tilt_deg > 45:
                    state = "撑不住了(按R)"
                elif tilt_deg > 6:
                    state = "正在撑住"
                else:
                    state = "站稳"
                print(
                    f"[{state}] 倾斜{tilt_deg:5.1f}°  "
                    f"高{data.qpos[2]:.2f}m  "
                    f"陀螺({wx:+.2f},{wy:+.2f},{wz:+.2f})  "
                    f"扶正矩({data.xfrc_applied[pelvis_id, 3]:+.0f},"
                    f"{data.xfrc_applied[pelvis_id, 4]:+.0f})"
                )

            sleep = dt - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
