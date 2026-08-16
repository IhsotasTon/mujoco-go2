"""读机械蜘蛛的“感觉”（站着，可推）.

你会看到：
  - 身体有多高、歪没歪、朝哪边
  - 四只脚谁踩地
  - 背上 IMU：转多快、加速度

操作（先点 MuJoCo 窗口，打开 NumLock；不要用 IJKL，那些是 MuJoCo 自己的显示开关）：
  小键盘 8 / 2     往前推 / 往后推
  小键盘 4 / 6     往左推 / 往右推
  小键盘 9 / 3     往上抬 / 往下压
  小键盘 0         重置回站立
  或：双击身体，Ctrl + 右键拖（MuJoCo 自带推）
  关闭窗口退出
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from spider_input import VK, key_down

ROOT = Path(__file__).resolve().parent
XML = ROOT / "spider_quad" / "spider_quad.xml"

KP = 28.0
KD = 0.9
PUSH = 18.0

FOOT_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
FOOT_CN = {"FL_foot": "左前", "FR_foot": "右前", "RL_foot": "左后", "RR_foot": "右后"}


def read_push() -> np.ndarray:
    f = np.zeros(3)
    if key_down(VK["NUM8"]):
        f[0] += PUSH
    if key_down(VK["NUM2"]):
        f[0] -= PUSH
    if key_down(VK["NUM4"]):
        f[1] += PUSH
    if key_down(VK["NUM6"]):
        f[1] -= PUSH
    if key_down(VK["NUM9"]):
        f[2] += PUSH
    if key_down(VK["NUM3"]):
        f[2] -= PUSH * 0.5
    return f


def quat_to_rpy_deg(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr, cosr))
    sinp = float(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    pitch = np.degrees(np.arcsin(sinp))
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny, cosy))
    return float(roll), float(pitch), float(yaw)


def feet_on_ground(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, bool]:
    foot_ids = {
        name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in FOOT_NAMES
    }
    on_ground = {name: False for name in FOOT_NAMES}
    for i in range(data.ncon):
        g1 = data.contact[i].geom1
        g2 = data.contact[i].geom2
        for name, gid in foot_ids.items():
            if g1 == gid or g2 == gid:
                on_ground[name] = True
    return on_ground


def sensor_index(model: mujoco.MjModel, name: str) -> int:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    return int(model.sensor_adr[sid])


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    q_stand = data.qpos[7:].copy()

    adr_quat = sensor_index(model, "imu_quat")
    adr_gyro = sensor_index(model, "imu_gyro")
    adr_acc = sensor_index(model, "imu_acc")
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")

    print(__doc__)

    last_print = -1.0
    r_was_down = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            r_down = key_down(VK["NUM0"])
            if r_down and not r_was_down:
                mujoco.mj_resetDataKeyframe(model, data, 0)
                print("已重置回站立")
            r_was_down = r_down

            data.xfrc_applied[:] = 0.0
            force = read_push()
            data.xfrc_applied[torso_id, :3] = force

            q = data.qpos[7:]
            dq = data.qvel[6:]
            tau = KP * (q_stand - q) - KD * dq
            for i in range(model.nu):
                lo, hi = model.actuator_ctrlrange[i]
                tau[i] = float(np.clip(tau[i], lo, hi))
            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time - last_print >= 0.35:
                last_print = data.time
                qw, qx, qy, qz = data.sensordata[adr_quat : adr_quat + 4]
                gx, gy, gz = data.sensordata[adr_gyro : adr_gyro + 3]
                ax, ay, az = data.sensordata[adr_acc : adr_acc + 3]
                roll, pitch, yaw = quat_to_rpy_deg(float(qw), float(qx), float(qy), float(qz))
                feet = feet_on_ground(model, data)
                feet_txt = " ".join(
                    f"{FOOT_CN[n]}:{'踩地' if feet[n] else '抬起'}" for n in FOOT_NAMES
                )
                push_txt = (
                    f"外力 ({force[0]:+.0f},{force[1]:+.0f},{force[2]:+.0f})"
                    if np.any(force)
                    else "外力 (无，按小键盘 8/4/6/2/9/3)"
                )
                print("=" * 52)
                print(f"时间 {data.time:5.1f}s   {push_txt}")
                print(f"高度 {data.qpos[2]:.3f} m   （站着大约 0.18）")
                print(f"IMU 左右{roll:+6.1f}°  前后{pitch:+6.1f}°  朝向{yaw:+6.1f}°")
                print(f"四脚 {feet_txt}")
                print(f"陀螺仪 {gx:+.2f} {gy:+.2f} {gz:+.2f}")
                print(f"加速度 {ax:+.2f} {ay:+.2f} {az:+.2f}   （站住时 z 接近重力）")

            sleep = model.opt.timestep - (time.time() - step_start)
            if sleep > 0:
                time.sleep(sleep)


if __name__ == "__main__":
    main()
