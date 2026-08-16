"""读 Go2 身上的“感觉”数据（大白话版）.

你会看到：
  - 身体有多高
  - 身体歪没歪（左右 / 前后）
  - 头朝哪边
  - 四只脚谁踩地、谁抬起
  - 背上面传感器测到的角速度、加速度

操作（按住键盘推狗，比鼠标好用）：
  I / K     往前推 / 往后推
  J / L     往左推 / 往右推
  U / O     往上抬 / 往下压
  R         重置回站立姿势
  关闭窗口退出

说明：鼠标在 MuJoCo 里主要是转视角，不是点一下就能推。
"""

import ctypes
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "unitree_go2" / "scene_with_sensors.xml"

KP = 90.0
KD = 2.5
PUSH = 80.0  # 推力大小，太小看不出，太大会直接摔飞

# 脚底碰撞体名字（模型里写好的）
FOOT_NAMES = ("FL", "FR", "RL", "RR")
FOOT_CN = {"FL": "左前", "FR": "右前", "RL": "左后", "RR": "右后"}

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
    """返回作用在狗身体上的力 [fx, fy, fz]."""
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


def quat_to_rpy_deg(w: float, x: float, y: float, z: float) -> tuple[float, float, float]:
    """把旋转数字换成三个好懂的角度（单位：度）."""
    # 左右歪
    sinr = 2 * (w * x + y * z)
    cosr = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr, cosr))

    # 前后仰
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    # 水平朝向（转圈）
    siny = 2 * (w * z + x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny, cosy))
    return float(roll), float(pitch), float(yaw)


def feet_on_ground(model: mujoco.MjModel, data: mujoco.MjData) -> dict[str, bool]:
    """哪只脚正在碰到东西（通常是地面）."""
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
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    q_stand = data.qpos[7:].copy()

    adr_quat = sensor_index(model, "imu_quat")
    adr_gyro = sensor_index(model, "imu_gyro")
    adr_acc = sensor_index(model, "imu_acc")

    print(__doc__)
    print("开始了。每隔一会儿打印一行状态。\n")

    # 机身刚体编号（推力打在狗肚子/背上）
    base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base")

    last_print = -1.0
    r_was_down = False
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # R：整只狗重置回站立
            r_down = key_down(VK["R"])
            if r_down and not r_was_down:
                mujoco.mj_resetDataKeyframe(model, data, 0)
                print("\n>>> 已重置回站立\n")
            r_was_down = r_down

            # 清掉上一帧的外力，再按当前按键施加
            data.xfrc_applied[:] = 0.0
            force = read_push()
            data.xfrc_applied[base_id, :3] = force

            # 还是用简单 PD 站着，方便你推它看传感器变化
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

                # 1) 从机身姿态算倾斜角
                roll, pitch, yaw = quat_to_rpy_deg(*data.qpos[3:7])
                height = float(data.qpos[2])

                # 2) 从模型里声明的传感器读数
                qw, qx, qy, qz = data.sensordata[adr_quat : adr_quat + 4]
                gx, gy, gz = data.sensordata[adr_gyro : adr_gyro + 3]
                ax, ay, az = data.sensordata[adr_acc : adr_acc + 3]
                # 用传感器姿态再算一遍，应该和上面接近
                s_roll, s_pitch, s_yaw = quat_to_rpy_deg(qw, qx, qy, qz)

                feet = feet_on_ground(model, data)
                feet_txt = " ".join(
                    f"{FOOT_CN[n]}:{'踩地' if feet[n] else '抬起'}" for n in FOOT_NAMES
                )

                push_txt = (
                    f"外力 ({force[0]:+.0f}, {force[1]:+.0f}, {force[2]:+.0f})"
                    if np.any(force)
                    else "外力 (无，按 I/J/K/L/U/O 推它)"
                )
                print("=" * 56)
                print(f"时间 {data.time:5.1f} 秒    {push_txt}")
                print(f"身体高度      {height:.3f} 米   （站着大概 0.25~0.30）")
                print(
                    f"左右歪        {roll:+6.1f}°   "
                    f"（0=不歪，正负=往一侧倒）"
                )
                print(
                    f"前后仰        {pitch:+6.1f}°   "
                    f"（0=不仰，正负=前倾/后仰）"
                )
                print(f"水平朝向      {yaw:+6.1f}°   （转圈会变）")
                print(f"四只脚        {feet_txt}")
                print(
                    f"背传感器-转快  "
                    f"x={gx:+.2f} y={gy:+.2f} z={gz:+.2f}  （越大转得越猛）"
                )
                print(
                    f"背传感器-加速度 "
                    f"x={ax:+.2f} y={ay:+.2f} z={az:+.2f}  "
                    f"（站稳时 z 大约接近重力）"
                )
                print(
                    f"传感器算出的倾斜 "
                    f"左右{s_roll:+.1f}° 前后{s_pitch:+.1f}° 朝向{s_yaw:+.1f}°"
                )

            elapsed = time.time() - step_start
            sleep_time = model.opt.timestep - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    main()
