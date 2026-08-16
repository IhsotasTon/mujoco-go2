"""打开 Unitree G1 人形机器人（只看，不控制）."""

from pathlib import Path

import mujoco
import mujoco.viewer

SCENE = Path(__file__).resolve().parent / "unitree_g1" / "scene.xml"

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

print(f"已加载 G1：关节自由度数 nq={model.nq}，电机数 nu={model.nu}")
print("关闭窗口退出。")
mujoco.viewer.launch(model, data)
