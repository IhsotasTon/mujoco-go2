"""打开自制四足蜘蛛（只看，不控制）。"""

from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent
XML = ROOT / "spider_quad" / "spider_quad.xml"

model = mujoco.MjModel.from_xml_path(str(XML))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)
mujoco.mj_forward(model, data)

print(f"Loaded: {XML}")
print(f"nq={model.nq}  关节电机={model.nu}  （4 条腿 × 3 关节）")
print("鼠标转视角，关闭窗口退出。")

mujoco.viewer.launch(model, data)
