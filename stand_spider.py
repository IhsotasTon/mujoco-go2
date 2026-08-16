"""自制四足蜘蛛：PD 撑住站立姿势。关闭窗口退出。"""

from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
XML = ROOT / "spider_quad" / "spider_quad.xml"

KP = 18.0
KD = 0.6


def main() -> None:
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    target = data.qpos[7:].copy()

    print("蜘蛛站立中。没有控制时它会瘫下去；这里用 PD 撑住。")
    print("关闭窗口退出。")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            q = data.qpos[7:]
            dq = data.qvel[6:]
            data.ctrl[:] = (target - q) * KP + (0.0 - dq) * KD
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()
