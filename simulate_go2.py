"""Load Unitree Go2 and run a real-time simulation."""

import time
from pathlib import Path

import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "unitree_go2" / "scene.xml"
XML = ROOT / "unitree_go2" / "go2.xml"

path = SCENE if SCENE.exists() else XML
if not path.exists():
    raise FileNotFoundError(
        f"Go2 model not found at {path}. "
        "Make sure the unitree_go2 folder is in this project."
    )

model = mujoco.MjModel.from_xml_path(str(path))
data = mujoco.MjData(model)

# Prefer standing keyframe if the model provides one.
if model.nkey > 0:
    mujoco.mj_resetDataKeyframe(model, data, 0)

print(f"Loaded: {path}")
print("Simulation running. Close the window to stop.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        step_start = time.time()
        mujoco.mj_step(model, data)
        viewer.sync()
        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
