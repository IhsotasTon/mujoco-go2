"""Open Unitree Go2 in the MuJoCo viewer."""

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

print(f"Loaded: {path}")
print(f"nq={model.nq}, nv={model.nv}, nu={model.nu}")
print("Close the viewer window to exit.")

mujoco.viewer.launch(model, data)
