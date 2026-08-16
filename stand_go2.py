"""Hold / gently move Unitree Go2 with PD joint control."""

import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "unitree_go2" / "scene.xml"

KP = 80.0
KD = 2.5

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, data, 0)

# Standing joint targets from keyframe (12 joints)
q_stand = data.qpos[7:].copy()

# Indices: each leg is [hip, thigh, calf]
# Squat by bending thighs/calves a bit
thigh_ids = [1, 4, 7, 10]
calf_ids = [2, 5, 8, 11]

print("PD control running — dog should slowly squat up and down.")
print("Close the window to stop.")

with mujoco.viewer.launch_passive(model, data) as viewer:
    # Make sure the viewer is not paused
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False

    while viewer.is_running():
        step_start = time.time()
        t = data.time

        # Slow squat: period ~ 2 seconds
        squat = 0.25 * (0.5 - 0.5 * math.cos(2 * math.pi * t / 2.0))
        q_des = q_stand.copy()
        q_des[thigh_ids] += squat
        q_des[calf_ids] -= 2.0 * squat

        q = data.qpos[7:]
        dq = data.qvel[6:]
        tau = KP * (q_des - q) - KD * dq

        for i in range(model.nu):
            lo, hi = model.actuator_ctrlrange[i]
            tau[i] = float(np.clip(tau[i], lo, hi))

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)
        viewer.sync()

        # Print once per second so you can see time advancing
        if int(t) != int(t - model.opt.timestep) and int(t) % 1 == 0:
            print(f"t={t:.1f}s  base_z={data.qpos[2]:.3f}")

        elapsed = time.time() - step_start
        sleep_time = model.opt.timestep - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)
