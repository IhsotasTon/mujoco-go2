"""Launch the better G1 walking demo (RoboCubPilot/g1_deploy_mujoco).

Run from project root:
  python run_g1_rl_walk.py

键盘推人（按住才有力）：
  I / K  往前推 / 往后推
  J / L  往左推 / 往右推
  U / O  往上抬 / 往下压
  R      倒地后重置回站立
鼠标只转视角，推不动。
"""

from __future__ import annotations

import os
import runpy
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
DEPLOY = os.path.join(ROOT, "g1_deploy_mujoco")


def main() -> None:
    if not os.path.isdir(DEPLOY):
        raise SystemExit(f"Missing folder: {DEPLOY}")
    os.chdir(DEPLOY)
    sys.argv = [
        "deploy_mujoco.py",
        "--policy",
        os.path.join(DEPLOY, "checkpoint", "policy.pt"),
    ]
    runpy.run_path(os.path.join(DEPLOY, "deploy_mujoco.py"), run_name="__main__")


if __name__ == "__main__":
    main()
