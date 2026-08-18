"""倒下后用后空翻回正，不结束回合。"""

from __future__ import annotations

import unittest

import numpy as np

from spider_walk_env import SpiderWalkEnv, projected_gravity


class BackflipRecoverTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SpiderWalkEnv(recover=True)
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_trigger_backflip_adds_negative_pitch_rate(self) -> None:
        self.env.data.qvel[:] = 0.0
        self.env.trigger_backflip()
        self.assertLess(float(self.env.data.qvel[4]), -1.0)

    def test_belly_up_does_not_terminate_when_recovering(self) -> None:
        self.env.data.qpos[2] = 0.12
        self.env.data.qpos[3:7] = (0.0, 1.0, 0.0, 0.0)
        self.env.data.qvel[:] = 0.0
        action = np.zeros(self.env.model.nu, dtype=np.float32)
        _obs, _r, term, _trunc, info = self.env.step(action)
        self.assertFalse(term)
        self.assertLess(float(info["up"]), 0.4)

    def test_backflip_from_belly_increases_upright(self) -> None:
        self.env.data.qpos[2] = 0.14
        self.env.data.qpos[3:7] = (0.0, 1.0, 0.0, 0.0)
        self.env.data.qvel[:] = 0.0
        action = np.zeros(self.env.model.nu, dtype=np.float32)
        ups = []
        for _ in range(80):
            _obs, _r, term, trunc, info = self.env.step(action)
            ups.append(float(info["up"]))
            if term or trunc:
                break
        self.assertGreater(max(ups), 0.55)
        self.assertGreater(float(projected_gravity(self.env.data.qpos[3:7])[2]), -0.2)

    def test_recover_does_not_truncate_at_horizon(self) -> None:
        self.env.steps = 599
        action = np.zeros(self.env.model.nu, dtype=np.float32)
        _obs, _r, _term, trunc, _info = self.env.step(action)
        self.assertFalse(trunc)

    def test_busy_while_backflip_blocks_walk(self) -> None:
        self.env.trigger_backflip()
        self.assertTrue(self.env.busy)


if __name__ == "__main__":
    unittest.main()
