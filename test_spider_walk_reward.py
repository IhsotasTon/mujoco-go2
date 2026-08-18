"""奖励必须让稳走比前冲摔更赚，否则 PPO 会学扑倒。"""

from __future__ import annotations

import unittest

import numpy as np

from spider_walk_env import SpiderWalkEnv


def _reward_at(env: SpiderWalkEnv, vx: float, wy: float, *, up_z: float | None = None) -> float:
    env.data.qvel[:] = 0.0
    env.data.qvel[0] = vx
    env.data.qvel[4] = wy
    if up_z is not None:
        env.data.qpos[2] = up_z
    action = np.zeros(env.model.nu, dtype=np.float32)
    r, _info = env._reward(action, hit=False)
    return float(r)


class WalkRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SpiderWalkEnv()
        self.env.reset(seed=0)

    def tearDown(self) -> None:
        self.env.close()

    def test_lunge_pays_less_than_steady_walk(self) -> None:
        # 若恢复 3.8*vx 且 wy 惩罚偏弱，前冲会重新赢过稳走。
        steady = _reward_at(self.env, vx=0.25, wy=0.20)
        lunge = _reward_at(self.env, vx=0.90, wy=1.90)
        self.assertGreater(steady, lunge)

    def test_speed_above_cap_does_not_pay_more(self) -> None:
        # 超过目标速度不应再加分，否则会冲到摔。
        at_cap = _reward_at(self.env, vx=0.35, wy=0.0)
        over_cap = _reward_at(self.env, vx=0.90, wy=0.0)
        self.assertLessEqual(over_cap, at_cap + 1e-6)

    def test_standing_still_worse_than_walking(self) -> None:
        walk = _reward_at(self.env, vx=0.20, wy=0.15)
        idle = _reward_at(self.env, vx=0.0, wy=0.0)
        self.assertGreater(walk, idle)

    def test_idle_reward_is_negative(self) -> None:
        # 站桩拿直立分会撑满 600 步、速度接近 0。
        idle = _reward_at(self.env, vx=0.0, wy=0.0)
        self.assertLess(idle, 0.0)

    def test_fallen_worse_than_standing_walk(self) -> None:
        walk = _reward_at(self.env, vx=0.20, wy=0.15)
        fallen = _reward_at(self.env, vx=0.20, wy=0.15, up_z=0.05)
        self.assertGreater(walk, fallen)

    def test_upright_walk_beats_tilted_sprint(self) -> None:
        # 若 upright 项太弱，策略会低头换速度，几十步后 up<0.4 结束。
        upright = _reward_at(self.env, vx=0.22, wy=0.15)
        pitch = 0.75
        self.env.data.qpos[3:7] = (np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0)
        tilted = _reward_at(self.env, vx=0.35, wy=0.15)
        self.assertGreater(upright, tilted)

    def test_standing_still_episode_cannot_fill_horizon(self) -> None:
        # 零动作站桩必须被提前掐掉，否则会撑满 600 步刷直立。
        action = np.zeros(self.env.model.nu, dtype=np.float32)
        ended = False
        for t in range(80):
            _obs, _r, term, trunc, _info = self.env.step(action)
            if term or trunc:
                ended = True
                self.assertLess(t + 1, 80)
                self.assertTrue(term)
                break
        self.assertTrue(ended)


if __name__ == "__main__":
    unittest.main()
