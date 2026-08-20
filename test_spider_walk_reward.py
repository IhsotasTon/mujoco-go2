"""奖励必须让稳走比前冲摔、站桩、甩头绕场更赚，否则 PPO 会学扑倒或南绕。"""

from __future__ import annotations

import unittest

import numpy as np

from eval_spider_walk import GAIT_GATES, gait_passed, make_eval_env
from spider_walk_env import (
    RF_CUTOFF,
    STALL_STEPS,
    WALK_ONLY_OBS_XY,
    SpiderWalkEnv,
    is_idle_motion,
    yaw_from_quat,
)


def _reward_at(
    env: SpiderWalkEnv,
    vx: float,
    wy: float,
    *,
    vy: float = 0.0,
    wz: float = 0.0,
    y: float = 0.0,
    yaw: float = 0.0,
    up_z: float | None = None,
    prev_vx: float | None = None,
) -> float:
    env.data.qvel[:] = 0.0
    env.data.qvel[0] = vx
    env.data.qvel[1] = vy
    env.data.qvel[4] = wy
    env.data.qvel[5] = wz
    env.data.qpos[0] = 0.0
    env.data.qpos[1] = y
    if up_z is not None:
        env.data.qpos[2] = up_z
    else:
        env.data.qpos[2] = 0.18
    env.data.qpos[3:7] = (np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0))
    env.prev_vx = float(vx if prev_vx is None else prev_vx)
    action = np.zeros(env.model.nu, dtype=np.float32)
    r, _info = env._reward(action, hit=False)
    return float(r)


class WalkRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.env = SpiderWalkEnv(walk_only=True)
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

    def test_slow_crawl_is_not_worth_farming(self) -> None:
        crawl = _reward_at(self.env, vx=0.03, wy=0.0)
        self.assertLess(crawl, 0.0)
        cruise = _reward_at(self.env, vx=0.18, wy=0.0)
        self.assertGreater(cruise, crawl)

    def test_fallen_worse_than_standing_walk(self) -> None:
        walk = _reward_at(self.env, vx=0.20, wy=0.15)
        fallen = _reward_at(self.env, vx=0.20, wy=0.15, up_z=0.05)
        self.assertGreater(walk, fallen)

    def test_upright_walk_beats_tilted_sprint(self) -> None:
        # 若 upright 项太弱，策略会低头换速度，几十步后 up<0.4 结束。
        upright = _reward_at(self.env, vx=0.22, wy=0.15)
        pitch = 0.75
        self.env.data.qpos[3:7] = (np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0)
        self.env.data.qvel[:] = 0.0
        self.env.data.qvel[0] = 0.35
        self.env.data.qvel[4] = 0.15
        self.env.data.qpos[1] = 0.0
        self.env.data.qpos[2] = 0.18
        self.env.prev_vx = 0.35
        action = np.zeros(self.env.model.nu, dtype=np.float32)
        tilted, _ = self.env._reward(action, hit=False)
        self.assertGreater(upright, float(tilted))

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

    def test_spin_or_sidestep_without_vx_is_idle(self) -> None:
        # 步态优先：原地转 / 蟹行不能当成「在动」而躲过停步掐断。
        self.assertTrue(is_idle_motion(0.0, 0.0, 0.0))
        self.assertTrue(is_idle_motion(0.0, 0.25, 0.0))
        self.assertTrue(is_idle_motion(0.0, 0.0, 0.40))
        self.assertFalse(is_idle_motion(0.15, 0.0, 0.0))

    def test_spin_in_place_episode_cannot_fill_horizon(self) -> None:
        action = np.zeros(self.env.model.nu, dtype=np.float32)
        ended = False
        for t in range(STALL_STEPS + 10):
            self.env.data.qvel[0] = 0.02
            self.env.data.qvel[5] = 1.2
            _obs, _r, term, trunc, _info = self.env.step(action)
            if term or trunc:
                ended = True
                self.assertTrue(term)
                self.assertLessEqual(t + 1, STALL_STEPS + 5)
                break
        self.assertTrue(ended)

    def test_crab_walk_pays_less_than_straight(self) -> None:
        straight = _reward_at(self.env, vx=0.22, wy=0.0, vy=0.0)
        crab = _reward_at(self.env, vx=0.22, wy=0.0, vy=0.25)
        self.assertGreater(straight, crab)

    def test_yaw_heading_cost_beats_facing_south(self) -> None:
        on_heading = _reward_at(self.env, vx=0.20, wy=0.0, yaw=0.0)
        south = _reward_at(self.env, vx=0.20, wy=0.0, yaw=-1.2)
        self.assertGreater(on_heading, south)

    def test_y_drift_costs_more_than_on_axis(self) -> None:
        on_axis = _reward_at(self.env, vx=0.20, wy=0.0, y=0.0)
        bypass = _reward_at(self.env, vx=0.20, wy=0.0, y=-1.5)
        self.assertGreater(on_axis, bypass)

    def test_south_bypass_loses_to_straight_heading(self) -> None:
        # 播放里 x:0→2.74、y:0→-1.5、wz 冲到 -4.7 不能看起来像成功走路。
        straight = _reward_at(self.env, vx=0.20, wy=0.10, y=0.0, yaw=0.0, wz=0.05)
        bypass = _reward_at(
            self.env, vx=0.20, wy=0.10, y=-1.5, yaw=-0.5, wz=-4.7
        )
        self.assertGreater(straight, bypass)

    def test_yaw_spike_is_worse_than_calm_yaw(self) -> None:
        calm = _reward_at(self.env, vx=0.20, wy=0.0, wz=0.10)
        spike = _reward_at(self.env, vx=0.20, wy=0.0, wz=4.7)
        self.assertGreater(calm, spike)

    def test_spin_to_move_loses_to_forward_walk(self) -> None:
        forward = _reward_at(self.env, vx=0.20, wy=0.10, wz=0.05)
        spin = _reward_at(self.env, vx=0.06, wy=0.10, wz=1.4)
        self.assertGreater(forward, spin)

    def test_steady_vx_beats_burst_from_near_stop(self) -> None:
        # 存活分若在 0.08 处跳变，策略会 0.03 再 0.72 一窜一停。
        steady = _reward_at(self.env, vx=0.22, wy=0.0, prev_vx=0.22)
        burst = _reward_at(self.env, vx=0.72, wy=0.0, prev_vx=0.03)
        self.assertGreater(steady, burst)

    def test_forward_walk_beats_orbiting(self) -> None:
        passing = _reward_at(self.env, vx=0.22, wy=0.0, vy=0.12, wz=0.10)
        orbit = _reward_at(self.env, vx=0.02, wy=0.0, vy=0.22, wz=0.70)
        self.assertGreater(passing, orbit)

    def test_blocked_pillar_does_not_pay_for_spin_or_crab(self) -> None:
        # 柱前 |vy|/|wz| 加分会教「离开跑道」。步态优先时即使测距被挡也不该发这笔钱。
        import mujoco

        env = SpiderWalkEnv(walk_only=False)
        env.reset(seed=0)
        env.data.mocap_pos[0] = (0.45, 0.0, 0.11)
        mujoco.mj_forward(env.model, env.data)
        rf = env._rangefinders()
        self.assertLess(float(rf[2]), 0.50)
        stop = _reward_at(env, vx=0.04, wy=0.10)
        sidestep = _reward_at(env, vx=0.04, wy=0.10, vy=0.25)
        spin = _reward_at(env, vx=0.04, wy=0.10, wz=0.80)
        env.close()
        self.assertGreater(stop, sidestep)
        self.assertGreater(stop, spin)


class WalkOnlyEnvTests(unittest.TestCase):
    def test_walk_only_parks_pillars_beyond_rangefinder(self) -> None:
        env = SpiderWalkEnv(walk_only=True)
        env.reset(seed=1)
        for i, (x, y) in enumerate(WALK_ONLY_OBS_XY):
            self.assertGreater(float(env.data.mocap_pos[i, 0]), RF_CUTOFF + 10.0)
            self.assertAlmostEqual(float(env.data.mocap_pos[i, 0]), x, places=5)
            self.assertAlmostEqual(float(env.data.mocap_pos[i, 1]), y, places=5)
        rf = env._rangefinders()
        self.assertTrue(np.all(rf > 0.95))
        env.close()

    def test_train_make_env_is_walk_only(self) -> None:
        from train_spider_walk import make_env

        env = make_env()
        self.assertTrue(env.walk_only)
        env.reset(seed=2)
        self.assertGreater(float(env.data.mocap_pos[0, 0]), 40.0)
        env.close()

    def test_eval_env_is_walk_only(self) -> None:
        env = make_eval_env()
        self.assertTrue(env.walk_only)
        env.reset(seed=3)
        self.assertGreater(float(env.data.mocap_pos[0, 0]), 40.0)
        env.close()

    def test_play_layout_keeps_course_pillars(self) -> None:
        from spider_walk_env import PLAY_OBS_XY

        env = SpiderWalkEnv(recover=True, walk_only=False, play_layout=True)
        self.assertFalse(env.walk_only)
        self.assertTrue(env.play_layout)
        env.reset(seed=0)
        for i, (x, y) in enumerate(PLAY_OBS_XY):
            self.assertAlmostEqual(float(env.data.mocap_pos[i, 0]), x, places=5)
            self.assertAlmostEqual(float(env.data.mocap_pos[i, 1]), y, places=5)
        env.close()

    def test_gait_gates_reject_crawl_and_south_bypass(self) -> None:
        self.assertGreater(GAIT_GATES["mean_vx"], 0.12)
        self.assertGreater(GAIT_GATES["frac_upright"], 0.80)
        self.assertLess(GAIT_GATES["mean_abs_wz"], 1.0)
        self.assertLess(GAIT_GATES["mean_abs_vy"], 0.20)
        self.assertLess(GAIT_GATES["mean_abs_y"], 0.50)
        self.assertGreater(GAIT_GATES["steps"], 400.0)
        crawl = dict(
            frac_upright=1.0,
            mean_vx=0.122,
            mean_abs_vy=0.20,
            mean_abs_wz=0.80,
            mean_abs_y=0.80,
            mean_abs_wy=0.50,
            steps=596.0,
        )
        bypass = dict(
            frac_upright=0.96,
            mean_vx=0.22,
            mean_abs_vy=0.18,
            mean_abs_wz=1.10,
            mean_abs_y=0.90,
            mean_abs_wy=0.40,
            steps=500.0,
        )
        good = dict(
            frac_upright=0.95,
            mean_vx=0.20,
            mean_abs_vy=0.05,
            mean_abs_wz=0.12,
            mean_abs_y=0.10,
            mean_abs_wy=0.40,
            steps=580.0,
        )
        self.assertFalse(gait_passed(crawl))
        self.assertFalse(gait_passed(bypass))
        self.assertTrue(gait_passed(good))

    def test_yaw_from_quat_zero_and_south(self) -> None:
        identity = np.array([1.0, 0.0, 0.0, 0.0])
        self.assertAlmostEqual(yaw_from_quat(identity), 0.0, places=5)
        south = np.array([np.cos(-np.pi / 4), 0.0, 0.0, np.sin(-np.pi / 4)])
        self.assertAlmostEqual(yaw_from_quat(south), -np.pi / 2, places=5)

    def test_eval_import_path_does_not_need_weights(self) -> None:
        import eval_spider_walk
        import train_spider_walk

        self.assertTrue(callable(eval_spider_walk.evaluate))
        self.assertTrue(callable(train_spider_walk.make_env))
        self.assertIn("mean_abs_wz", eval_spider_walk.GAIT_STAT_KEYS)


if __name__ == "__main__":
    unittest.main()
