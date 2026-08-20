# mujoco-go2

MuJoCo 学习项目：Unitree Go2 / G1，以及自制四足机械蜘蛛。

## 环境

- Python 3.12
- `pip install mujoco gymnasium numpy pyyaml`

强化学习训练另需：`torch`、`stable-baselines3`

## 蜘蛛（自制）

```powershell
python view_spider.py
python stand_spider.py
python walk_spider.py
```

`walk_spider.py` 先点 MuJoCo 窗口（不要用 WASD / Space，那些是 MuJoCo 自己的快捷键）：

| 键 | 作用 |
|----|------|
| 小键盘 8 | 前进 |
| 2 / 4 / 6 | 后退 / 左转 / 右转 |
| 5 | 停 |
| 9 | 跳 |
| 7 | 前空翻（手写冲量，落地不稳） |
| 0 | 重置 |

云上训练先只做**稳定前进步态**（柱子停远处，不进奖励；不要再用 `train_spider.py` 混训空翻，也不要一上来避障）：

```bash
export MUJOCO_GL=egl
python train_spider_walk.py --timesteps 200000 --n-envs 8
python eval_spider_walk.py
```

200k 后看 `eval_spider_walk.py` 打印的步态数字，过闸门再考虑加长：

| 指标 | 闸门 |
|------|------|
| `frac_upright` | > 0.85 |
| `mean_vx` | > 0.15（明显高于爬行） |
| `mean_abs_vy` | < 0.12 |
| `mean_abs_wz` | < 0.45 |
| `mean_abs_y` | < 0.35 |
| `mean_abs_wy` | < 1.2 |
| `steps` | > 450 |

直立率低、原地转、横着走出场：停下来改奖励，别空转 150 万。本地播放仍可放柱子看以后绕障，倒下后空翻只在播放：`python play_spider_walk.py`。

## Unitree Go2

```powershell
python view_go2.py
python stand_go2.py
python walk_go2.py
python sensors_go2.py
```

## Unitree G1

```powershell
python view_g1.py
python stand_g1.py
python run_g1_rl_walk.py
```

G1 稳定走路策略来自 [g1_deploy_mujoco](https://github.com/RoboCubPilot/g1_deploy_mujoco)。
