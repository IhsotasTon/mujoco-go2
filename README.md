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

云上训练爬行 + 空翻 + 倒地翻面：

```bash
export MUJOCO_GL=egl
python train_spider.py --timesteps 1500000
```

本地播放：`python play_spider_rl.py`（需要 `spider_quad/spider_ppo.zip`）

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
