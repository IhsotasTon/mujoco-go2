"""键盘读取：小键盘 + 顶排数字，不占用 MuJoCo 的字母键 / Space。"""

from __future__ import annotations

import ctypes

user32 = ctypes.windll.user32

# 小键盘、顶排数字、方向键（NumLock 关掉时小键盘会变成方向键）
VK = {
    "NUM0": 0x60,
    "NUM2": 0x62,
    "NUM4": 0x64,
    "NUM5": 0x65,
    "NUM6": 0x66,
    "NUM7": 0x67,
    "NUM8": 0x68,
    "HOME": 0x24,  # NumLock 关时的 7
    "D7": 0x37,    # 顶排 7
    "NUM9": 0x69,
    "PRIOR": 0x21,  # NumLock 关时的 9
    "D9": 0x39,     # 顶排 9
    "INSERT": 0x2D,  # NumLock 关时的 0
    "END": 0x23,     # NumLock 关时的 1
    "DOWN": 0x28,    # 2
    "LEFT": 0x25,    # 4
    "CLEAR": 0x0C,   # 5
    "RIGHT": 0x27,   # 6
    "UP": 0x26,      # 8
    "D8": 0x38,      # 顶排 8
    "D2": 0x32,
    "D4": 0x34,
    "D5": 0x35,
    "D6": 0x36,
    "D0": 0x30,
}


def any_down(*names: str) -> bool:
    for name in names:
        if user32.GetAsyncKeyState(VK[name]) & 0x8000:
            return True
    return False
