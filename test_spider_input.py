"""键盘模块必须在 macOS 上也能 import，否则播放脚本起不来。"""

from __future__ import annotations

import unittest


class SpiderInputTests(unittest.TestCase):
    def test_import_and_any_down_on_this_os(self) -> None:
        import spider_input

        self.assertFalse(spider_input.any_down("NUM0"))


if __name__ == "__main__":
    unittest.main()
