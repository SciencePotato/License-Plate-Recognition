import ast
from pathlib import Path
import unittest
from unittest.mock import MagicMock

import numpy as np


def load_functions():
    tree = ast.parse(Path(__file__).with_name("util.py").read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {"np": np, "cv2": MagicMock(), "reader": MagicMock(),
                 "licensePlateModel": MagicMock(), "licenseMap": {}}
    exec(compile(tree, "util.py", "exec"), namespace)
    return namespace


class LicenseBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.module = load_functions()
        self.cv = self.module["cv2"]

    def test_both_deskew_angle_branches_return_rotated_image(self):
        image = np.ones((10, 20), dtype=np.uint8)
        for angle, corrected in [(-60, -30), (-30, 30)]:
            with self.subTest(angle=angle):
                self.cv.minAreaRect.return_value = (None, None, angle)
                result = self.module["deskew"](image)
                self.assertIs(result, self.cv.warpAffine.return_value)
                self.cv.getRotationMatrix2D.assert_called_with((10, 5), corrected, 1.0)

if __name__ == "__main__":
    unittest.main()
