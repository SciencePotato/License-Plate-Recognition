import ast
from pathlib import Path
import unittest

import numpy as np


class CornerOrderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = Path(__file__).with_name("util.py")
        tree = ast.parse(source.read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name == "order_points"]
        cls.namespace = {"np": np}
        exec(compile(tree, str(source), "exec"), cls.namespace)

    def test_translating_plate_does_not_change_corner_correspondence(self):
        points = np.float32([[0, 0], [0, 20], [100, 0], [100, 20]])
        order = self.namespace["order_points"]
        expected = np.float32([[0, 0], [0, 20], [100, 20], [100, 0]])
        for offset in ([0, 0], [1000, 0], [0, 1000], [-1000, 200]):
            with self.subTest(offset=offset):
                actual = order(points + offset) - offset
                np.testing.assert_array_equal(actual, expected)

    def test_shuffled_rotated_plate_orders_without_editing_input(self):
        points = np.float32([[30, 5], [5, 40], [45, 50], [70, 15]])
        original = points.copy()
        order = self.namespace["order_points"]
        expected = order(points)
        np.testing.assert_array_equal(order(points[[2, 0, 3, 1]]), expected)
        np.testing.assert_array_equal(points, original)


if __name__ == "__main__":
    unittest.main()
