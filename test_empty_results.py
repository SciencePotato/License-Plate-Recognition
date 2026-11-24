import ast
from pathlib import Path
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


class EmptyResultTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path(__file__).with_name("util.py").read_text())
        tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        self.module = {"np": np, "cv2": MagicMock(), "reader": MagicMock(),
                       "licensePlateModel": MagicMock(), "licenseMap": {}}
        exec(compile(tree, "util.py", "exec"), self.module)
        self.cv = self.module["cv2"]

    def test_no_plate_detections_is_not_an_error(self):
        detection = MagicMock()
        detection.boxes.data.tolist.return_value = []
        for detections in [[detection], []]:
            with self.subTest(detections=len(detections)), patch("builtins.print") as output:
                self.module["licensePlateModel"].predict.return_value = detections
                self.assertEqual(self.module["renderImageEasyOCR"](".", "unused.png", "."), (0, 0, 0))
                self.assertNotIn(("Error, Something went Wrong",), [call.args for call in output.call_args_list])
        self.assertEqual(self.cv.imshow.call_count, 2)

    def test_empty_ocr_does_not_pass_none_to_opencv(self):
        self.module["reader"].readtext.return_value = []
        self.assertEqual(self.module["readLicenseImage"](object()), (None, 0))
        detection = MagicMock()
        detection.boxes.data.tolist.return_value = [[0, 0, 10, 10, 0.9, 0]]
        self.module["licensePlateModel"].predict.return_value = [detection]
        self.cv.resize.return_value = np.ones((10, 10, 3), dtype=np.uint8)
        self.cv.threshold.return_value = (0, object())
        self.module["warpImage"] = MagicMock(return_value=object())
        with patch("builtins.print"):
            self.assertEqual(self.module["renderImageEasyOCR"](".", "unused.png", "."), (1, 0, 0))
        for call in self.cv.putText.call_args_list:
            self.assertIsInstance(call.args[1], str)


if __name__ == "__main__":
    unittest.main()
