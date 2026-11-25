import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

import numpy as np


class ImageErrorTests(unittest.TestCase):
    def renderer(self):
        tree = ast.parse(Path(__file__).with_name("util.py").read_text())
        fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "renderImageEasyOCR")
        cv = Mock(THRESH_BINARY_INV=8, THRESH_OTSU=16)
        cv.imread.return_value = cv.resize.return_value = np.zeros((20, 30, 3), dtype=np.uint8)
        cv.threshold.return_value = (64, object())
        boxes = types.SimpleNamespace(data=types.SimpleNamespace(tolist=lambda: [[1, 2, 8, 9, .9, 0]]))
        ns = dict(np=np, cv2=cv, licensePlateModel=Mock(predict=Mock(return_value=[types.SimpleNamespace(boxes=boxes)])),
                  warpImage=Mock(return_value=object()), readLicenseImage=Mock(return_value=("SYNTHETIC", .9)))
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "util.py", "exec"), ns)
        return ns, cv

    def test_failed_fallback_propagates_instead_of_returning_counts(self):
        ns, cv = self.renderer()
        cv.cvtColor.side_effect = OSError("synthetic conversion failed")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(OSError, "conversion failed"):
            ns["renderImageEasyOCR"]("synthetic", "input.png", "unused")
        cv.imshow.assert_not_called()

    def test_interrupt_never_reenters_fallback_or_reports_success(self):
        ns, cv = self.renderer()
        cv.cvtColor.side_effect = KeyboardInterrupt()
        with redirect_stdout(io.StringIO()), self.assertRaises(KeyboardInterrupt):
            ns["renderImageEasyOCR"]("synthetic", "input.png", "unused")
        self.assertEqual(cv.cvtColor.call_count, 1)
        ns["readLicenseImage"].assert_not_called()

    def test_successful_recognition_keeps_reporting_counts(self):
        ns, cv = self.renderer()
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ns["renderImageEasyOCR"]("synthetic", "input.png", "unused"), (0, 0, 1))
        cv.imshow.assert_called_once()

    def test_preprocessing_failure_does_not_silently_run_raw_ocr(self):
        ns, cv = self.renderer()
        cv.GaussianBlur.side_effect = OSError("synthetic blur failure")
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(OSError, "blur failure"):
            ns["renderImageEasyOCR"]("synthetic", "input.png", "unused")
        ns["readLicenseImage"].assert_not_called()

    def test_absent_quadrilateral_still_uses_original_crop(self):
        ns, _ = self.renderer()
        ns["warpImage"].return_value = None
        with redirect_stdout(io.StringIO()):
            self.assertEqual(ns["renderImageEasyOCR"]("synthetic", "input.png", "unused"), (0, 0, 1))
        ns["readLicenseImage"].assert_called_once()

    def test_confidence_winner_and_ties_keep_text_paired(self):
        for raw, warped, expected in [(.95, .8, "RAW"), (.8, .95, "WARP"), (.9, .9, "WARP")]:
            with self.subTest(raw=raw, warped=warped):
                ns, cv = self.renderer()
                ns["readLicenseImage"].side_effect = [("RAW", raw), ("WARP", warped)]
                with redirect_stdout(io.StringIO()):
                    self.assertEqual(ns["renderImageEasyOCR"]("synthetic", "input.png", "unused"), (0, 0, 1))
                self.assertEqual(cv.putText.call_args.args[1], expected)


if __name__ == "__main__":
    unittest.main()
