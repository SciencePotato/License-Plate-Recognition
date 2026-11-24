import csv
import io
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch


def row(filename="plate.jpg", **values):
    return dict(filename=filename, width=200, height=100,
                xmin=20, ymin=10, xmax=100, ymax=50, **values)


class AnnotationTests(unittest.TestCase):
    def convert(self, rows):
        outputs = {}
        csv_text = io.StringIO()
        writer = csv.DictWriter(csv_text, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        texts = {"./data/test_data.csv": csv_text.getvalue(),
                 "./data/train_data.csv": csv_text.getvalue().splitlines()[0] + "\n"}

        class Output(io.StringIO):
            def close(self):
                outputs[self.name] = self.getvalue()
                super().close()

        def open_file(path, mode="r", **kwargs):
            name = str(path)
            if "w" in mode:
                handle = Output()
                handle.name = name
                return handle
            return io.StringIO(texts[name])

        fake_pandas = SimpleNamespace(read_csv=lambda path: SimpleNamespace(
            iterrows=lambda: enumerate(rows if path.endswith("test_data.csv") else [])))
        with patch.dict(sys.modules, {"pandas": fake_pandas}), \
                patch("builtins.open", side_effect=open_file), \
                patch("os.makedirs"), patch("builtins.print"):
            runpy.run_path(str(Path(__file__).with_name("create.py")), run_name="__main__")
        return outputs

    def test_yolo_uses_normalized_center_and_extent(self):
        outputs = self.convert([row()])
        fields = next(iter(outputs.values())).split()
        self.assertEqual(fields[0], "0")
        self.assertEqual(list(map(float, fields[1:])), [0.3, 0.3, 0.4, 0.4])

    def test_all_boxes_for_one_image_are_preserved(self):
        outputs = self.convert([row(), row()])
        self.assertEqual(len(next(iter(outputs.values())).splitlines()), 2)

    def test_multi_dot_filename_keeps_its_stem(self):
        outputs = self.convert([row("vehicle.front.jpg")])
        self.assertEqual(list(outputs), ["./data/labels/vehicle.front.txt"])

    def test_invalid_dimensions_fail_with_clear_error(self):
        invalid = row()
        invalid["width"] = 0
        with self.assertRaisesRegex(ValueError, "dimensions"):
            self.convert([invalid])


if __name__ == "__main__":
    unittest.main()
