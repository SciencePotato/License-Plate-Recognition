import csv
import math
import os


def create_annotations(csv_paths, output_dir):
    annotations = {}
    for path in csv_paths:
        with open(path, newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                width, height, xmin, ymin, xmax, ymax = (
                    float(row[key]) for key in
                    ("width", "height", "xmin", "ymin", "xmax", "ymax"))
                if not all(math.isfinite(value) for value in
                           (width, height, xmin, ymin, xmax, ymax)):
                    raise ValueError("Annotation dimensions and coordinates must be finite")
                if width <= 0 or height <= 0:
                    raise ValueError("Image dimensions must be positive")
                if not (0 <= xmin < xmax <= width and 0 <= ymin < ymax <= height):
                    raise ValueError("Annotation bounds must lie inside the image")
                stem = os.path.splitext(os.path.basename(row["filename"]))[0]
                center_x = (xmin + xmax) / (2 * width)
                center_y = (ymin + ymax) / (2 * height)
                box_width = (xmax - xmin) / width
                box_height = (ymax - ymin) / height
                annotations.setdefault(stem, []).append(
                    f"0 {center_x} {center_y} {box_width} {box_height}\n")

    os.makedirs(output_dir, exist_ok=True)
    for stem, lines in annotations.items():
        with open(os.path.join(output_dir, stem + ".txt"), "w", encoding="utf-8") as target:
            target.writelines(lines)


if __name__ == "__main__":
    create_annotations(("./data/test_data.csv", "./data/train_data.csv"), "./data/labels")
