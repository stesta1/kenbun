from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402

from monitoreo_urbano.vision_pipeline.detector import YOLODetector  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "datasets"
SPLITS = ["train", "val", "test"]

COCO_TO_UNIFIED = {"person": 0, "car": 1, "motorcycle": 2}
CONFIDENCE_THRESHOLD = 0.45


def main() -> None:
    detector = YOLODetector(
        weights_path=str(ROOT / "data" / "models" / "yolo" / "yolo11n_coco.pt"),
        device="cuda:0",
        confidence_threshold=CONFIDENCE_THRESHOLD,
        classes=list(COCO_TO_UNIFIED),
    )

    total_images = 0
    total_added = 0

    for split in SPLITS:
        img_dir = OUT / "images" / split
        lbl_dir = OUT / "labels" / split
        images = sorted(img_dir.iterdir())

        for i, img_path in enumerate(images):
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            h, w = image.shape[:2]

            detections = detector.detect(image)
            new_lines = []
            for det in detections:
                cls_idx = COCO_TO_UNIFIED[det.class_name]
                cx = (det.bbox.x1 + det.bbox.x2) / 2 / w
                cy = (det.bbox.y1 + det.bbox.y2) / 2 / h
                bw = (det.bbox.x2 - det.bbox.x1) / w
                bh = (det.bbox.y2 - det.bbox.y1) / h
                new_lines.append(f"{cls_idx} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if new_lines:
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                with open(lbl_path, "a", encoding="utf-8") as fh:
                    fh.write("\n".join(new_lines) + "\n")
                total_added += len(new_lines)

            total_images += 1
            if total_images % 1000 == 0:
                print(f"  {total_images} imágenes procesadas, {total_added} cajas agregadas hasta ahora")

        print(f"{split}: listo")

    print(f"\nTotal: {total_images} imágenes, {total_added} cajas de person/car/motorcycle agregadas")


if __name__ == "__main__":
    main()
