from __future__ import annotations

import csv
import random
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data" / "datasets" / "_sources"
OUT = ROOT / "data" / "datasets"
SPLITS = ["train", "val", "test"]
RNG_SEED = 42


def _ensure_dirs() -> None:
    for split in SPLITS:
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)


def _write_label(split: str, stem: str, lines: list[str]) -> None:
    (OUT / "labels" / split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_fire_smoke() -> int:
    src = SOURCES / "fire_smoke" / "data"
    remap = {0: 4, 1: 3}
    count = 0

    for split in SPLITS:
        img_dir = src / split / "images"
        lbl_dir = src / split / "labels"
        if not img_dir.exists():
            continue

        for img_path in img_dir.iterdir():
            lbl_path = lbl_dir / f"{img_path.stem}.txt"
            if not lbl_path.exists():
                continue

            new_lines = []
            for line in lbl_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue
                cls = int(float(parts[0]))
                if cls not in remap:
                    continue
                new_lines.append(" ".join([str(remap[cls]), *parts[1:5]]))

            if not new_lines:
                continue

            stem = f"dfire_{img_path.stem}"
            shutil.copy2(img_path, OUT / "images" / split / f"{stem}{img_path.suffix}")
            _write_label(split, stem, new_lines)
            count += 1

    return count


def prepare_fight() -> int:
    src = SOURCES / "fight_synthetic" / "Aggressive_Poses_Dataset"
    img_dir = src / "images"
    lbl_dir = src / "labels"
    if not img_dir.exists():
        return 0

    images = sorted(img_dir.iterdir())
    rng = random.Random(RNG_SEED)
    rng.shuffle(images)
    n = len(images)
    split_at = {"train": int(n * 0.8), "val": int(n * 0.9)}

    count = 0
    for i, img_path in enumerate(images):
        split = "train" if i < split_at["train"] else ("val" if i < split_at["val"] else "test")
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        new_lines = []
        for line in lbl_path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            new_lines.append(" ".join(["6", *parts[1:5]]))

        if not new_lines:
            continue

        stem = f"fight_{img_path.stem}"
        shutil.copy2(img_path, OUT / "images" / split / f"{stem}{img_path.suffix}")
        _write_label(split, stem, new_lines)
        count += 1

    return count


def _extract_frame(video_path: Path, frame_index: int):
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            return None
        return frame
    finally:
        cap.release()


def _prepare_accident_csv(csv_path: Path, path_column: str, split_column: str | None) -> tuple[int, int]:
    accident_dir = SOURCES / "accident"
    rng = random.Random(RNG_SEED)
    ok_count = 0
    fail_count = 0

    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        video_path = accident_dir / row[path_column]
        if not video_path.exists():
            fail_count += 1
            continue

        if split_column and row.get(split_column) in ("train", "val", "test"):
            split = row[split_column]
        else:
            r = rng.random()
            split = "train" if r < 0.8 else ("val" if r < 0.9 else "test")

        try:
            frame_index = int(float(row["accident_frame"]))
            frame = _extract_frame(video_path, frame_index)
        except Exception:
            frame = None

        if frame is None:
            fail_count += 1
            continue

        cx, cy = float(row["center_x"]), float(row["center_y"])
        w = float(row["x2"]) - float(row["x1"])
        h = float(row["y2"]) - float(row["y1"])
        if w <= 0 or h <= 0:
            fail_count += 1
            continue

        stem = f"accident_{video_path.stem}"
        cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), frame)
        _write_label(split, stem, [f"5 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"])
        ok_count += 1

    return ok_count, fail_count


def prepare_accident() -> int:
    accident_dir = SOURCES / "accident"
    total_ok = 0

    real_csv = accident_dir / "metadata-real.csv"
    if real_csv.exists():
        ok, fail = _prepare_accident_csv(real_csv, path_column="path", split_column="split_in_distribution")
        print(f"  accident/real: {ok} ok, {fail} fallaron")
        total_ok += ok

    synth_csv = accident_dir / "metadata-synthetic.csv"
    if synth_csv.exists():
        ok, fail = _prepare_accident_csv(synth_csv, path_column="rgb_path", split_column=None)
        print(f"  accident/synthetic: {ok} ok, {fail} fallaron")
        total_ok += ok

    return total_ok


def write_data_yaml() -> None:
    content = """path: data/datasets
train: images/train
val: images/val
test: images/test

names:
  0: person
  1: car
  2: motorcycle
  3: fire
  4: smoke
  5: accident
  6: fight
"""
    (OUT / "data.yaml").write_text(content, encoding="utf-8")


def main() -> None:
    _ensure_dirs()

    print("Procesando D-Fire (fire/smoke)...")
    n_fire = prepare_fire_smoke()
    print(f"  {n_fire} imágenes")

    print("Procesando fight_synthetic...")
    n_fight = prepare_fight()
    print(f"  {n_fight} imágenes")

    print("Procesando ACCIDENT (extracción de frames)...")
    n_accident = prepare_accident()
    print(f"  {n_accident} imágenes")

    write_data_yaml()

    for split in SPLITS:
        n_img = len(list((OUT / "images" / split).iterdir()))
        n_lbl = len(list((OUT / "labels" / split).iterdir()))
        print(f"{split}: {n_img} imágenes, {n_lbl} labels")

    print(f"\ndata.yaml escrito en {OUT / 'data.yaml'}")


if __name__ == "__main__":
    sys.exit(main())
