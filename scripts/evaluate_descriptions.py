"""Mide cuánto de lo que describe el VLM es apariencia de las personas (RF4)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from monitoreo_urbano.common.config import settings  # noqa: E402
from monitoreo_urbano.vlm_describer.appearance import analyze_description  # noqa: E402
from monitoreo_urbano.vlm_describer.describer import build_describer, describe_without_physical_traits  # noqa: E402

FIGHT_DIRS = ("val", "test")
VIDEO_FRAMES = 6
SIZE = (960, 720)


def fight_images(limit: int) -> list[Path]:
    images: list[Path] = []
    for split in FIGHT_DIRS:
        images += sorted((settings.data_dir / "datasets" / "images" / split).glob("fight_*"))
    return images[:limit]


def video_frames(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for index in np.linspace(0, max(total - 2, 0), VIDEO_FRAMES).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, SIZE))
    cap.release()
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prompts-dir", type=Path, default=settings.vlm_prompts_dir)
    parser.add_argument("--label", default="actual")
    parser.add_argument("--fight-images", type=int, default=12)
    parser.add_argument("--videos", type=Path, nargs="*", default=[])
    parser.add_argument("--save", type=Path)
    parser.add_argument("--reasons", nargs="*", default=["posible_altercado", "default"], help="tipos de evento con los que se describe cada video")
    parser.add_argument("--no-guard", action="store_true", help="no reintenta cuando aparecen rasgos físicos")
    args = parser.parse_args()

    settings.vlm_prompts_dir = args.prompts_dir
    describer = build_describer()

    cases: list[tuple[str, list[np.ndarray], str, list[str]]] = []
    for image in fight_images(args.fight_images):
        frame = cv2.imread(str(image))
        if frame is not None:
            cases.append((image.name, [frame], "fight", ["person"]))
    for video in args.videos:
        frames = video_frames(video)
        for reason in args.reasons:
            cases.append((f"{video.name} [{reason}]", frames, reason, ["person"]))

    results = []
    for name, frames, reason, classes in cases:
        if args.no_guard:
            text = describer.describe(frames, "Cámara de prueba", classes, trigger_reason=reason).description_nl
        else:
            text = describe_without_physical_traits(describer, frames, "Cámara de prueba", classes, trigger_reason=reason).description_nl
        report = analyze_description(text)
        results.append({"caso": name, "motivo": reason, "texto": text, "rasgos": report.sensitive, "ropa": report.clothing})
        flag = "RASGOS " + ",".join(report.sensitive) if report.sensitive else "ok"
        print(f"[{args.label}] {name:40} {flag}", flush=True)

    total = len(results)
    with_traits = sum(1 for r in results if r["rasgos"])
    with_clothing = sum(1 for r in results if r["ropa"])
    print(f"\n[{args.label}] {total} descripciones: {with_traits} con rasgos físicos o demográficos "
          f"({100 * with_traits / max(total, 1):.0f}%), {with_clothing} con ropa ({100 * with_clothing / max(total, 1):.0f}%)")
    if args.save:
        args.save.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
