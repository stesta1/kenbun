"""Clasificador de choque por trayectorias de vehículos, entrenado desde cero con ACCIDENT real.

Uso:
    python scripts/train_trajectory_classifier.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_classifier_events as ece  # noqa: E402
from monitoreo_urbano.common.schemas import BoundingBox, Detection  # noqa: E402
from monitoreo_urbano.vision_pipeline.camera_source import Frame  # noqa: E402
from monitoreo_urbano.vision_pipeline.tracker import ByteTrackTracker  # noqa: E402

CACHE = ROOT / "data" / "eval_cache"
TEACHER = ROOT / "data" / "teacher_cache"
FPS = 6.0
VEHICLES = ("car", "motorcycle")
CONF = 0.35
LAG = 3
WIN = 6
POS_AFTER_S = 4.0
NEG_BEFORE_S = 2.0
FEATURES = ["n_veh", "mean_speed", "max_speed", "mean_speed_before", "frac_stopped", "max_decel", "stopped_from_moving",
            "max_dir_change", "max_pair_iou", "max_iou_increase", "min_pair_dist", "tracks_lost", "tracks_new", "log_area_ratio",
            "r_max_decel", "r_stopped_from_moving", "r_max_dir_change", "r_max_iou_increase", "r_tracks_lost"]


def cache_path(weights_id: str, video: Path, imgsz: int = 960, fps: float = FPS) -> Path:
    key = hashlib.md5(f"{weights_id}|{video.name}|{float(fps)}|0.2|{imgsz}".encode()).hexdigest()[:12]
    return CACHE / f"{video.stem.replace(' ', '_')}_{key}.json"


def run_tracker(frames: list[dict]) -> list[dict[int, tuple]]:
    tracker = ByteTrackTracker(frame_rate=FPS, track_buffer_seconds=3.0)
    black = np.zeros((8, 8, 3), dtype=np.uint8)
    base = datetime(2026, 1, 1)
    out = []
    for i, item in enumerate(frames):
        dets = [Detection(class_name=n, confidence=c, bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2))
                for n, c, x1, y1, x2, y2 in item["dets"] if n in VEHICLES and c >= CONF]
        tracked = tracker.update(dets, Frame(image=black, timestamp=base + timedelta(seconds=item["t"]), frame_index=i))
        out.append({t.track_id: (t.bbox.x1, t.bbox.y1, t.bbox.x2, t.bbox.y2) for t in tracked})
    return out


def iou(a: tuple, b: tuple) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def centre(b: tuple) -> tuple[float, float, float]:
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, math.sqrt(max((b[2] - b[0]) * (b[3] - b[1]), 1.0))


def velocity(tracks: list[dict], i: int, tid: int):
    if i - LAG < 0 or tid not in tracks[i] or tid not in tracks[i - LAG]:
        return None
    x1, y1, s1 = centre(tracks[i][tid])
    x0, y0, s0 = centre(tracks[i - LAG][tid])
    dx, dy = x1 - x0, y1 - y0
    scale = (s1 + s0) / 2
    return math.hypot(dx, dy) / scale / (LAG / FPS), math.atan2(dy, dx)


def frame_features(tracks: list[dict], i: int) -> list[float]:
    now = tracks[i]
    before_i = i - WIN
    v_now = {t: velocity(tracks, i, t) for t in now}
    v_now = {t: v for t, v in v_now.items() if v}
    v_bef = {}
    if before_i >= 0:
        v_bef = {t: velocity(tracks, before_i, t) for t in tracks[before_i]}
        v_bef = {t: v for t, v in v_bef.items() if v}
    speeds = [v[0] for v in v_now.values()]
    decel, stopped, dirch = 0.0, 0, 0.0
    for t, (s1, a1) in v_now.items():
        if t in v_bef:
            s0, a0 = v_bef[t]
            decel = max(decel, s0 - s1)
            stopped += int(s0 > 1.0 and s1 < 0.3)
            if s0 > 0.5 and s1 > 0.5:
                d = abs(a1 - a0)
                dirch = max(dirch, min(d, 2 * math.pi - d))
    ids = list(now)
    max_iou, min_dist, max_inc = 0.0, 5.0, 0.0
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            v = iou(now[ids[a]], now[ids[b]])
            max_iou = max(max_iou, v)
            ca, cb = centre(now[ids[a]]), centre(now[ids[b]])
            min_dist = min(min_dist, math.hypot(ca[0] - cb[0], ca[1] - cb[1]) / ((ca[2] + cb[2]) / 2))
            if before_i >= 0 and ids[a] in tracks[before_i] and ids[b] in tracks[before_i]:
                max_inc = max(max_inc, v - iou(tracks[before_i][ids[a]], tracks[before_i][ids[b]]))
    recent = range(max(0, i - WIN + 1), i + 1)
    seen_before = set().union(*[set(tracks[j]) for j in range(max(0, i - 2 * WIN + 1), max(0, i - WIN + 1))]) if i - WIN > 0 else set()
    lost = len(seen_before - set().union(*[set(tracks[j]) for j in recent]))
    new = len(set().union(*[set(tracks[j]) for j in recent]) - seen_before) if seen_before else 0
    area_now = sum((b[2] - b[0]) * (b[3] - b[1]) for b in now.values())
    area_bef = sum((b[2] - b[0]) * (b[3] - b[1]) for b in tracks[before_i].values()) if before_i >= 0 else 0.0
    ratio = math.log((area_now + 1.0) / (area_bef + 1.0))
    return [len(now), float(np.mean(speeds)) if speeds else 0.0, max(speeds) if speeds else 0.0,
            float(np.mean([v[0] for v in v_bef.values()])) if v_bef else 0.0,
            sum(s < 0.2 for s in speeds) / len(speeds) if speeds else 0.0, decel, stopped, dirch, max_iou, max_inc, min_dist,
            lost, new, ratio]


def video_matrix(tracks: list[dict]) -> np.ndarray:
    base = np.array([frame_features(tracks, i) for i in range(len(tracks))], dtype=float)
    idx = {n: k for k, n in enumerate(FEATURES[:14])}
    cols = [idx["max_decel"], idx["stopped_from_moving"], idx["max_dir_change"], idx["max_iou_increase"], idx["tracks_lost"]]
    rolling = np.zeros((len(tracks), len(cols)))
    for i in range(len(tracks)):
        rolling[i] = base[max(0, i - WIN + 1): i + 1, cols].max(axis=0)
    return np.hstack([base, rolling])


def main() -> None:
    global CONF
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=Path, default=ROOT / "data" / "models" / "yolo" / "baseline_v6.pt")
    parser.add_argument("--out", type=Path, default=ROOT.parent / "reentreno" / "cls_traj_probs.json")
    parser.add_argument("--source", choices=("v6", "teacher"), default="v6")
    parser.add_argument("--conf", type=float, default=CONF)
    args = parser.parse_args()
    weights_id = f"{args.weights.name}-{args.weights.stat().st_size}-{int(args.weights.stat().st_mtime)}"
    meta = {Path(r["path"]).stem: r for r in csv.DictReader(open(ece.SOURCE / "metadata-real.csv", encoding="utf-8"))}
    splits = {s: ece.split_stems(s) for s in ("train", "val", "test")}
    CONF = args.conf
    mats: dict[str, np.ndarray] = {}
    times: dict[str, list[float]] = {}

    def load(stem: str, video: Path):
        path = TEACHER / f"{stem.replace(' ', '_')}.json" if args.source == "teacher" else cache_path(weights_id, video)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))["frames"]

    for n, stem in enumerate(sorted(meta), 1):
        frames = load(stem, ece.SOURCE / meta[stem]["path"])
        if frames:
            mats[stem] = video_matrix(run_tracker(frames))
            times[stem] = [f["t"] for f in frames]
        if n % 200 == 0:
            print(f"características {n}/{len(meta)}", flush=True)
    frames = load("V7 uncut", ece.V7)
    if frames:
        mats["V7 uncut"] = video_matrix(run_tracker(frames))
        times["V7 uncut"] = [f["t"] for f in frames]
    print("videos con características:", len(mats))

    def rows(split: str):
        X, y = [], []
        for stem in sorted(splits[split]):
            if stem not in mats:
                continue
            t0 = float(meta[stem]["accident_time"])
            for i in range(WIN * 2, len(mats[stem])):
                t = times[stem][i]
                if t0 <= t <= t0 + POS_AFTER_S:
                    X.append(mats[stem][i]); y.append(1)
                elif t < t0 - NEG_BEFORE_S:
                    X.append(mats[stem][i]); y.append(0)
        return np.array(X), np.array(y)

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score, roc_auc_score

    Xtr, ytr = rows("train")
    Xva, yva = rows("val")
    print(f"train {len(ytr)} (choque {int(ytr.sum())}), val {len(yva)} (choque {int(yva.sum())})")
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
                                           class_weight="balanced", early_stopping=True, validation_fraction=0.15, random_state=0)
    model.fit(Xtr, ytr)
    pva = model.predict_proba(Xva)[:, 1]
    print(f"val por fotograma: AUC {roc_auc_score(yva, pva):.3f}, AP {average_precision_score(yva, pva):.3f}")

    out = {"weights": "trayectorias (HistGradientBoosting)", "videos": {}}
    for stem, M in mats.items():
        p = np.zeros(len(M))
        if len(M) > WIN * 2:
            p[WIN * 2:] = model.predict_proba(M[WIN * 2:])[:, 1]
        row = {"split": "normal" if stem == "V7 uncut" else next(s for s in ("train", "val", "test") if stem in splits[s]),
               "fps": FPS, "t_from": times[stem][0], "p": [round(float(x), 4) for x in p]}
        if stem != "V7 uncut":
            m = meta[stem]
            row.update({"accident_time": float(m["accident_time"]), "quality": m["quality"], "duration": float(m["duration"]),
                        "height": int(m["height"]), "scene": m["scene_layout"], "weather": m["weather"], "day_time": m["day_time"]})
        if row["split"] != "train":
            out["videos"][stem] = row
    args.out.write_text(json.dumps(out), encoding="utf-8")
    print("escrito", args.out, len(out["videos"]))


if __name__ == "__main__":
    main()
