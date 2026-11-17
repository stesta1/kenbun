from __future__ import annotations

import argparse
import signal

from monitoreo_urbano.common.cameras import get_camera
from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.instances import get_instance
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.runtime_config import RuntimeThresholds
from monitoreo_urbano.vision_pipeline.detector import YOLODetector
from monitoreo_urbano.vision_pipeline.instance import VisionInstance
from monitoreo_urbano.vision_pipeline.instance_store import PostgresInstanceStore

logger = get_logger(__name__)


def run(
    camera_ids: list[str],
    analysis_fps: float | None = None,
    preview: bool | None = None,
    device: str | None = None,
    name: str = "adhoc",
) -> None:
    cameras = [get_camera(camera_id) for camera_id in camera_ids]
    for camera in cameras:
        logger.info("Cámara %s (%s) <- %s", camera.camera_id, camera.label, camera.source)
    if preview is not None:
        settings.preview_enabled = preview

    store = PostgresInstanceStore()
    thresholds = RuntimeThresholds()
    store.refresh_thresholds(thresholds)
    detector = YOLODetector(thresholds=thresholds, device=device)

    instance = VisionInstance(
        cameras, detector=detector, store=store, thresholds=thresholds, analysis_fps=analysis_fps, name=name
    )

    def request_stop(signum, frame) -> None:  # noqa: ARG001
        logger.info("Señal %s: deteniendo la instancia", signum)
        instance.request_stop()

    for name in ("SIGINT", "SIGTERM", "SIGBREAK"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), request_stop)

    instance.start()
    try:
        while not instance.wait(1.0):
            pass
    finally:
        instance.stop()


def _camera_ids(values: list[str]) -> list[str]:
    ids: list[str] = []
    for value in values:
        ids.extend(part.strip() for part in value.split(",") if part.strip())
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Módulo 1: captura + detección + seguimiento + disparo de eventos, para una o varias cámaras"
    )
    parser.add_argument(
        "--instance",
        default=None,
        help="Nombre de una instancia de config/instances.yaml: fija sus cámaras, su GPU y sus fps de análisis",
    )
    parser.add_argument(
        "--camera-id",
        action="append",
        default=[],
        help="Debe existir en config/cameras.yaml. Se puede repetir o separar por comas para atender varias cámaras",
    )
    parser.add_argument(
        "--analysis-fps",
        type=float,
        default=None,
        help="Fotogramas por segundo a analizar por cámara (por defecto, los nativos de la cámara; ver MU_ANALYSIS_FPS)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        default=None,
        help="Publica data/live/<cámara>.jpg con las cajas dibujadas, para depurar (ver MU_PREVIEW_ENABLED)",
    )
    args = parser.parse_args()
    if bool(args.instance) == bool(args.camera_id):
        parser.error("Indicá --instance o --camera-id (uno solo de los dos)")

    if args.instance:
        instance = get_instance(args.instance)
        camera_ids, device, name = instance.cameras, instance.device, instance.name
        analysis_fps = args.analysis_fps if args.analysis_fps is not None else instance.analysis_fps
    else:
        camera_ids, device, name, analysis_fps = _camera_ids(args.camera_id), None, "adhoc", args.analysis_fps
    run(camera_ids, analysis_fps=analysis_fps, preview=args.preview, device=device, name=name)


if __name__ == "__main__":
    main()
