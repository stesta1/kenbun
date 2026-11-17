from __future__ import annotations

import argparse

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.db import get_connection, init_db
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.vlm_describer.describer import build_describer
from monitoreo_urbano.vlm_describer.worker import run_worker_loop

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Módulo 2: descripción de eventos con el VLM. Se pueden correr varios workers a la vez"
    )
    parser.add_argument(
        "--device",
        default=None,
        help="GPU de este worker (ej. cuda:1). Por defecto MU_VLM_DEVICE. Cada worker carga su propia copia del modelo",
    )
    args = parser.parse_args()
    if args.device:
        settings.vlm_device = args.device
    logger.info("Worker del VLM en %s", settings.vlm_device)

    conn = get_connection()
    init_db(conn)
    describer = build_describer()
    try:
        run_worker_loop(conn, describer)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
