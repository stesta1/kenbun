from __future__ import annotations

import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from monitoreo_urbano.common.logging import get_logger

logger = get_logger(__name__)

_PENDING_WARNING_THRESHOLD = 50


@dataclass(frozen=True)
class ClipSinkStats:
    pending: int
    completed: int
    failed: int
    workers: int


class ClipSink:
    """Escribe los clips fuera del hilo del detector.

    Codificar el video, subirlo a MinIO e insertar el evento tarda decenas o cientos de milisegundos; hecho en
    el mismo hilo que la detección, frenaría el análisis de todas las cámaras mientras dura. Acá lo hace un
    grupo pequeño de hilos propios. `process` recibe lo que se encoló (normalmente una función de cierre de
    clip ya enlazada a su cámara) y se ejecuta en un hilo del grupo.

    Un clip que falla se registra y se cuenta; no detiene al grupo ni al resto de las cámaras.
    """

    def __init__(self, workers: int = 2) -> None:
        self._queue: queue.Queue[Callable[[], None] | None] = queue.Queue()
        self._workers = max(1, workers)
        self._threads = [
            threading.Thread(target=self._run, name=f"clip-writer-{i}", daemon=True) for i in range(self._workers)
        ]
        self._lock = threading.Lock()
        self._completed = 0
        self._failed = 0
        self._inflight = 0
        self._started = False

    def start(self) -> ClipSink:
        if not self._started:
            self._started = True
            for thread in self._threads:
                thread.start()
        return self

    def submit(self, job: Callable[[], None]) -> None:
        with self._lock:
            self._inflight += 1
            pending = self._inflight
        if pending == _PENDING_WARNING_THRESHOLD:
            logger.warning("Hay %d clips esperando a escribirse; el almacenamiento o el video van atrasados", pending)
        self._queue.put(job)

    def stats(self) -> ClipSinkStats:
        with self._lock:
            return ClipSinkStats(
                pending=self._inflight, completed=self._completed, failed=self._failed, workers=self._workers
            )

    def stop(self, timeout: float = 30.0) -> bool:
        for _ in self._threads:
            self._queue.put(None)
        end = time.monotonic() + timeout
        for thread in self._threads:
            if thread.ident is not None:
                thread.join(max(0.0, end - time.monotonic()))
        return all(not thread.is_alive() for thread in self._threads) and self.stats().pending == 0

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            try:
                job()
                with self._lock:
                    self._completed += 1
            except Exception:  # noqa: BLE001
                with self._lock:
                    self._failed += 1
                logger.exception("Falló la escritura de un clip")
            finally:
                with self._lock:
                    self._inflight -= 1
