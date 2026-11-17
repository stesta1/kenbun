from __future__ import annotations

import threading
import time

from monitoreo_urbano.vision_pipeline.clip_sink import ClipSink


def _wait_for(condition, timeout: float = 3.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return condition()


def test_runs_the_submitted_jobs():
    done: list[int] = []
    sink = ClipSink(workers=2).start()
    try:
        for index in range(5):
            sink.submit(lambda index=index: done.append(index))

        assert _wait_for(lambda: len(done) == 5)
        assert sorted(done) == [0, 1, 2, 3, 4]
        assert _wait_for(lambda: sink.stats().completed == 5)
    finally:
        sink.stop()


def test_a_failing_job_is_counted_and_does_not_stop_the_others():
    done: list[int] = []
    sink = ClipSink(workers=1).start()
    try:

        def boom() -> None:
            raise OSError("MinIO no responde")

        sink.submit(boom)
        sink.submit(lambda: done.append(1))

        assert _wait_for(lambda: done == [1])
        stats = sink.stats()
        assert _wait_for(lambda: sink.stats().failed == 1 and sink.stats().completed == 1)
        assert stats.workers == 1
    finally:
        sink.stop()


def test_jobs_run_in_other_threads_and_do_not_block_the_caller():
    release = threading.Event()
    started = threading.Event()
    sink = ClipSink(workers=1).start()
    try:

        def slow() -> None:
            started.set()
            release.wait(5)

        began = time.monotonic()
        sink.submit(slow)
        elapsed = time.monotonic() - began

        assert elapsed < 0.2
        assert started.wait(2)
        assert sink.stats().pending == 1
    finally:
        release.set()
        sink.stop()


def test_stop_waits_for_the_pending_jobs():
    done: list[int] = []
    sink = ClipSink(workers=1).start()
    for index in range(3):
        sink.submit(lambda index=index: (time.sleep(0.05), done.append(index)))

    finished = sink.stop(timeout=5)

    assert finished is True
    assert done == [0, 1, 2]
    assert sink.stats().pending == 0


def test_stop_reports_when_it_ran_out_of_time():
    release = threading.Event()
    sink = ClipSink(workers=1).start()
    sink.submit(lambda: release.wait(5))
    try:
        assert sink.stop(timeout=0.2) is False
    finally:
        release.set()


def test_stop_before_start_does_not_fail():
    assert ClipSink(workers=2).stop(timeout=1) in (True, False)
