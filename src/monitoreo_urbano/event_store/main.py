from __future__ import annotations

from monitoreo_urbano.event_store.consumer import run_indexer_loop
from monitoreo_urbano.event_store.hybrid_search import EventStore


def main() -> None:
    store = EventStore()
    try:
        run_indexer_loop(store)
    finally:
        store.close()


if __name__ == "__main__":
    main()
