from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from datetime import datetime

from psycopg_pool import ConnectionPool

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.db import get_pool, init_db
from monitoreo_urbano.common.repository import get_all_cameras, get_by_id, set_embedding, set_indexed
from monitoreo_urbano.common.schemas import CameraInfo, EventRecord, EventStatus, ReviewStatus, SearchResultItem
from monitoreo_urbano.event_store.embeddings import EmbeddingModel, SentenceTransformerEmbedding

_RRF_K = 60
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def _strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn")


def _build_tsquery(text: str) -> str:
    tokens = _TOKEN_PATTERN.findall(text)
    if not tokens:
        return ""
    return " | ".join(tokens)


def _scope_filters(
    camera_ids: list[str] | None,
    start_ts: datetime | None,
    end_ts: datetime | None,
    exclude_event_ids: list[str] | None = None,
    categories: list[str] | None = None,
    vehicle_type: str | None = None,
    vehicle_color: str | None = None,
) -> tuple[str, list]:
    # Solo se busca lo que un operador confirmó: lo pendiente o descartado no aparece, sea alerta o revisión.
    sql, params = " AND review_status = %s", [ReviewStatus.CONFIRMED.value]
    if categories:
        sql += " AND trigger_reason = ANY(%s)"
        params.append(categories)
    if vehicle_type or vehicle_color:
        # Los vehículos los carga el operador en event_vehicles; el color se compara sin tildes ni mayúsculas.
        sql += " AND EXISTS (SELECT 1 FROM event_vehicles v WHERE v.event_id = events.event_id"
        if vehicle_type:
            sql += " AND v.vehicle_type = %s"
            params.append(vehicle_type)
        if vehicle_color:
            sql += " AND translate(lower(v.color), 'áéíóúü', 'aeiouu') LIKE %s"
            params.append("%" + _strip_accents(vehicle_color) + "%")
        sql += ")"
    if exclude_event_ids:
        sql += " AND event_id <> ALL(%s)"
        params.append(exclude_event_ids)
    if camera_ids is not None:
        sql += " AND camera_id = ANY(%s)"
        params.append(camera_ids)
    if start_ts is not None:
        sql += " AND start_ts >= %s"
        params.append(start_ts)
    if end_ts is not None:
        sql += " AND end_ts <= %s"
        params.append(end_ts)
    return sql, params


def _fts_search(
    conn,
    query: str,
    top_k: int,
    camera_ids: list[str] | None,
    start_ts: datetime | None = None,
    end_ts: datetime | None = None,
    exclude_event_ids: list[str] | None = None,
    categories: list[str] | None = None,
    vehicle_type: str | None = None,
    vehicle_color: str | None = None,
) -> list[tuple[str, float]]:
    tsquery = _build_tsquery(query)
    if not tsquery:
        return []

    sql = """
        SELECT event_id, ts_rank_cd(search_vector, to_tsquery('spanish', %s)) AS score
        FROM events
        WHERE search_vector @@ to_tsquery('spanish', %s) AND status = %s
    """
    params: list = [tsquery, tsquery, EventStatus.INDEXED.value]
    filters, filter_params = _scope_filters(camera_ids, start_ts, end_ts, exclude_event_ids, categories, vehicle_type, vehicle_color)
    sql += filters
    params.extend(filter_params)
    sql += " ORDER BY score DESC LIMIT %s"
    params.append(top_k)

    cur = conn.execute(sql, params)
    return [(row["event_id"], float(row["score"])) for row in cur.fetchall()]


def _vector_search(
    conn,
    query_vector,
    top_k: int,
    camera_ids: list[str] | None,
    start_ts: datetime | None = None,
    end_ts: datetime | None = None,
    exclude_event_ids: list[str] | None = None,
    categories: list[str] | None = None,
    vehicle_type: str | None = None,
    vehicle_color: str | None = None,
) -> list[tuple[str, float]]:
    vector_literal = "[" + ",".join(str(float(v)) for v in query_vector) + "]"

    sql = """
        SELECT event_id, 1 - (embedding <=> %s::vector) AS score
        FROM events
        WHERE embedding IS NOT NULL AND status = %s
    """
    params: list = [vector_literal, EventStatus.INDEXED.value]
    filters, filter_params = _scope_filters(camera_ids, start_ts, end_ts, exclude_event_ids, categories, vehicle_type, vehicle_color)
    sql += filters
    params.extend(filter_params)
    sql += " ORDER BY embedding <=> %s::vector ASC LIMIT %s"
    params.extend([vector_literal, top_k])

    cur = conn.execute(sql, params)
    return [(row["event_id"], float(row["score"])) for row in cur.fetchall()]


class EventStore:
    def __init__(self, embedder: EmbeddingModel | None = None, pool: ConnectionPool | None = None) -> None:
        self.embedder = embedder or SentenceTransformerEmbedding()
        self.pool = pool or get_pool()
        with self.pool.connection() as conn:
            init_db(conn)

    def complete_event_description(self, event: EventRecord) -> None:
        assert event.description_nl, "El evento debe tener descripción antes de indexarse"
        indexed_text = event.description_nl
        if event.vehicles:
            indexed_text += " Vehículos involucrados: " + "; ".join(vehicle.label for vehicle in event.vehicles) + "."
        vector = self.embedder.embed([indexed_text])[0]
        with self.pool.connection() as conn:
            set_embedding(conn, event.event_id, vector)
            set_indexed(conn, event.event_id)

    def get_events(self, event_ids: list[str], camera_ids: frozenset[str] | None = None) -> list[EventRecord]:
        events: list[EventRecord] = []
        with self.pool.connection() as conn:
            for event_id in event_ids:
                event = get_by_id(conn, event_id)
                if event is not None and (camera_ids is None or event.camera_id in camera_ids):
                    events.append(event)
        return events

    @staticmethod
    def _camera_filter(camera_ids: frozenset[str] | None, camera_id: str | None) -> list[str] | None:
        allowed: frozenset[str] | None = camera_ids
        if camera_id:
            allowed = frozenset({camera_id}) if allowed is None else allowed & {camera_id}
        return None if allowed is None else sorted(allowed)

    def recent_events(
        self,
        camera_id: str | None = None,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        camera_ids: frozenset[str] | None = None,
        exclude_event_ids: frozenset[str] | None = None,
        categories: frozenset[str] | None = None,
        limit: int = 20,
        vehicle_type: str | None = None,
        vehicle_color: str | None = None,
    ) -> list[EventRecord]:
        camera_filter = self._camera_filter(camera_ids, camera_id)
        if camera_filter == []:
            return []
        filters, params = _scope_filters(
            camera_filter,
            start_ts,
            end_ts,
            sorted(exclude_event_ids) if exclude_event_ids else None,
            sorted(categories) if categories else None,
            vehicle_type,
            vehicle_color,
        )
        sql = f"SELECT event_id FROM events WHERE status = %s{filters} ORDER BY start_ts DESC LIMIT %s"
        with self.pool.connection() as conn:
            rows = conn.execute(sql, [EventStatus.INDEXED.value, *params, limit]).fetchall()
            events = [get_by_id(conn, row["event_id"]) for row in rows]
        return [event for event in events if event is not None]

    def list_cameras(self) -> list[CameraInfo]:
        with self.pool.connection() as conn:
            return get_all_cameras(conn)

    def hybrid_search(
        self,
        query: str,
        camera_id: str | None = None,
        start_ts: datetime | None = None,
        end_ts: datetime | None = None,
        camera_ids: frozenset[str] | None = None,
        exclude_event_ids: frozenset[str] | None = None,
        categories: frozenset[str] | None = None,
        vehicle_type: str | None = None,
        vehicle_color: str | None = None,
    ) -> list[SearchResultItem]:
        excluded = sorted(exclude_event_ids) if exclude_event_ids else None
        category_filter = sorted(categories) if categories else None
        camera_filter = self._camera_filter(camera_ids, camera_id)
        if camera_filter == []:
            return []

        query_vector = (
            self.embedder.embed_query(query)
            if hasattr(self.embedder, "embed_query")
            else self.embedder.embed([query])[0]
        )

        with self.pool.connection() as conn:
            vector_hits = _vector_search(
                conn,
                query_vector,
                settings.hybrid_search_top_k_vector,
                camera_filter,
                start_ts,
                end_ts,
                excluded,
                category_filter,
                vehicle_type,
                vehicle_color,
            )
            fts_hits = _fts_search(
                conn,
                query,
                settings.hybrid_search_top_k_fts,
                camera_filter,
                start_ts,
                end_ts,
                excluded,
                category_filter,
                vehicle_type,
                vehicle_color,
            )

            rrf_scores: dict[str, float] = defaultdict(float)
            vector_score_by_id: dict[str, float] = {}
            fts_score_by_id: dict[str, float] = {}

            for rank, (event_id, score) in enumerate(vector_hits):
                rrf_scores[event_id] += 1.0 / (_RRF_K + rank + 1)
                vector_score_by_id[event_id] = score

            for rank, (event_id, score) in enumerate(fts_hits):
                rrf_scores[event_id] += 1.0 / (_RRF_K + rank + 1)
                fts_score_by_id[event_id] = score

            ranked = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)

            results: list[SearchResultItem] = []
            for event_id, combined_score in ranked:
                event = get_by_id(conn, event_id)
                if event is None:
                    continue

                results.append(
                    SearchResultItem(
                        event=event,
                        vector_score=vector_score_by_id.get(event_id),
                        fts_score=fts_score_by_id.get(event_id),
                        combined_score=combined_score,
                    )
                )
                if len(results) >= settings.hybrid_search_final_k:
                    break

        return results

    def close(self) -> None:
        self.pool.close()
