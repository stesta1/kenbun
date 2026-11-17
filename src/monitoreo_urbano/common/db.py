from __future__ import annotations

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from monitoreo_urbano.common.config import settings

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS events (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    event_id TEXT UNIQUE NOT NULL,
    camera_id TEXT NOT NULL,
    camera_label TEXT NOT NULL,
    track_id INTEGER NOT NULL,
    object_classes TEXT NOT NULL,          -- JSON list, ej. ["car", "car"]
    trigger_reason TEXT,                   -- motivo devuelto por EventRule, ej. "persona_junto_a_vehiculo"
    start_ts TIMESTAMP NOT NULL,
    end_ts TIMESTAMP NOT NULL,
    clip_path TEXT NOT NULL,               -- object key en el bucket settings.s3_bucket_clips
    status TEXT NOT NULL DEFAULT 'pending_description',
    description_nl TEXT,
    attributes TEXT,                       -- JSON (EventAttributes)
    confidence DOUBLE PRECISION,
    review_status TEXT NOT NULL DEFAULT 'pending',
    reviewed_by TEXT,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,         -- último cambio de cualquier tipo; cursor de polling para alertas
    described_at TIMESTAMP,
    search_vector tsvector
);

-- Dimensión fija en 1024 porque embedding_model_name (multilingual-e5-large) es una config fija del
-- sistema, no algo que cambie en runtime. Cambiar de modelo a uno de otra dimensión requiere migrar.
ALTER TABLE events ADD COLUMN IF NOT EXISTS embedding vector(1024);
ALTER TABLE events DROP COLUMN IF EXISTS faiss_index_id;

-- Corrección humana de la descripción (CU4). description_nl_original conserva lo que escribió el
-- VLM la primera vez que una corrección lo pisa; queda NULL mientras nadie corrigió nada.
ALTER TABLE events ADD COLUMN IF NOT EXISTS description_nl_original TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS description_edited BOOLEAN NOT NULL DEFAULT FALSE;
ALTER TABLE events ADD COLUMN IF NOT EXISTS trigger_reason_original TEXT;

-- Vehículos de un choque, cargados por el operador al revisar el evento (tipo fijo, color en texto libre).
-- events.vehicles guarda la misma lista en JSON para leer el evento sin joins; events.vehicles_text es el texto
-- que entra al índice de búsqueda y al embedding. event_vehicles es la fuente para filtrar por tipo y color.
ALTER TABLE events ADD COLUMN IF NOT EXISTS vehicles TEXT;
ALTER TABLE events ADD COLUMN IF NOT EXISTS vehicles_text TEXT;

CREATE TABLE IF NOT EXISTS event_vehicles (
    event_id TEXT NOT NULL REFERENCES events(event_id) ON DELETE CASCADE,
    position INTEGER NOT NULL,
    vehicle_type TEXT NOT NULL,
    color TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (event_id, position)
);
CREATE INDEX IF NOT EXISTS idx_event_vehicles_type_color ON event_vehicles(vehicle_type, color);

CREATE TABLE IF NOT EXISTS runtime_config (
    key TEXT PRIMARY KEY,               -- nombre del setting, ej. "yolo_confidence_threshold"
    value TEXT NOT NULL,                -- override activo; si no hay fila, se usa el default de .env
    updated_at TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS cameras (
    camera_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    source TEXT NOT NULL,               -- URL RTSP republicada por MediaMTX (o cualquier fuente de CameraSource)
    comuna TEXT,
    lat DOUBLE PRECISION,
    lon DOUBLE PRECISION,
    updated_at TIMESTAMP NOT NULL
);

-- Calles donde está la cámara, ej. «Av. Santa Fe y Callao». La consulta la usa para ubicar el hecho y buscar por lugar.
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS ubicacion TEXT;

ALTER TABLE cameras ADD COLUMN IF NOT EXISTS health_status TEXT NOT NULL DEFAULT 'ok';
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS health_since TIMESTAMP;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS health_updated_at TIMESTAMP;
CREATE INDEX IF NOT EXISTS idx_cameras_health_updated_at ON cameras(health_updated_at);

-- Latido de la instancia de visión que analiza la cámara. Si deja de renovarse, la cámara figura como
-- 'sin_analisis' aunque su último estado de salud haya sido 'ok' (la instancia cayó y nadie lo escribe).
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS analysis_heartbeat_at TIMESTAMP;
ALTER TABLE cameras ADD COLUMN IF NOT EXISTS analysis_instance TEXT;

CREATE INDEX IF NOT EXISTS idx_events_status ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id);
CREATE INDEX IF NOT EXISTS idx_events_start_ts ON events(start_ts);
CREATE INDEX IF NOT EXISTS idx_events_updated_at ON events(updated_at);
CREATE INDEX IF NOT EXISTS idx_events_search_vector ON events USING GIN(search_vector);

CREATE OR REPLACE FUNCTION events_search_vector_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector :=
        setweight(to_tsvector('spanish', coalesce(NEW.description_nl, '')), 'A') ||
        setweight(to_tsvector('spanish', coalesce(NEW.vehicles_text, '')), 'A') ||
        setweight(to_tsvector('spanish', coalesce(NEW.camera_label, '')), 'B') ||
        setweight(to_tsvector('spanish', coalesce((SELECT ubicacion FROM cameras WHERE camera_id = NEW.camera_id), '')), 'B') ||
        setweight(to_tsvector('spanish', coalesce(NEW.object_classes, '')), 'C');
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS events_search_vector_trigger ON events;
CREATE TRIGGER events_search_vector_trigger
BEFORE INSERT OR UPDATE ON events
FOR EACH ROW EXECUTE FUNCTION events_search_vector_update();
"""


def get_connection(database_url: str | None = None) -> psycopg.Connection:
    return psycopg.connect(database_url or settings.database_url, row_factory=dict_row, autocommit=True)


def get_pool(database_url: str | None = None) -> ConnectionPool:
    return ConnectionPool(
        database_url or settings.database_url,
        kwargs={"row_factory": dict_row},
        min_size=1,
        max_size=8,
        open=True,
    )


def init_db(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)
    conn.commit()
