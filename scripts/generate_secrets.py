"""Genera las credenciales propias del despliegue y las escribe en .env (RNF5)."""
from __future__ import annotations

import argparse
import secrets
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parents[1] / ".env"
POSTGRES_USER = "monitoreo"
POSTGRES_DB = "monitoreo_urbano"
API_PORT = 8000


def _build_values() -> dict[str, str]:
    postgres_password = secrets.token_urlsafe(24)
    minio_user = f"mu-{secrets.token_hex(4)}"
    minio_password = secrets.token_urlsafe(24)
    return {
        "MU_API_KEY": secrets.token_urlsafe(32),
        "POSTGRES_USER": POSTGRES_USER,
        "POSTGRES_PASSWORD": postgres_password,
        "MU_DATABASE_URL": f"postgresql://{POSTGRES_USER}:{postgres_password}@localhost:5432/{POSTGRES_DB}",
        "MINIO_ROOT_USER": minio_user,
        "MINIO_ROOT_PASSWORD": minio_password,
        "MU_S3_ACCESS_KEY": minio_user,
        "MU_S3_SECRET_KEY": minio_password,
        "MTX_AUTHMETHOD": "http",
        "MTX_AUTHHTTPADDRESS": f"http://host.docker.internal:{API_PORT}/auth/mediamtx",
    }


_LINKED = {"POSTGRES_PASSWORD", "MU_DATABASE_URL", "MINIO_ROOT_USER", "MINIO_ROOT_PASSWORD", "MU_S3_ACCESS_KEY", "MU_S3_SECRET_KEY"}


def _read_existing(lines: list[str]) -> dict[str, int]:
    found: dict[str, int] = {}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            found[stripped.split("=", 1)[0].strip()] = index
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="regenera todos los valores, aunque ya existan")
    parser.add_argument("--env-file", type=Path, default=ENV_PATH)
    parser.add_argument(
        "--access-only",
        action="store_true",
        help="solo la clave de servicio y la autenticación de MediaMTX; no toca las credenciales de "
        "Postgres y MinIO (úsalo si ya hay volúmenes con datos)",
    )
    args = parser.parse_args()

    lines = args.env_file.read_text(encoding="utf-8").splitlines() if args.env_file.exists() else []
    existing = _read_existing(lines)
    values = _build_values()
    if args.access_only:
        values = {k: v for k, v in values.items() if k not in _LINKED and k != "POSTGRES_USER"}

    keep_linked = not args.force and any(key in existing for key in _LINKED)

    written: list[str] = []
    appended: list[str] = []
    for key, value in values.items():
        if key in existing:
            if not args.force and (key not in _LINKED or keep_linked):
                continue
            lines[existing[key]] = f"{key}={value}"
            written.append(key)
        elif key in _LINKED and keep_linked:
            continue
        else:
            appended.append(f"{key}={value}")
            written.append(key)

    if appended:
        if lines and lines[-1].strip():
            lines.append("")
        lines.append("# --- Credenciales generadas por scripts/generate_secrets.py (RNF5) ---")
        lines.extend(appended)

    if not written:
        print(f"{args.env_file} ya tiene todas las credenciales. Usá --force para regenerarlas.")
        return

    args.env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Escribí en {args.env_file}: {', '.join(written)}")
    print("Los valores no se muestran acá; están en ese archivo (que git ignora).")
    if keep_linked and any(key in _LINKED for key in existing):
        print("Las credenciales de Postgres y MinIO ya existían y no se tocaron (usá --force para cambiarlas).")
    if any(key in _LINKED for key in written):
        print("Si ya hay volúmenes de Postgres o MinIO, recrearlos o cambiar la contraseña adentro (ver ayuda).")
    print("Para MediaMTX: docker compose up -d --force-recreate mediamtx")


if __name__ == "__main__":
    main()
