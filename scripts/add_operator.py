"""Da de alta un operador en config/operators.yaml y muestra su clave (una sola vez)."""
from __future__ import annotations

import argparse
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from monitoreo_urbano.common.auth import OPEN_OPERATOR_NAME, SERVICE_OPERATOR_NAME, load_operators  # noqa: E402
from monitoreo_urbano.common.config import settings  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("name", help="nombre del operador; queda registrado en cada decisión de revisión")
    parser.add_argument("--cameras", help="camera_id separados por coma; omitir para todas")
    parser.add_argument("--file", type=Path, default=settings.operators_config_path)
    args = parser.parse_args()

    name = args.name.strip()
    if not name or any(c in name for c in ":#'\"\n"):
        sys.exit("El nombre no puede estar vacío ni tener los caracteres : # ' \" ni saltos de línea.")
    if name in (SERVICE_OPERATOR_NAME, OPEN_OPERATOR_NAME) or name in {op.name for _, op in load_operators(args.file)}:
        sys.exit(f"Ya existe un operador llamado {name!r} (o el nombre está reservado).")

    key = secrets.token_urlsafe(32)
    entry = f"  - name: {name}\n    key: {key}\n"
    if args.cameras:
        cameras = [c.strip() for c in args.cameras.split(",") if c.strip()]
        entry += f"    cameras: [{', '.join(cameras)}]\n"

    text = args.file.read_text(encoding="utf-8") if args.file.exists() else "operators:\n"
    if text and not text.endswith("\n"):
        text += "\n"
    args.file.write_text(text + "\n" + entry, encoding="utf-8")

    print(f"Operador {name!r} agregado a {args.file}.")
    print(f"Clave (anotala ahora, no se vuelve a mostrar en ningún otro lado): {key}")


if __name__ == "__main__":
    main()
