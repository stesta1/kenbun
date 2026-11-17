from __future__ import annotations

from pathlib import Path

from monitoreo_urbano.common.config import settings

_DEFAULT_CATEGORY = "default"
# Marca que `avoid_terms` lleva cuando la descripción narró imagen por imagen en lugar de contar el hecho completo.
NARRATION_TERM = "narración por fotogramas"


def _read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Falta el archivo de prompt {path} (ver settings.vlm_prompts_dir)")
    return path.read_text(encoding="utf-8").strip()


def appearance_restricted(trigger_reason: str | None) -> bool:
    return trigger_reason not in settings.vlm_appearance_exempt_reasons


def build_system_prompt(trigger_reason: str | None) -> str:
    preamble = _read(settings.vlm_prompts_dir / "_preamble.txt")
    output_format = _read(settings.vlm_prompts_dir / "_output_format.txt")
    appearance_rules = _read(settings.vlm_prompts_dir / "_appearance_rules.txt") if appearance_restricted(trigger_reason) else ""

    category_path = settings.vlm_prompts_dir / f"{trigger_reason}.txt" if trigger_reason else None
    if category_path is None or not category_path.exists():
        category_path = settings.vlm_prompts_dir / f"{_DEFAULT_CATEGORY}.txt"
    category_focus = _read(category_path)

    sections = [preamble, appearance_rules, category_focus, output_format]
    return "\n\n".join(section for section in sections if section)


def build_user_prompt(camera_label: str, object_classes: list[str], avoid_terms: list[str] | None = None) -> str:
    classes = ", ".join(object_classes) if object_classes else "sin clasificar"
    prompt = (
        f"Cámara: {camera_label}. Clases detectadas por el sistema de visión: {classes}. "
        "Describí el video adjunto siguiendo las reglas del sistema, "
        "con una única línea DESCRIPCION."
    )
    traits = [term for term in avoid_terms or [] if term != NARRATION_TERM]
    if traits:
        prompt += (
            f" Tu respuesta anterior mencionó rasgos físicos o demográficos de las personas ({', '.join(traits)}). "
            "Escribila de nuevo sin ninguno de esos términos: nombrá a las personas solo como «una persona», "
            "«otra persona» o «las personas» y describí únicamente lo que hacen."
        )
    if NARRATION_TERM in (avoid_terms or []):
        prompt += (
            " Tu respuesta anterior narró imagen por imagen. Escribila de nuevo como un único relato continuo del hecho, "
            "de principio a fin, sin las palabras «fotograma», «frame», «imagen», «primero», «segundo» referidas a imágenes "
            "ni «en la secuencia»."
        )
    return prompt
