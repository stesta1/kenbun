from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from PIL import Image

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger
from monitoreo_urbano.common.schemas import CATEGORY_LABELS, EventAttributes
from monitoreo_urbano.vlm_describer.appearance import analyze_description
from monitoreo_urbano.vlm_describer.prompts import (
    NARRATION_TERM,
    appearance_restricted,
    build_system_prompt,
    build_user_prompt,
)

logger = get_logger(__name__)

_RESPONSE_PATTERN = re.compile(r"DESCRIPCION:\s*(.*?)\s*ATRIBUTOS:\s*(\{.*\})\s*$", re.DOTALL)

SCENE_TEXT_MARK = "[texto de la escena]"
# Fragmentos entre comillas y palabras en mayúsculas sostenidas (con guion, o de ocho letras o más y sin dos puntos, que
# son las etiquetas de los campos): así transcribe el modelo un cartel. Son heurísticas: reducen el riesgo, no lo eliminan.
_QUOTED_TEXT = tuple(re.compile(p) for p in (r"«[^»\n]{1,200}»", r"“[^”\n]{1,200}”", r'"[^"\n]{1,200}"'))
_SHOUTED_TEXT = re.compile(r"\b[A-ZÁÉÍÓÚÑ]{3,}(?:-[A-ZÁÉÍÓÚÑ]{2,})+\b|\b[A-ZÁÉÍÓÚÑ]{8,}\b(?!:)")


# Frases que hablan de texto en la escena (el cartel, lo que dice, una pantalla): se quitan enteras. La descripción es solo de hechos.
_TEXT_WORDS = re.compile(
    r"\b(?:cartel(?:es)?|letrero|pancarta|pantalla|texto|leyenda|inscripci[oó]n|r[oó]tulo|banner|pizarra|anuncio|publicidad|"
    r"escrit[oa]s?|dice|dicen|palabras?|mensaje|letras?|se lee)\b",
    re.IGNORECASE,
)
_SENTENCE = re.compile(r"[^.!?\n]+[.!?]?")


def strip_text_mentions(text: str) -> str:
    out = _SENTENCE.sub(lambda m: "" if _TEXT_WORDS.search(m.group(0)) else m.group(0), text)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]{2,}", " ", out)).strip()


def mask_scene_text(text: str) -> str:
    for pattern in _QUOTED_TEXT:
        text = pattern.sub(SCENE_TEXT_MARK, text)
    return _SHOUTED_TEXT.sub(SCENE_TEXT_MARK, text)


def _mask_attributes(attributes: EventAttributes) -> EventAttributes:
    return EventAttributes(
        vehicle_colors=[mask_scene_text(v) for v in attributes.vehicle_colors],
        clothing_colors=[mask_scene_text(v) for v in attributes.clothing_colors],
        direction=mask_scene_text(attributes.direction) if attributes.direction else attributes.direction,
        behavior_tags=[mask_scene_text(v) for v in attributes.behavior_tags],
    )


@dataclass
class DescriptionResult:
    description_nl: str
    attributes: EventAttributes
    confidence: float | None = None


class VLMDescriber(ABC):
    @abstractmethod
    def describe(
        self,
        frames: list[np.ndarray],
        camera_label: str,
        object_classes: list[str],
        trigger_reason: str | None = None,
        avoid_terms: list[str] | None = None,
    ) -> DescriptionResult:
        raise NotImplementedError


MIN_DESCRIPTION_CHARS = 60
# Calificaciones que contradicen un hecho detectado por el sistema ("fue un simulacro") y negaciones del hecho ("no es un choque").
# Son heurísticas sobre lo que suele imponer un cartel; no cubren cualquier formulación.
_CONTRADICTION = re.compile(
    r"\b(?:simulacro|simulaci[oó]n|puesta en escena|falsa alarma|pel[ií]cula|rodaje|filmaci[oó]n)\b"
    r"|\bno (?:es|fue|hubo|hay|se trata de) (?:un |una |ning[uú]n |ninguna )?(?:choque|colisi[oó]n|accidente|incendio|fuego|pelea|altercado)\b",
    re.IGNORECASE,
)


# Términos de cada hecho. Una descripción que habla de un hecho de otra categoría que la detectada no se guarda: la categoría la fija
# el sistema y un cartel no puede cambiarla.
_FIRE = re.compile(r"incendi|fuego|llamas|ardiendo", re.IGNORECASE)
_CRASH = re.compile(r"choc|colisi|estrell|embist|atropell|volc|impact", re.IGNORECASE)
_FIGHT = re.compile(r"pelea|pelean|agred|forceje|altercado|pu[ñn]etazo|patada", re.IGNORECASE)
_FOREIGN = {
    "accident": (_FIRE, _FIGHT),
    "posible_choque": (_FIRE, _FIGHT),
    "fire": (_CRASH, _FIGHT),
    "smoke": (_CRASH, _FIGHT),
    "fight": (_FIRE, _CRASH),
    "posible_altercado": (_FIRE, _CRASH),
}


_ORDINAL_M = r"(?:primer|segundo|tercer|cuarto|quinto|sexto|s[eé]ptimo|octavo|[uú]ltimo)"
_ORDINAL_F = r"(?:primera|segunda|tercera|cuarta|quinta|sexta|s[eé]ptima|octava|[uú]ltima)"
_FRAME_NARRATION = re.compile(
    rf"\bfotogramas?\b|\bframes?\b|\b{_ORDINAL_M}\s+(?:cuadro|fotograma|frame)\b|\b{_ORDINAL_F}\s+(?:imagen|escena)\b"
    rf"|\bimagen\s+\d+\b|\ben la secuencia\b|\bsecuencia de (?:im[aá]genes|fotogramas)\b",
    re.IGNORECASE,
)


def narrates_frames(description: str) -> bool:
    return bool(_FRAME_NARRATION.search(description))


def injection_signals(description: str, trigger_reason: str | None = None) -> list[str]:
    signals = []
    if narrates_frames(description):
        signals.append("narra fotograma por fotograma")
    if any(pattern.search(description) for pattern in _FOREIGN.get(trigger_reason or "", ())):
        signals.append("habla de un hecho de otra categoría que la detectada")
    if len(description.strip()) < MIN_DESCRIPTION_CHARS:
        signals.append("descripción demasiado corta")
    if _CONTRADICTION.search(description):
        signals.append("califica o niega el hecho detectado")
    return signals


def fallback_description(trigger_reason: str | None, camera_label: str) -> DescriptionResult:
    label = CATEGORY_LABELS.get(trigger_reason or "", "evento")
    text = f"Evento de categoría «{label}» detectado en {camera_label}. La descripción automática del video no pasó la validación y se omitió."
    return DescriptionResult(description_nl=text, attributes=EventAttributes())


def drop_foreign_sentences(description: str, trigger_reason: str | None = None) -> str:
    """Quita las oraciones que hablan de un hecho de otra categoría que la detectada (y los campos que queden vacíos).

    Antes toda la descripción se reemplazaba por la neutra; el modelo suele agregar un «humo y fuego» inventado a un choque y se
    perdía el resto, que estaba bien. La categoría la sigue fijando el sistema: lo que afirme otro hecho no se guarda."""
    foreign = _FOREIGN.get(trigger_reason or "", ())
    if not foreign:
        return description
    kept = _SENTENCE.sub(lambda m: "" if any(p.search(m.group(0)) for p in foreign) else m.group(0), description)
    empty_label = re.compile(r"\s*[A-ZÁÉÍÓÚÑ_]{4,}:\s*")
    lines = [line for line in kept.splitlines() if not empty_label.fullmatch(line)]
    text = re.sub(r"[ \t]{2,}", " ", "\n".join(lines))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def describe_without_physical_traits(
    describer: VLMDescriber,
    frames: list[np.ndarray],
    camera_label: str,
    object_classes: list[str],
    trigger_reason: str | None = None,
    max_attempts: int = 3,
) -> DescriptionResult:
    result = _describe_checking_appearance(describer, frames, camera_label, object_classes, trigger_reason, max_attempts)
    if settings.vlm_validate_output:
        cleaned = drop_foreign_sentences(result.description_nl, trigger_reason)
        if cleaned != result.description_nl:
            logger.warning("Se quitaron oraciones de otra categoría: [OMITIDO]")
            result = DescriptionResult(description_nl=cleaned, attributes=result.attributes, confidence=result.confidence)
        signals = injection_signals(result.description_nl, trigger_reason)
        if signals:
            logger.warning("Descripción descartada (%s): [OMITIDO]", ", ".join(signals))
            return fallback_description(trigger_reason, camera_label)
    return result


def _describe_checking_appearance(
    describer: VLMDescriber,
    frames: list[np.ndarray],
    camera_label: str,
    object_classes: list[str],
    trigger_reason: str | None = None,
    max_attempts: int = 3,
) -> DescriptionResult:
    result = describer.describe(frames, camera_label, object_classes, trigger_reason)
    restricted = appearance_restricted(trigger_reason)
    for attempt in range(1, max_attempts):
        report = analyze_description(result.description_nl) if restricted else None
        avoid = list(report.sensitive) if report is not None and report.has_sensitive_traits else []
        if narrates_frames(result.description_nl):
            avoid.append(NARRATION_TERM)
        if not avoid:
            return result
        logger.warning(
            "La descripción debe reescribirse (%s), intento %d de %d", ", ".join(avoid), attempt + 1, max_attempts
        )
        result = describer.describe(frames, camera_label, object_classes, trigger_reason, avoid_terms=avoid)

    remaining = analyze_description(result.description_nl).sensitive if restricted else []
    if remaining:
        logger.warning("La descripción sigue mencionando rasgos físicos o demográficos (%s) y se guarda así", ", ".join(remaining))
    return result


def collapse_repetitions(text: str) -> str:
    """Quita las oraciones repetidas: en un bucle, la decodificación voraz repite una frase hasta agotar los tokens."""
    seen: set[str] = set()
    lines = []
    for line in text.splitlines():
        kept = []
        for sentence in re.findall(r"[^.!?]+[.!?]?\s*", line):
            key = re.sub(r"\W+", " ", re.sub(r"^\s*[A-ZÁÉÍÓÚÑ_]{4,}:\s*", "", sentence)).strip().lower()
            if key in seen:
                continue
            seen.add(key)
            kept.append(sentence)
        if "".join(kept).strip() or not line.strip():
            lines.append("".join(kept).rstrip())
    return "\n".join(lines)


def _trim_unfinished(text: str) -> str:
    if not text or text[-1] in ".!?»\")":
        return text
    end = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    return text[: end + 1] if end > 0 else text


def _parse_response(raw_text: str) -> DescriptionResult:
    match = _RESPONSE_PATTERN.search(raw_text)
    if not match:
        if not re.match(r"\s*DESCRIPCION:", raw_text):
            logger.warning("La respuesta del VLM no siguió el formato esperado, se guarda como texto libre")
        text = _trim_unfinished(re.sub(r"^\s*DESCRIPCION:\s*", "", raw_text.strip()))
        return DescriptionResult(
            description_nl=strip_text_mentions(mask_scene_text(text)) if settings.vlm_mask_scene_text else text,
            attributes=EventAttributes(),
        )

    description_nl, attrs_json = match.groups()
    try:
        attrs_dict = json.loads(attrs_json)
    except json.JSONDecodeError:
        logger.warning("No se pudo parsear el bloque ATRIBUTOS como JSON, se ignora")
        attrs_dict = {}

    direction = attrs_dict.get("direction")
    if isinstance(direction, list):
        attrs_dict["direction"] = ", ".join(str(d) for d in direction) or None

    try:
        attributes = EventAttributes(**attrs_dict)
    except Exception:
        logger.warning("El bloque ATRIBUTOS no coincide con el esquema esperado, se guarda vacío: [OMITIDO]")
        attributes = EventAttributes()

    description_nl = description_nl.strip()
    if settings.vlm_mask_scene_text:
        description_nl, attributes = strip_text_mentions(mask_scene_text(description_nl)), _mask_attributes(attributes)
    return DescriptionResult(description_nl=description_nl, attributes=attributes)


MAX_FRAMES = 8
# Una sola generación por descripción, sin reintentos: las respuestas del 7B rondan los 250 a 300 tokens y 256 cortaba la línea ATRIBUTOS.
MAX_NEW_TOKENS = 320
# Instantes, en segundos respecto del disparo, de los fotogramas que ve el modelo: el disparo llega unos instantes después
# del hecho (la regla acumula detecciones), y las consecuencias se ven en los segundos siguientes.
EVENT_FRAME_OFFSETS = (-3.0, -1.5, -0.5, 0.0, 1.0, 2.5, 5.0, 10.0)


def select_event_frames(
    frames: list[np.ndarray],
    trigger_seconds: float | None,
    duration_seconds: float | None,
    offsets: tuple[float, ...] = EVENT_FRAME_OFFSETS,
) -> list[np.ndarray]:
    count = len(frames)
    if count <= len(offsets) or trigger_seconds is None or duration_seconds is None or duration_seconds <= 0:
        return frames
    if not 0 <= trigger_seconds <= duration_seconds:
        return frames
    per_second = (count - 1) / duration_seconds
    indices = sorted({min(count - 1, max(0, round((trigger_seconds + offset) * per_second))) for offset in offsets})
    if len(indices) < 3:
        return frames
    return [frames[i] for i in indices]


def _sample_frames(frames: list[np.ndarray], max_frames: int | None = None) -> list[np.ndarray]:
    max_frames = max_frames or settings.vlm_max_frames
    if len(frames) <= max_frames:
        return frames
    step = len(frames) / max_frames
    return [frames[int(i * step)] for i in range(max_frames)]


def _to_pil_rgb(frame_bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(frame_bgr[:, :, ::-1])


class Qwen2VLDescriber(VLMDescriber):
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        self.model_name = model_name or settings.vlm_model_name
        self.device = device or settings.vlm_device

        logger.info("Cargando VLM %s (device=%s)", self.model_name, self.device)
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, min_pixels=128 * 28 * 28, max_pixels=settings.vlm_max_pixels
        )
        load_options: dict = {"torch_dtype": torch.float16, "device_map": self.device}
        if settings.vlm_quantization == "bnb4":
            from transformers import BitsAndBytesConfig

            load_options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
            )
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(self.model_name, **load_options)

    def describe(
        self,
        frames: list[np.ndarray],
        camera_label: str,
        object_classes: list[str],
        trigger_reason: str | None = None,
        avoid_terms: list[str] | None = None,
    ) -> DescriptionResult:
        sampled = _sample_frames(frames)
        images = [_to_pil_rgb(f) for f in sampled]

        as_video = settings.vlm_input_mode == "video"
        # Como video, el modelo recibe la secuencia completa y no una pila de imágenes sueltas que tienda a narrar una por una.
        visual = [{"type": "video"}] if as_video else [{"type": "image", "image": image} for image in images]
        messages = [
            {"role": "system", "content": build_system_prompt(trigger_reason)},
            {
                "role": "user",
                "content": [*visual, {"type": "text", "text": build_user_prompt(camera_label, object_classes, avoid_terms)}],
            },
        ]

        text_prompt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        media = {"videos": [images]} if as_video else {"images": images}
        inputs = self._processor(text=[text_prompt], return_tensors="pt", **media).to(self._model.device)

        return _parse_response(collapse_repetitions(self._generate(inputs)))

    def _generate(self, inputs) -> str:
        generated_ids = self._model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
        generated_ids_trimmed = generated_ids[:, inputs["input_ids"].shape[1] :]
        raw_text = self._processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )[0]
        return raw_text


class LLaVADescriber(VLMDescriber):
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        raise NotImplementedError(
            "Backend LLaVA no implementado. Usar Qwen2VLDescriber (default) o "
            "implementar esta clase antes de setear MU_VLM_BACKEND=llava."
        )

    def describe(
        self,
        frames: list[np.ndarray],
        camera_label: str,
        object_classes: list[str],
        trigger_reason: str | None = None,
        avoid_terms: list[str] | None = None,
    ) -> DescriptionResult:
        raise NotImplementedError


def build_describer() -> VLMDescriber:
    if settings.vlm_backend == "qwen2-vl":
        return Qwen2VLDescriber()
    if settings.vlm_backend == "llava":
        return LLaVADescriber()
    raise ValueError(f"vlm_backend desconocido: {settings.vlm_backend}")
