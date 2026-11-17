from __future__ import annotations

import numpy as np
import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import EventAttributes
from monitoreo_urbano.vlm_describer.describer import (
    DescriptionResult,
    VLMDescriber,
    describe_without_physical_traits,
    injection_signals,
    narrates_frames,
)
from monitoreo_urbano.vlm_describer.prompts import NARRATION_TERM, build_user_prompt

CONTINUOUS = "Un auto negro que circulaba por el carril derecho chocó contra un camión rojo, que quedó detenido sobre la banquina."


@pytest.mark.parametrize(
    "text",
    [
        "En el primer fotograma se ven varios autos.",
        "En la secuencia de fotogramas se observa un choque.",
        "En el segundo frame el auto negro gira.",
        "En la primera imagen hay tránsito normal y en la segunda un choque.",
        "Imagen 3: el camión queda detenido.",
    ],
)
def test_narration_by_images_is_detected(text):
    assert narrates_frames(text)
    assert "narra fotograma por fotograma" in injection_signals(text, "accident")


@pytest.mark.parametrize(
    "text",
    [
        CONTINUOUS,
        "Primero el auto frena y después choca contra el camión.",
        "Dos autos colisionaron y uno de ellos quedó en el carril izquierdo.",
        "El segundo auto quedó en el último carril, junto al primer camión.",
    ],
)
def test_a_continuous_account_is_not_flagged(text):
    assert not narrates_frames(text)


class ScriptedDescriber(VLMDescriber):
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[list[str] | None] = []

    def describe(self, frames, camera_label, object_classes, trigger_reason=None, avoid_terms=None):
        self.calls.append(avoid_terms)
        return DescriptionResult(description_nl=self.texts.pop(0), attributes=EventAttributes())


def _describe(describer):
    return describe_without_physical_traits(describer, [np.zeros((2, 2, 3), dtype=np.uint8)], "Cámara 1", ["car"], "accident")


def test_a_description_that_narrates_frames_is_requested_again():
    describer = ScriptedDescriber(["En el primer fotograma hay tránsito y en el segundo un choque entre dos autos.", CONTINUOUS])

    result = _describe(describer)

    assert result.description_nl == CONTINUOUS
    assert describer.calls == [None, [NARRATION_TERM]]


def test_the_retry_prompt_asks_for_a_single_continuous_account():
    prompt = build_user_prompt("Cámara 1", ["car"], [NARRATION_TERM])

    assert "imagen por imagen" in prompt and "único relato continuo" in prompt
    assert "rasgos físicos" not in prompt


def test_a_description_that_keeps_narrating_is_replaced_by_the_neutral_one(monkeypatch):
    monkeypatch.setattr(settings, "vlm_validate_output", True)
    narrated = "En el primer fotograma hay tránsito y en el segundo un choque entre dos autos de distinto color."
    describer = ScriptedDescriber([narrated, narrated, narrated])

    result = _describe(describer)

    assert "no pasó la validación" in result.description_nl
    assert len(describer.calls) == 3
