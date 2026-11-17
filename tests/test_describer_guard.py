from __future__ import annotations

import numpy as np
import pytest

from monitoreo_urbano.vlm_describer import describer as describer_module
from monitoreo_urbano.vlm_describer.describer import DescriptionResult, VLMDescriber, describe_without_physical_traits
from monitoreo_urbano.vlm_describer.prompts import build_user_prompt


class ScriptedDescriber(VLMDescriber):
    def __init__(self, texts: list[str]) -> None:
        self.texts = list(texts)
        self.calls: list[list[str] | None] = []

    def describe(self, frames, camera_label, object_classes, trigger_reason=None, avoid_terms=None):
        self.calls.append(avoid_terms)
        return DescriptionResult(description_nl=self.texts.pop(0), attributes=None)


FRAMES = [np.zeros((4, 4, 3), dtype=np.uint8)]


@pytest.fixture(autouse=True)
def only_the_appearance_guard(monkeypatch):
    monkeypatch.setattr(describer_module.settings, "vlm_validate_output", False)


def _run(describer: ScriptedDescriber, **kwargs) -> DescriptionResult:
    return describe_without_physical_traits(describer, FRAMES, "Cámara", ["person"], "fight", **kwargs)


def test_a_clean_description_is_returned_without_retrying():
    describer = ScriptedDescriber(["Una persona empuja a otra, que retrocede."])
    result = _run(describer)
    assert result.description_nl.startswith("Una persona empuja")
    assert describer.calls == [None]


def test_retries_once_telling_the_model_which_terms_to_drop():
    describer = ScriptedDescriber(["Dos hombres pelean.", "Dos personas pelean."])
    result = _run(describer)
    assert result.description_nl == "Dos personas pelean."
    assert describer.calls == [None, ["hombres"]]


def test_retries_up_to_the_attempt_limit_and_keeps_the_last_one():
    describer = ScriptedDescriber(["Dos hombres pelean.", "Un hombre y una mujer pelean.", "Una mujer empuja a un hombre."])
    result = _run(describer)
    assert result.description_nl == "Una mujer empuja a un hombre."
    assert describer.calls == [None, ["hombres"], ["hombre", "mujer"]]


def test_a_second_retry_can_fix_what_the_first_did_not():
    describer = ScriptedDescriber(["Dos hombres pelean.", "Un hombre empuja.", "Una persona empuja a otra."])
    result = _run(describer)
    assert result.description_nl == "Una persona empuja a otra."
    assert len(describer.calls) == 3


def test_max_attempts_of_one_never_retries():
    describer = ScriptedDescriber(["Dos hombres pelean."])
    result = _run(describer, max_attempts=1)
    assert result.description_nl == "Dos hombres pelean."
    assert describer.calls == [None]


def test_the_retry_prompt_lists_the_terms_to_avoid():
    prompt = build_user_prompt("Cámara", ["person"], avoid_terms=["hombre", "piel"])
    assert "hombre, piel" in prompt
    assert "sin ninguno de esos términos" in prompt


def test_the_first_prompt_has_no_correction():
    assert "anterior" not in build_user_prompt("Cámara", ["person"])


def test_accidents_are_not_rewritten_even_if_they_mention_people():
    describer = ScriptedDescriber(["Un hombre baja del auto y camina hacia la vereda."])
    result = describe_without_physical_traits(describer, FRAMES, "Cámara", ["car"], "accident")
    assert result.description_nl.startswith("Un hombre baja")
    assert describer.calls == [None]
