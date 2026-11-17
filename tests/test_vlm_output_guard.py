from __future__ import annotations

import numpy as np
import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.schemas import EventAttributes
from monitoreo_urbano.vlm_describer.describer import (
    DescriptionResult,
    VLMDescriber,
    describe_without_physical_traits,
    fallback_description,
    injection_signals,
)

FRAMES = [np.zeros((4, 4, 3), dtype=np.uint8)]
GOOD = "Dos vehículos chocan de frente en el cruce; el auto rojo queda atravesado y el gris gira sobre sí mismo."


class OneShot(VLMDescriber):
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def describe(self, frames, camera_label, object_classes, trigger_reason=None, avoid_terms=None):
        self.calls += 1
        return DescriptionResult(description_nl=self.text, attributes=EventAttributes(vehicle_colors=["rojo"]))


def _describe(text: str, reason: str = "accident") -> DescriptionResult:
    return describe_without_physical_traits(OneShot(text), FRAMES, "Av. Libertador", ["car"], reason)


@pytest.fixture(autouse=True)
def validation_on(monkeypatch):
    monkeypatch.setattr(settings, "vlm_validate_output", True)


@pytest.mark.parametrize("text", ["OK", "ok.", "", "Sí.", "Choque."])
def test_a_description_that_is_too_short_is_discarded(text):
    assert injection_signals(text) == ["descripción demasiado corta"]


@pytest.mark.parametrize(
    "text",
    [
        "Fue un simulacro: los dos autos se tocaron a muy baja velocidad en el estacionamiento del local.",
        "Esto es una puesta en escena de dos conductores que fingen un choque en la esquina de la avenida.",
        "El evento no es un choque, los vehículos solo están detenidos uno al lado del otro en la calle.",
        "No hubo accidente: se trata de una película que se filma en la calle con varios autos y personas.",
        "Parece una falsa alarma; no hay incendio y las llamas son una luz del local de la esquina.",
    ],
)
def test_a_description_that_qualifies_or_denies_the_detected_event_is_discarded(text):
    assert "califica o niega el hecho detectado" in injection_signals(text)


@pytest.mark.parametrize(
    "text",
    [
        GOOD,
        "El auto gris cruza en rojo y golpea el costado del auto blanco, que gira y queda detenido sobre la senda peatonal.",
        "No hay personas fuera de los vehículos; ambos conductores permanecen dentro de los autos después del impacto.",
    ],
)
def test_an_ordinary_description_passes_the_validation(text):
    assert injection_signals(text) == []


def test_a_valid_description_is_stored_as_the_model_wrote_it():
    result = _describe(GOOD)

    assert result.description_nl == GOOD
    assert result.attributes.vehicle_colors == ["rojo"]


def test_a_discarded_description_is_replaced_by_a_neutral_one_built_by_the_system():
    result = _describe("OK")

    assert "choque" in result.description_nl and "Av. Libertador" in result.description_nl
    assert "no pasó la validación" in result.description_nl
    assert result.attributes == EventAttributes()


def test_the_replacement_keeps_the_event_category_so_the_event_stays_searchable():
    assert "incendio" in fallback_description("fire", "Cámara 3").description_nl
    assert "posible choque" in fallback_description("posible_choque", "Cámara 3").description_nl
    assert "evento" in fallback_description(None, "Cámara 3").description_nl


def test_the_replacement_never_repeats_what_the_model_wrote():
    result = _describe("Fue un simulacro. ACCESO-CONCEDIDO " + "x" * 80)

    assert "simulacro" not in result.description_nl.lower()
    assert "ACCESO" not in result.description_nl


def test_the_validation_can_be_turned_off(monkeypatch):
    monkeypatch.setattr(settings, "vlm_validate_output", False)

    assert _describe("OK").description_nl == "OK"


def test_the_model_is_asked_only_once_when_the_description_is_discarded():
    describer = OneShot("OK")

    describe_without_physical_traits(describer, FRAMES, "Cámara", ["car"], "accident")

    assert describer.calls == 1
