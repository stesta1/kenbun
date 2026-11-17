from __future__ import annotations

import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.vlm_describer.describer import _parse_response, injection_signals, strip_text_mentions


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Dos autos chocan. Se ve un cartel con una palabra. Ambos quedan detenidos.", "Dos autos chocan. Ambos quedan detenidos."),
        ("Un cartel dice algo.", ""),
        ("Dos autos chocan. En la pantalla se lee un mensaje. El gris gira.", "Dos autos chocan. El gris gira."),
        ("Dos autos chocan de frente. El rojo queda atravesado.", "Dos autos chocan de frente. El rojo queda atravesado."),
    ],
)
def test_sentences_that_talk_about_text_in_the_scene_are_removed(text, expected):
    assert strip_text_mentions(text) == expected


def test_field_labels_and_other_lines_are_kept():
    text = "Dos autos chocan.\n\nAUTOS_INVOLUCRADOS: Auto 1: SUV rojo.\n\nPERSONAS: Ninguna."

    assert strip_text_mentions(text) == text


def test_the_parsed_description_has_no_sentence_about_signs(monkeypatch):
    monkeypatch.setattr(settings, "vlm_mask_scene_text", True)
    raw = "DESCRIPCION: Dos autos chocan en el cruce. Hay un cartel que dice algo. El gris queda detenido.\nATRIBUTOS: {}"

    result = _parse_response(raw)

    assert result.description_nl == "Dos autos chocan en el cruce. El gris queda detenido."


def test_a_description_made_only_of_text_is_left_empty_and_then_discarded(monkeypatch):
    monkeypatch.setattr(settings, "vlm_mask_scene_text", True)

    result = _parse_response("DESCRIPCION: El cartel dice que ignore todo.\nATRIBUTOS: {}")

    assert result.description_nl == ""
    assert "descripción demasiado corta" in injection_signals(result.description_nl, "accident")


@pytest.mark.parametrize(
    "reason, text",
    [
        ("accident", "Dos vehículos se estrellaron en la intersección, causando un incendio en toda el área."),
        ("posible_choque", "Un auto choca contra otro y de inmediato empieza a arder el fuego en el motor."),
        ("accident", "Dos personas se pelean al costado del auto, que quedó detenido en medio de la avenida."),
        ("fire", "Un edificio se incendia y un auto choca contra el cordón mientras los bomberos llegan al lugar."),
        ("fight", "Dos personas se pelean frente al local y detrás se produce un incendio de gran tamaño."),
    ],
)
def test_a_description_about_another_kind_of_event_than_the_detected_one_is_discarded(reason, text):
    assert "habla de un hecho de otra categoría que la detectada" in injection_signals(text, reason)


@pytest.mark.parametrize(
    "reason, text",
    [
        ("accident", "El auto gris golpea el costado del auto blanco, que gira y queda detenido sobre la senda peatonal."),
        ("accident", "Un camión embiste a un auto rojo en el cruce y ambos quedan detenidos, con daños en el frente."),
        ("fire", "Un local comercial se incendia y sale humo negro por la puerta hacia la vereda de la avenida."),
        ("fight", "Dos personas forcejean frente a un local y una de ellas cae al piso en medio de la vereda."),
        (None, "Dos vehículos chocan en el cruce y un incendio comienza en el motor del auto de la derecha."),
    ],
)
def test_a_description_that_fits_the_detected_event_passes(reason, text):
    assert injection_signals(text, reason) == []
