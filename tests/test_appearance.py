from __future__ import annotations

import pytest

from monitoreo_urbano.vlm_describer.appearance import analyze_description


@pytest.mark.parametrize(
    "text",
    [
        "Un hombre golpea a otra persona frente al local.",
        "Una mujer corre hacia la esquina.",
        "Dos jóvenes forcejean en la vereda.",
        "Un señor de 40 años se aleja caminando.",
        "La persona tiene el pelo rubio y barba.",
        "Un niño cruza la calle.",
        "Un chico de contextura grande empuja a otro.",
        "Persona de piel oscura sale corriendo.",
        "Una persona negra se acerca al auto.",
        "Son menores de edad.",
    ],
)
def test_detects_sensitive_traits(text):
    assert analyze_description(text).has_sensitive_traits


@pytest.mark.parametrize(
    "text",
    [
        "Dos personas forcejean junto a un auto estacionado.",
        "Una persona empuja a otra y se aleja corriendo hacia la derecha.",
        "Un auto blanco choca de frente contra un SUV gris.",
        "Hay fuego en el techo de un vehículo, con llamas y humo.",
        "Una camisa negra y un auto negro.",
        "Ocurrió una pelea con contacto físico directo.",
        "El peatón cruza con la señal en rojo.",
        "No se bajó nadie de los vehículos.",
    ],
)
def test_does_not_flag_actions_and_vehicles(text):
    assert not analyze_description(text).has_sensitive_traits


def test_is_insensitive_to_accents_and_case():
    assert analyze_description("UN VARÓN SE ALEJA").has_sensitive_traits
    assert analyze_description("un varon se aleja").has_sensitive_traits


def test_reports_clothing_separately_from_sensitive_traits():
    report = analyze_description("Una persona con campera roja y mochila empuja a otra.")
    assert report.clothing == ["campera", "mochila"]
    assert not report.has_sensitive_traits


def test_reports_which_terms_were_found():
    report = analyze_description("Un hombre y una mujer discuten.")
    assert report.sensitive == ["hombre", "mujer"]
