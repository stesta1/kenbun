from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_SENSITIVE_WORDS = (
    "hombre", "hombres", "mujer", "mujeres", "varon", "varones", "chico", "chica", "chicos", "chicas",
    "nino", "nina", "ninos", "ninas", "senor", "senora", "senores", "senoras", "muchacho", "muchacha",
    "muchachos", "muchachas", "joven", "jovenes", "anciano", "anciana", "ancianos", "adulto", "adulta",
    "adultos", "masculino", "femenino", "barba", "barbudo", "calvo", "cabello", "pelo", "pelos", "rubio",
    "rubia", "morocho", "morocha", "moreno", "morena", "piel", "tez", "gordo", "gorda", "flaco", "flaca",
    "delgado", "delgada", "musculoso", "corpulento", "contextura", "afrodescendiente", "asiatico",
    "asiatica", "latino", "latina", "caucasico", "indigena", "extranjero", "extranjera",
)

_SENSITIVE_PATTERN = re.compile(
    r"\b(?:" + "|".join(_SENSITIVE_WORDS) + r")\b"
    r"|\b\d+\s+anos\b"
    r"|\b(?:de|con)\s+\d+\s+anos\b"
    r"|\bmenores?\s+de\s+edad\b"
    r"|\bpersonas?\s+(?:negr[oa]s?|blanc[oa]s?)\b",
)

_CLOTHING_WORDS = (
    "camisa", "camiseta", "remera", "campera", "buzo", "pantalon", "pantalones", "jean", "jeans", "short",
    "shorts", "zapatilla", "zapatillas", "calzado", "gorra", "gorro", "mochila", "capucha", "vestido",
    "falda", "ropa", "abrigo", "chaleco", "sombrero", "bufanda", "pollera", "campera", "sudadera",
)

_CLOTHING_PATTERN = re.compile(r"\b(?:" + "|".join(_CLOTHING_WORDS) + r")\b")


def _normalize(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


@dataclass(frozen=True)
class AppearanceReport:
    sensitive: list[str] = field(default_factory=list)
    clothing: list[str] = field(default_factory=list)

    @property
    def has_sensitive_traits(self) -> bool:
        return bool(self.sensitive)


def analyze_description(text: str) -> AppearanceReport:
    normalized = _normalize(text)
    return AppearanceReport(
        sensitive=sorted(set(_SENSITIVE_PATTERN.findall(normalized))),
        clothing=sorted(set(_CLOTHING_PATTERN.findall(normalized))),
    )
