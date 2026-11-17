from __future__ import annotations

from pathlib import Path

import pytest

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.vlm_describer.prompts import build_system_prompt


@pytest.fixture
def prompts_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(settings, "vlm_prompts_dir", tmp_path)
    (tmp_path / "_preamble.txt").write_text("PREAMBULO", encoding="utf-8")
    (tmp_path / "_output_format.txt").write_text("FORMATO", encoding="utf-8")
    (tmp_path / "_appearance_rules.txt").write_text("REGLAS APARIENCIA", encoding="utf-8")
    (tmp_path / "default.txt").write_text("FOCO GENERICO", encoding="utf-8")
    (tmp_path / "accident.txt").write_text("FOCO CHOQUE", encoding="utf-8")
    (tmp_path / "fire.txt").write_text("FOCO FUEGO", encoding="utf-8")
    return tmp_path


def test_uses_category_specific_focus_when_reason_known(prompts_dir):
    prompt = build_system_prompt("fire")
    assert "PREAMBULO" in prompt
    assert "FOCO FUEGO" in prompt
    assert "FORMATO" in prompt
    assert "FOCO GENERICO" not in prompt


def test_falls_back_to_default_when_reason_unknown(prompts_dir):
    prompt = build_system_prompt("motivo_que_no_existe")
    assert "FOCO GENERICO" in prompt


def test_falls_back_to_default_when_reason_is_none(prompts_dir):
    prompt = build_system_prompt(None)
    assert "FOCO GENERICO" in prompt


def test_preamble_always_comes_before_category_focus_and_format(prompts_dir):
    prompt = build_system_prompt("fire")
    assert prompt.index("PREAMBULO") < prompt.index("FOCO FUEGO") < prompt.index("FORMATO")


def test_raises_if_preamble_file_is_missing(prompts_dir):
    (prompts_dir / "_preamble.txt").unlink()
    with pytest.raises(FileNotFoundError):
        build_system_prompt("fire")


def test_reflects_file_edits_without_caching(prompts_dir):
    first = build_system_prompt("fire")
    (prompts_dir / "fire.txt").write_text("FOCO FUEGO EDITADO", encoding="utf-8")
    second = build_system_prompt("fire")
    assert "EDITADO" not in first
    assert "FOCO FUEGO EDITADO" in second


def test_appearance_rules_apply_to_restricted_categories(prompts_dir):
    prompt = build_system_prompt("fire")
    assert "REGLAS APARIENCIA" in prompt
    assert prompt.index("PREAMBULO") < prompt.index("REGLAS APARIENCIA") < prompt.index("FOCO FUEGO")


def test_appearance_rules_apply_to_the_default_category(prompts_dir):
    assert "REGLAS APARIENCIA" in build_system_prompt(None)


def test_accidents_are_exempt_from_the_appearance_rules(prompts_dir):
    prompt = build_system_prompt("accident")
    assert "FOCO CHOQUE" in prompt
    assert "REGLAS APARIENCIA" not in prompt


def test_the_exempt_categories_come_from_settings(prompts_dir, monkeypatch):
    monkeypatch.setattr(settings, "vlm_appearance_exempt_reasons", ["fire"])
    assert "REGLAS APARIENCIA" not in build_system_prompt("fire")
    assert "REGLAS APARIENCIA" in build_system_prompt("accident")


def test_real_prompts_keep_the_rules_out_of_the_accident_prompt():
    accident = build_system_prompt("accident")
    fight = build_system_prompt("fight")
    assert "PROHIBIDO mencionar género" not in accident
    assert "PROHIBIDO mencionar género" in fight
