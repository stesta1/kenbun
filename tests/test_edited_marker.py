from __future__ import annotations

from datetime import timedelta

from monitoreo_urbano.query_service.rag_pipeline import EDITED_MARK, _describe, _edited_note

from .conftest import make_event
from .test_chat import BASE, FakeStore, ScriptedLLM, plan, run, search, verdict


def _event(event_id: str, edited: bool):
    return make_event(
        event_id=event_id, trigger_reason="accident", description_nl="Dos autos chocan.", description_edited=edited,
        start_ts=BASE - timedelta(hours=1), end_ts=BASE - timedelta(hours=1) + timedelta(seconds=30),
    )


def test_an_edited_description_is_marked_in_the_listing_the_assistant_reads():
    assert _describe(_event("a", True), 1).endswith(EDITED_MARK)


def test_an_original_description_is_not_marked():
    assert EDITED_MARK not in _describe(_event("a", False), 1)


def test_the_note_counts_only_the_edited_events():
    note = _edited_note([_event("a", True), _event("b", False), _event("c", True)])

    assert "de 2 de los eventos mostrados" in note
    assert note.endswith("fue corregida por un operador.")


def test_there_is_no_note_when_nothing_was_edited():
    assert _edited_note([_event("a", False)]) == ""
    assert _edited_note([]) == ""


def test_the_answer_ends_with_a_fixed_note_when_a_shown_event_was_edited():
    llm = ScriptedLLM(plan(search("choque", ["accident"])), verdict("Hubo un choque.", [1]))

    response, _ = run(llm, FakeStore([_event("editado", True)]), query="¿Hubo choques?")

    assert response.answer.startswith("Hubo un choque.")
    assert response.answer.endswith("Nota: la descripción de 1 de los eventos mostrados fue corregida por un operador.")
    assert response.events[0].description_edited is True


def test_the_note_is_written_by_the_system_even_if_the_model_does_not_mention_it():
    llm = ScriptedLLM(plan(search("choque", ["accident"])), verdict("Sí, hubo un choque ayer.", [1]))

    response, _ = run(llm, FakeStore([_event("editado", True)]), query="¿Hubo choques?")

    assert "corregida por un operador" in response.answer


def test_the_answer_has_no_note_when_the_shown_events_are_original():
    llm = ScriptedLLM(plan(search("choque", ["accident"])), verdict("Hubo un choque.", [1]))

    response, _ = run(llm, FakeStore([_event("original", False)]), query="¿Hubo choques?")

    assert response.answer == "Hubo un choque."


def test_an_edited_event_the_model_did_not_choose_does_not_add_a_note():
    llm = ScriptedLLM(plan(search("choque", ["accident"])), verdict("Hubo un choque.", [1]))
    shown = _event("original", False)
    hidden = _event("editado", True)

    response, _ = run(llm, FakeStore([shown, hidden]), query="¿Hubo choques?")

    assert [event.event_id for event in response.events] == ["original"]
    assert "corregida" not in response.answer
