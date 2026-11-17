from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from monitoreo_urbano.common.schemas import CameraInfo, ChatRequest, ChatTurn, SearchResultItem
from monitoreo_urbano.query_service.llm_client import LLMClient
from monitoreo_urbano.query_service.rag_pipeline import (
    FALLBACK_ANSWER,
    NOTHING_ANSWER,
    PLAN_SYSTEM_PROMPT,
    RAGPipeline,
    _parse_plan,
)

from .conftest import make_event

BASE = datetime(2026, 10, 6, 12, 0, 0)

LANZA = make_event(
    event_id="lanza", trigger_reason="posible_altercado", description_nl="Una persona con una lanza se pelea con otra.",
    start_ts=BASE, end_ts=BASE + timedelta(seconds=30),
)
FUEGO = make_event(
    event_id="fuego", trigger_reason="fire", description_nl="Un edificio se incendia con humo negro.",
    start_ts=BASE - timedelta(hours=2), end_ts=BASE - timedelta(hours=2) + timedelta(seconds=30),
)
CHOQUE = make_event(
    event_id="choque", trigger_reason="accident", description_nl="Dos autos chocan de frente.",
    start_ts=BASE - timedelta(days=1), end_ts=BASE - timedelta(days=1) + timedelta(seconds=30),
)
ALL = {event.event_id: event for event in (LANZA, FUEGO, CHOQUE)}
CAMERAS = [
    CameraInfo(camera_id="cam-01", label="Esquina", source="rtsp://x"),
    CameraInfo(camera_id="cam-02", label="Plaza", source="rtsp://y"),
]


class FakeStore:
    def __init__(self, found=None) -> None:
        self.found = list(ALL.values()) if found is None else found
        self.searches: list[dict] = []
        self.recents: list[dict] = []

    def list_cameras(self):
        return CAMERAS

    def get_events(self, event_ids, camera_ids=None):
        return [ALL[event_id] for event_id in event_ids if event_id in ALL]

    def hybrid_search(self, query, **kwargs):
        self.searches.append({"query": query, **kwargs})
        excluded = kwargs.get("exclude_event_ids") or frozenset()
        return [SearchResultItem(event=e, combined_score=1.0) for e in self.found if e.event_id not in excluded]

    def recent_events(self, **kwargs):
        self.recents.append(kwargs)
        excluded = kwargs.get("exclude_event_ids") or frozenset()
        return [e for e in self.found if e.event_id not in excluded]


class ScriptedLLM(LLMClient):
    def __init__(self, *replies) -> None:
        self.replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def generate(self, system_prompt, user_prompt, json_mode=False):
        self.prompts.append((system_prompt, user_prompt))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply if isinstance(reply, str) else json.dumps(reply)


def plan(*searches, discard=(), intent="buscar", reply="", rejection=None):
    if rejection is None:
        rejection = "no" if discard else ""
    return {
        "intencion": intent, "respuesta": reply, "busquedas": list(searches),
        "rechazo": rejection, "descartar": list(discard),
    }


def search(query="", categories=(), desde=None, hasta=None, camara=None, tiempo=None):
    if tiempo is None:
        tiempo = "x" if desde or hasta else ""
    return {
        "consulta": query, "categorias": list(categories), "tiempo": tiempo,
        "desde": desde, "hasta": hasta, "camara": camara,
    }


def verdict(text="Hubo un evento.", clips=(1,)):
    return {"respuesta": text, "clips": list(clips)}


def run(llm, store=None, **kwargs):
    store = store or FakeStore()
    return RAGPipeline(store, llm).answer(**kwargs), store


def test_the_interpreted_search_is_what_runs_against_the_store():
    llm = ScriptedLLM(plan(search("incendio fuego llamas", ["fire"])), verdict("Hubo un incendio.", [1]))
    response, store = run(llm, query="Hubo fuego?")
    assert store.searches[0]["query"] == "incendio fuego llamas"
    assert store.searches[0]["categories"] == frozenset({"fire"})
    assert response.answer == "Hubo un incendio."


def test_the_model_decides_which_of_the_candidates_are_shown():
    llm = ScriptedLLM(plan(search("lanza")), verdict(clips=[1]))
    response, _ = run(llm, store=FakeStore([LANZA, CHOQUE, FUEGO]), query="pelea con lanza")
    assert [e.event_id for e in response.events] == ["lanza"]


def test_shown_events_are_ordered_from_the_most_recent_to_the_oldest():
    llm = ScriptedLLM(plan(search("x")), verdict(clips=[1, 2, 3]))
    response, _ = run(llm, store=FakeStore([CHOQUE, LANZA, FUEGO]), query="x")
    assert [e.event_id for e in response.events] == ["lanza", "fuego", "choque"]


def test_clip_numbers_that_do_not_exist_are_ignored():
    llm = ScriptedLLM(plan(search("x")), verdict(clips=[0, 1, 9, "a", 1]))
    response, _ = run(llm, store=FakeStore([LANZA]), query="x")
    assert [e.event_id for e in response.events] == ["lanza"]


def test_a_time_only_question_lists_recent_events_instead_of_searching_text():
    llm = ScriptedLLM(plan(search("", desde="2026-10-06 12:00", hasta="2026-10-06 20:00", tiempo="hoy por la tarde")), verdict(clips=[1]))
    _, store = run(llm, query="que paso hoy por la tarde")
    assert store.searches == []
    assert store.recents[0]["start_ts"] is not None and store.recents[0]["end_ts"] is not None
    assert store.recents[0]["start_ts"] < store.recents[0]["end_ts"]


def test_several_searches_are_merged_without_duplicates():
    llm = ScriptedLLM(plan(search("incendio", ["fire"]), search("fuego")), verdict(clips=[1]))
    _, store = run(llm, store=FakeStore([FUEGO]), query="fuego")
    assert len(store.searches) == 2
    assert "#1 " in llm.prompts[1][1] and "#2 " not in llm.prompts[1][1]


def test_the_operator_scope_is_always_applied():
    llm = ScriptedLLM(plan(search("x", camara="cam-02")), verdict())
    _, store = run(llm, query="x", camera_ids=frozenset({"cam-02"}))
    assert store.searches[0]["camera_ids"] == frozenset({"cam-02"})
    assert store.searches[0]["camera_id"] is None


def test_the_ui_filters_narrow_the_interpreted_time_range():
    llm = ScriptedLLM(plan(search("x", desde="2026-10-01 00:00", hasta="2026-10-31 00:00")), verdict())
    ui_start, ui_end = datetime(2026, 10, 5), datetime(2026, 10, 10)
    _, store = run(llm, query="x", start_ts=ui_start, end_ts=ui_end)
    assert store.searches[0]["start_ts"] == ui_start
    assert store.searches[0]["end_ts"] == ui_end


def test_small_talk_is_answered_without_searching():
    llm = ScriptedLLM(plan(intent="charla", reply="¡Hola! ¿Qué querés saber?"))
    response, store = run(llm, query="hola")
    assert response.answer == "¡Hola! ¿Qué querés saber?"
    assert response.events == [] and store.searches == [] and store.recents == []
    assert len(llm.prompts) == 1


def test_when_nothing_relevant_is_found_the_search_is_replanned_once():
    llm = ScriptedLLM(
        plan(search("incendio", ["fire"], desde="2026-09-06 00:00")),
        verdict("No hay nada.", []),
        plan(search("fuego")),
        verdict("Hubo un incendio.", [1]),
    )
    response, _ = run(llm, query="Hubo fuego el último mes?")
    assert "no encontró nada útil" in llm.prompts[2][1]
    assert response.answer == "Hubo un incendio."
    assert response.events


def test_the_retry_happens_only_once():
    llm = ScriptedLLM(plan(search("a")), verdict("Nada.", []), plan(search("b")))
    response, store = run(llm, store=FakeStore([]), query="x")
    assert len(store.searches) == 2
    assert len(llm.prompts) == 3
    assert response.events == []


def test_with_no_candidates_at_all_the_first_answer_is_kept():
    llm = ScriptedLLM(plan(search("a")), verdict("No encontré nada en ese período.", []), plan(search("b")))
    response, _ = run(llm, store=FakeStore([]), query="x")
    assert response.answer == "No encontré nada en ese período."


def test_an_unparsable_plan_falls_back_to_searching_the_question():
    llm = ScriptedLLM("no es json", verdict())
    _, store = run(llm, query="pelea con lanza")
    assert store.searches[0]["query"] == "pelea con lanza"


def test_a_failing_planner_does_not_break_the_chat():
    llm = ScriptedLLM(RuntimeError("caído"), verdict())
    response, store = run(llm, query="pelea")
    assert store.searches[0]["query"] == "pelea"
    assert response.events


def test_a_failing_writer_still_returns_the_events_found():
    llm = ScriptedLLM(plan(search("x")), RuntimeError("caído"))
    response, _ = run(llm, store=FakeStore([LANZA]), query="x")
    assert response.answer == FALLBACK_ANSWER
    assert [e.event_id for e in response.events] == ["lanza"]


def test_a_failing_writer_with_no_events_says_nothing_was_found():
    llm = ScriptedLLM(plan(search("x")), RuntimeError("caído"), plan(search("x")))
    response, _ = run(llm, store=FakeStore([]), query="x")
    assert response.answer == NOTHING_ANSWER


def test_rejected_events_are_excluded_and_reported():
    history = [ChatTurn(role="user", content="pelea?"), ChatTurn(role="assistant", content="Sí.", event_ids=["lanza"])]
    llm = ScriptedLLM(plan(search("pelea con auto rojo"), discard=[1]), verdict("Nada más.", []), plan(search("x")))
    response, store = run(llm, query="ese no es, había un auto rojo", history=history)
    assert store.searches[0]["exclude_event_ids"] == frozenset({"lanza"})
    assert response.excluded_event_ids == ["lanza"]


def test_exclusions_sent_by_the_client_accumulate_with_new_ones():
    history = [ChatTurn(role="assistant", content="Sí.", event_ids=["fuego"])]
    llm = ScriptedLLM(plan(search("x"), discard=[1]), verdict())
    response, store = run(llm, query="no", history=history, excluded_event_ids=frozenset({"lanza"}))
    assert response.excluded_event_ids == ["fuego", "lanza"]
    assert store.searches[0]["exclude_event_ids"] == frozenset({"fuego", "lanza"})


def test_numbers_that_do_not_match_a_shown_event_are_ignored():
    history = [ChatTurn(role="assistant", content="Sí.", event_ids=["lanza"])]
    llm = ScriptedLLM(plan(search("x"), discard=[0, 2, "z"]), verdict())
    response, _ = run(llm, query="no", history=history)
    assert response.excluded_event_ids == []


def test_the_planner_sees_the_conversation_the_shown_events_the_cameras_and_the_current_time():
    history = [ChatTurn(role="user", content="pelea?"), ChatTurn(role="assistant", content="Sí.", event_ids=["lanza"])]
    llm = ScriptedLLM(plan(search("x")), verdict())
    run(llm, query="ese no es", history=history)
    prompt = llm.prompts[0][1]
    assert "pelea?" in prompt and "una lanza" in prompt and "cam-02: Plaza" in prompt
    assert "Fecha y hora actuales" in prompt


def test_the_writer_sees_the_question_the_search_and_the_events():
    llm = ScriptedLLM(plan(search("lanza", ["fight"])), verdict())
    run(llm, store=FakeStore([LANZA]), query="alguna pelea con lanza?")
    prompt = llm.prompts[1][1]
    assert "alguna pelea con lanza?" in prompt and "una lanza" in prompt and "posible altercado" in prompt


def test_only_the_most_recent_messages_are_used(monkeypatch):
    from monitoreo_urbano.common.config import settings

    monkeypatch.setattr(settings, "chat_history_messages", 2)
    history = [ChatTurn(role="user", content=f"mensaje {i}") for i in range(5)]
    llm = ScriptedLLM(plan(search("x")), verdict())
    run(llm, query="x", history=history)
    assert "mensaje 4" in llm.prompts[0][1] and "mensaje 2" not in llm.prompts[0][1]


def test_the_response_reports_what_was_searched():
    llm = ScriptedLLM(plan(search("incendio", ["fire"], desde="2026-10-06 00:00")), verdict())
    response, _ = run(llm, query="x")
    assert "incendio" in response.search_query and "desde" in response.search_query


def test_plan_parsing_discards_unknown_categories_cameras_and_bad_dates():
    raw = json.dumps(plan(search("x", ["fire", "ufo"], desde="ayer", hasta="2026-10-06 20:00", camara="cam-99")))
    only = _parse_plan(raw, "x", {"cam-01"}, 0).searches[0]
    assert only.categories == frozenset({"fire"})
    assert only.start is None and only.end is not None and only.cameras == frozenset()


def test_plan_times_are_interpreted_as_local_time_and_stored_as_utc():
    parsed = _parse_plan(json.dumps(plan(search("x", desde="2026-10-06 12:00"))), "x", set(), 0)
    expected = datetime(2026, 10, 6, 12, 0).astimezone(timezone.utc).replace(tzinfo=None)
    assert parsed.searches[0].start == expected


def test_the_planner_prompt_teaches_time_expressions_and_synonyms():
    for needle in ("por la tarde", "fuego", "alerta", "descartar", "YYYY-MM-DD HH:MM"):
        assert needle in PLAN_SYSTEM_PROMPT


def test_the_request_accepts_a_conversation():
    request = ChatRequest(
        query="ese no es",
        history=[{"role": "user", "content": "hola"}, {"role": "assistant", "content": "chau", "event_ids": ["a"]}],
        excluded_event_ids=["a"],
    )
    assert request.history[1].event_ids == ["a"]


def test_the_request_limits_the_size_of_the_conversation():
    with pytest.raises(ValidationError):
        ChatRequest(query="x", history=[{"role": "user", "content": "x"}] * 41)
    with pytest.raises(ValidationError):
        ChatRequest(query="x", history=[{"role": "user", "content": "x" * 4001}])
    with pytest.raises(ValidationError):
        ChatRequest(query="x" * 2001)
    with pytest.raises(ValidationError):
        ChatRequest(query="x", history=[{"role": "robot", "content": "x"}])
    with pytest.raises(ValidationError):
        ChatRequest(query="x", excluded_event_ids=["a"] * 201)


def test_fight_and_possible_altercado_are_always_searched_together():
    raw = json.dumps(plan(search("pelea", ["fight"])))
    assert _parse_plan(raw, "x", set(), 0).searches[0].categories == frozenset({"fight", "posible_altercado"})
    raw = json.dumps(plan(search("choque", ["accident"])))
    assert _parse_plan(raw, "x", set(), 0).searches[0].categories == frozenset({"accident", "posible_choque"})


def test_the_planner_gets_precomputed_reference_dates():
    llm = ScriptedLLM(plan(search("x")), verdict())
    run(llm, query="x")
    today = datetime.now().strftime("%Y-%m-%d")
    month_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
    assert f"hoy {today}" in llm.prompts[0][1] and f"hace 30 d\u00edas {month_ago}" in llm.prompts[0][1]


def test_a_time_range_the_operator_never_mentioned_is_ignored():
    llm = ScriptedLLM(plan(search("lanza", desde="2026-10-07 00:00", tiempo="")), verdict())
    _, store = run(llm, query="hubo algo con lanza?")
    assert store.searches[0]["start_ts"] is None


def test_a_time_expression_that_is_not_in_the_conversation_is_ignored():
    llm = ScriptedLLM(plan(search("lanza", desde="2026-10-07 00:00", tiempo="hoy")), verdict())
    _, store = run(llm, query="hubo algo con lanza?")
    assert store.searches[0]["start_ts"] is None


def test_a_time_expression_said_earlier_in_the_conversation_still_counts():
    history = [ChatTurn(role="user", content="Hubo fuego AYER por la noche?"), ChatTurn(role="assistant", content="No.")]
    llm = ScriptedLLM(plan(search("fuego", desde="2026-10-06 20:00", tiempo="ayer por la noche")), verdict())
    _, store = run(llm, query="y choques?", history=history)
    assert store.searches[0]["start_ts"] is not None


def test_small_talk_is_understood_even_if_the_model_names_the_reply_differently():
    raw = {"intencion": "charla", "charla": "De nada!", "busquedas": []}
    response, store = run(ScriptedLLM(raw), query="gracias")
    assert response.answer == "De nada!" and store.searches == []


def test_the_events_shown_to_the_model_carry_the_weekday_to_avoid_date_mistakes():
    llm = ScriptedLLM(plan(search("x")), verdict())
    run(llm, store=FakeStore([LANZA]), query="x")
    assert any(day in llm.prompts[1][1] for day in ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"))


def test_internal_event_numbers_are_removed_from_the_answer():
    llm = ScriptedLLM(plan(search("x")), verdict("Vi que en el evento #1, en la camara Plaza, hubo un choque. El evento 2 tambien.", [1]))
    response, _ = run(llm, query="x")
    assert "#" not in response.answer and "evento 2" not in response.answer
    assert "camara Plaza" in response.answer


def test_the_planner_prompt_includes_examples_with_todays_real_dates():
    llm = ScriptedLLM(plan(search("x")), verdict())
    run(llm, query="x")
    today = datetime.now().strftime("%Y-%m-%d")
    assert "Ejemplos" in llm.prompts[0][0] and f"{today} 12:00" in llm.prompts[0][0]


def test_a_rejection_the_operator_never_said_is_ignored():
    history = [ChatTurn(role="assistant", content="Si.", event_ids=["lanza"])]
    llm = ScriptedLLM(plan(search("x"), discard=[1], rejection="ese no es"), verdict())
    response, store = run(llm, query="hubo una pelea con lanza?", history=history)
    assert response.excluded_event_ids == []
    assert store.searches[0]["exclude_event_ids"] is None


def test_a_discard_without_quoted_rejection_is_ignored():
    history = [ChatTurn(role="assistant", content="Si.", event_ids=["lanza"])]
    llm = ScriptedLLM(plan(search("x"), discard=[1], rejection=""), verdict())
    response, _ = run(llm, query="ese no es", history=history)
    assert response.excluded_event_ids == []
