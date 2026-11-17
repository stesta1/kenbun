from __future__ import annotations

import json

import httpx
import pytest

from monitoreo_urbano.query_service.llm_client import LLMClient, OpenAICompatibleLLMClient


def _client_with(handler) -> OpenAICompatibleLLMClient:
    client = OpenAICompatibleLLMClient(base_url="http://llm.test/v1", model_name="modelo-x", temperature=0.2)
    client._client = httpx.Client(base_url="http://llm.test/v1", transport=httpx.MockTransport(handler))
    return client


def test_generate_sends_both_prompts_and_returns_the_content():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "hola"}}]})

    client = _client_with(handler)

    assert client.generate("sistema", "usuario") == "hola"
    assert seen["path"].endswith("/chat/completions")
    assert seen["body"]["model"] == "modelo-x"
    assert seen["body"]["temperature"] == 0.2
    assert seen["body"]["messages"] == [
        {"role": "system", "content": "sistema"},
        {"role": "user", "content": "usuario"},
    ]
    assert "response_format" not in seen["body"]
    client.close()


def test_generate_in_json_mode_asks_for_a_json_object():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    _client_with(handler).generate("s", "u", json_mode=True)

    assert seen["body"]["response_format"] == {"type": "json_object"}


def test_generate_propagates_http_errors():
    client = _client_with(lambda request: httpx.Response(500, json={"error": "caído"}))

    with pytest.raises(httpx.HTTPStatusError):
        client.generate("s", "u")


def test_abstract_client_cannot_be_used_directly():
    class Incomplete(LLMClient):
        def generate(self, system_prompt, user_prompt, json_mode=False):
            return super().generate(system_prompt, user_prompt, json_mode)

    with pytest.raises(NotImplementedError):
        Incomplete().generate("s", "u")
