from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from monitoreo_urbano.event_store.embeddings import EmbeddingModel, SentenceTransformerEmbedding


class FakeSentenceTransformer:

    def __init__(self, model_name: str, device: str) -> None:
        self.model_name = model_name
        self.device = device
        self.encoded: list[list[str]] = []

    def encode(self, texts, normalize_embeddings, convert_to_numpy):
        self.encoded.append(list(texts))
        return np.ones((len(texts), 4), dtype="float64")

    def get_sentence_embedding_dimension(self) -> int:
        return 4


@pytest.fixture
def embedding(monkeypatch) -> SentenceTransformerEmbedding:
    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return SentenceTransformerEmbedding(model_name="modelo-falso", device="cpu")


def test_documents_are_embedded_with_the_passage_prefix(embedding):
    vectors = embedding.embed(["choque en la esquina", "incendio en un local"])

    assert embedding._model.encoded == [["passage: choque en la esquina", "passage: incendio en un local"]]
    assert vectors.shape == (2, 4)
    assert vectors.dtype == np.float32


def test_queries_are_embedded_with_the_query_prefix(embedding):
    vector = embedding.embed_query("choques de ayer")

    assert embedding._model.encoded == [["query: choques de ayer"]]
    assert vector.shape == (4,)
    assert vector.dtype == np.float32


def test_dimension_comes_from_the_loaded_model(embedding):
    assert embedding.dimension == 4


def test_dimension_prefers_the_new_method_name_when_the_library_has_it(embedding):
    embedding._model.get_embedding_dimension = lambda: 8

    assert embedding.dimension == 8


def test_abstract_model_defines_the_interface():
    class Incomplete(EmbeddingModel):
        @property
        def dimension(self) -> int:
            return super().dimension

        def embed(self, texts):
            return super().embed(texts)

    model = Incomplete()
    with pytest.raises(NotImplementedError):
        model.dimension
    with pytest.raises(NotImplementedError):
        model.embed(["x"])
