from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from monitoreo_urbano.common.config import settings
from monitoreo_urbano.common.logging import get_logger

logger = get_logger(__name__)


class EmbeddingModel(ABC):
    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        raise NotImplementedError


class SentenceTransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str | None = None, device: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or settings.embedding_model_name
        self.device = device or settings.embedding_device
        logger.info("Cargando modelo de embeddings %s (device=%s)", self.model_name, self.device)
        self._model = SentenceTransformer(self.model_name, device=self.device)

    @property
    def dimension(self) -> int:
        if hasattr(self._model, "get_embedding_dimension"):
            return self._model.get_embedding_dimension()
        return self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> np.ndarray:
        prefixed = [f"passage: {t}" for t in texts]
        vectors = self._model.encode(prefixed, normalize_embeddings=True, convert_to_numpy=True)
        return vectors.astype("float32")

    def embed_query(self, text: str) -> np.ndarray:
        vector = self._model.encode([f"query: {text}"], normalize_embeddings=True, convert_to_numpy=True)
        return vector[0].astype("float32")
