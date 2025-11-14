import os
from abc import ABC, abstractmethod

import cohere
import numpy as np
from openai import OpenAI


class EmbeddingClient(ABC):
    """Base class for embedding clients."""

    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text."""
        pass

    @staticmethod
    def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


class OpenAIEmbeddingClient(EmbeddingClient):
    """OpenAI embedding client."""

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text using OpenAI."""
        response = self.client.embeddings.create(input=text, model=self.model)
        return response.data[0].embedding


class CohereEmbeddingClient(EmbeddingClient):
    """Cohere embedding client."""

    def __init__(self, model: str = "embed-english-v3.0"):
        self.model = model
        self.client = cohere.Client(api_key=os.getenv("COHERE_API_KEY"))

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text using Cohere."""
        response = self.client.embed(
            texts=[text], model=self.model, input_type="search_document"
        )
        return response.embeddings[0]
