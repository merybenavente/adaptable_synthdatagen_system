import os
from typing import Literal

import cohere
import numpy as np

InputType = Literal["search_document", "search_query", "classification", "clustering"]
EmbeddingType = Literal["float", "int8", "uint8", "binary", "ubinary"]


class CohereClient:
    """Client for interacting with Cohere APIs."""

    def __init__(
        self,
        model: str = "embed-v4.0",
        embedding_dimension: int = 1024,
    ):
        self.model = model
        self.embedding_dimension = embedding_dimension
        self.client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))

    def embed(
        self,
        texts: list[str],
        input_type: InputType = "search_document",
        embedding_types: list[EmbeddingType] = ["float"],
    ) -> cohere.EmbedResponse:
        """Generate embeddings for a list of texts."""
        return self.client.embed(
            texts=texts,
            model=self.model,
            input_type=input_type,
            output_dimension=self.embedding_dimension,
            embedding_types=embedding_types,
        )

    def embed_query(
        self,
        query: str,
        embedding_types: list[EmbeddingType] = ["float"],
    ) -> cohere.EmbedResponse:
        """Generate embeddings for a search query."""
        return self.embed(
            texts=[query],
            input_type="search_query",
            embedding_types=embedding_types,
        )

    def embed_documents(
        self,
        documents: list[str],
        embedding_types: list[EmbeddingType] = ["float"],
    ) -> cohere.EmbedResponse:
        """Generate embeddings for documents to be stored in a vector database."""
        return self.embed(
            texts=documents,
            input_type="search_document",
            embedding_types=embedding_types,
        )

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Calculate cosine similarity between two embeddings."""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
