import os

import numpy as np
from openai import OpenAI

from src.core.base_validator import BaseValidator, ValidationResult
from src.core.spec import Sample, Spec


class SemanticSimilarityValidator(BaseValidator):
    """Validates semantic similarity between paraphrase and original text."""

    def __init__(self, config: dict):
        super().__init__(config)
        # TODO: client, embedding model and threshold choices should be parameters
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.threshold = config.get("threshold", 0.85)

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text using OpenAI API."""
        response = self.client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def validate(self, sample: Sample, spec: Spec) -> ValidationResult:
        """Validate semantic similarity between paraphrase and original."""
        original_text = spec.task_input
        paraphrase_text = sample.content

        # Get embeddings
        original_embedding = self._get_embedding(original_text)
        paraphrase_embedding = self._get_embedding(paraphrase_text)

        # Calculate similarity
        similarity_score = self._cosine_similarity(original_embedding, paraphrase_embedding)

        # TODO claude after i review your code: Implement bidirectional entailment check
        # Use NLI model to verify:
        # 1. original entails paraphrase (forward)
        # 2. paraphrase entails original (backward)
        # Both should be true for semantic preservation
        # Consider using model like: "microsoft/deberta-v3-base-mnli"

        return ValidationResult(
            score=similarity_score,
            passed=similarity_score >= self.threshold
        )
