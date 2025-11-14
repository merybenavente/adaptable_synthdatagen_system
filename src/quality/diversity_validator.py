import os

import numpy as np
from openai import OpenAI

from src.core.base_validator import BaseValidator, ValidationResult
from src.core.spec import Sample, Spec


class DiversityValidator(BaseValidator):
    """Validates batch-level diversity (pairwise similarity among paraphrases)."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.embedding_model = config.get("embedding_model", "text-embedding-3-small")
        self.threshold = config.get("threshold", 0.3)

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

    def validate_batch(self, samples: list[Sample], spec: Spec) -> ValidationResult:
        """Validate diversity by calculating average pairwise similarity among paraphrases."""
        if len(samples) < 2:
            return ValidationResult(score=1.0, passed=True)

        # Get embeddings for all samples
        embeddings = [self._get_embedding(sample.content) for sample in samples]

        # Calculate pairwise similarities
        similarities = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = self._cosine_similarity(embeddings[i], embeddings[j])
                similarities.append(sim)

        # Average pairwise similarity (lower is better = more diverse)
        avg_similarity = np.mean(similarities)

        # Diversity score = 1 - avg_similarity (higher is better)
        diversity_score = 1.0 - avg_similarity

        return ValidationResult(
            score=diversity_score,
            passed=diversity_score >= self.threshold
        )
