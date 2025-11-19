import os

import cohere
import numpy as np
from openai import OpenAI

from src.core.base_validator import BaseValidator, ValidationResult
from src.core.models import Sample, Spec


# TODO: Compute diversity within current batch and already accepted - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/4
class DiversityValidator(BaseValidator):
    """Validates batch-level diversity (pairwise similarity among paraphrases)."""

    def __init__(self, config: dict):
        super().__init__(config)
        embedding_model_full = config.get("embedding_model", "openai/text-embedding-3-small")
        self.provider, self.embedding_model = self._parse_model_name(embedding_model_full)
        self.threshold = config.get("threshold", 0.3)
        self.client = self._initialize_client()

    def is_sample_level(self) -> bool:
        """Return False - this validator does not operate on individual samples."""
        return False

    def is_batch_level(self) -> bool:
        """Return True - this validator operates on batches."""
        return True

    def _parse_model_name(self, model_name: str) -> tuple[str, str]:
        """Parse model name to extract provider and model."""
        if "/" in model_name:
            provider, model = model_name.split("/", 1)
            return provider, model
        # Default to OpenAI if no provider specified
        return "openai", model_name

    def _initialize_client(self):
        """Initialize embedding client based on provider."""
        if self.provider == "openai":
            return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.provider == "cohere":
            return cohere.Client(api_key=os.getenv("COHERE_API_KEY"))
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider}")

    def _get_embedding(self, text: str) -> list[float]:
        """Get embedding vector for text using configured provider."""
        if self.provider == "openai":
            response = self.client.embeddings.create(input=text, model=self.embedding_model)
            return response.data[0].embedding
        elif self.provider == "cohere":
            response = self.client.embed(
                texts=[text], model=self.embedding_model, input_type="search_document"
            )
            return response.embeddings[0]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        v1 = np.array(vec1)
        v2 = np.array(vec2)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    def _content_to_text(self, content: str | dict) -> str:
        """Convert sample content to text for embedding."""
        if isinstance(content, str):
            return content
        if isinstance(content, dict):
            # Serialize dict to JSON string for embedding
            import json
            return json.dumps(content, ensure_ascii=False)
        # Fallback for other types
        return str(content)

    def validate_batch(self, samples: list[Sample], spec: Spec) -> ValidationResult:
        """Validate diversity by calculating average pairwise similarity among paraphrases."""
        if len(samples) < 2:
            return ValidationResult(score=1.0, passed=True)

        # Get embeddings for all samples
        embeddings = [
            self._get_embedding(self._content_to_text(sample.content)) for sample in samples
        ]

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

        return ValidationResult(score=diversity_score, passed=diversity_score >= self.threshold)
