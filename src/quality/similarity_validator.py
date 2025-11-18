from src.core.base_validator import BaseValidator, ValidationResult
from src.core.models import Sample, Spec
from src.core.type_guards import is_ml_augmentation_dict
from src.utils import CohereEmbeddingClient, EmbeddingClient, OpenAIEmbeddingClient


class SimilarityValidator(BaseValidator):
    """Validates semantic similarity between generated and original text."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.threshold = config.get("threshold", 0.85)
        self.embedding_client = self._create_embedding_client(config)

    def is_sample_level(self) -> bool:
        """Return True - this validator operates on individual samples."""
        return True

    def is_batch_level(self) -> bool:
        """Return False - this validator does not operate on batches."""
        return False

    def _create_embedding_client(self, config: dict) -> EmbeddingClient:
        """Factory method to create embedding client based on config."""
        embedding_model_full = config.get("embedding_model", "openai/text-embedding-3-small")
        provider, model = self._parse_model_name(embedding_model_full)

        if provider == "openai":
            return OpenAIEmbeddingClient(model=model)
        elif provider == "cohere":
            return CohereEmbeddingClient(model=model)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    def _parse_model_name(self, model_name: str) -> tuple[str, str]:
        """Parse model name to extract provider and model."""
        if "/" in model_name:
            provider, model = model_name.split("/", 1)
            return provider, model
        return "openai", model_name

    def _extract_original_text(self, spec: Spec) -> str:
        """Extract original text from spec based on domain and input type."""
        # For ML augmentation, extract from dict
        if is_ml_augmentation_dict(spec.task_input):
            return spec.task_input["original_input"]

        # For task_rewrite domain, check examples first
        if spec.domain == "task_rewrite":
            examples = spec.constraints.get("examples", [])
            if examples and isinstance(examples, list) and len(examples) > 0:
                first_example = examples[0]
                if isinstance(first_example, dict):
                    return first_example.get("input", str(spec.task_input))
                else:
                    return str(first_example)
            # No examples - use task_input itself
            return str(spec.task_input)

        # Default: use task_input as string
        return str(spec.task_input)

    def validate(self, sample: Sample, spec: Spec) -> ValidationResult:
        """Validate semantic similarity using embeddings."""
        original_text = self._extract_original_text(spec)
        generated_text = sample.content

        # Compute embeddings
        original_embedding = self.embedding_client.get_embedding(original_text)
        generated_embedding = self.embedding_client.get_embedding(generated_text)

        # Calculate similarity
        similarity_score = self.embedding_client.cosine_similarity(
            original_embedding, generated_embedding
        )

        passed = similarity_score >= self.threshold

        return ValidationResult(
            score=similarity_score,
            passed=passed,
            metadata={"similarity_score": similarity_score},
        )
