from src.core.base_validator import BaseValidator, ValidationResult
from src.core.models import Sample, Spec
from src.core.type_guards import is_ml_augmentation_dict
from src.utils import (
    CohereEmbeddingClient,
    DeBERTaClient,
    EmbeddingClient,
    OpenAIEmbeddingClient,
)


class SemanticSimilarityValidator(BaseValidator):
    """Validates semantic similarity between paraphrase and original text."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.threshold = config.get("threshold", 0.85)
        self.entailment_threshold = config.get("entailment_threshold", 0.5)
        self.embedding_client = self._create_embedding_client(config)
        self.nli_client = DeBERTaClient()

    def _create_embedding_client(self, config: dict) -> EmbeddingClient:
        """Factory method to create embedding client based on config."""
        embedding_model_full = config.get(
            "embedding_model", "openai/text-embedding-3-small"
        )
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

    def _check_bidirectional_entailment(
        self, text1: str, text2: str
    ) -> tuple[bool, float]:
        """Check bidirectional entailment between two texts using NLI."""
        forward_scores = self.nli_client.classify(
            premise=text1, hypothesis=text2, return_probabilities=False
        )
        backward_scores = self.nli_client.classify(
            premise=text2, hypothesis=text1, return_probabilities=False
        )

        forward_entailment = forward_scores["entailment"]
        backward_entailment = backward_scores["entailment"]

        # Both directions should show entailment for semantic preservation
        bidirectional_entailment = min(forward_entailment, backward_entailment)
        passed = (
            forward_entailment >= self.entailment_threshold
            and backward_entailment >= self.entailment_threshold
        )

        return passed, bidirectional_entailment

    def validate(self, sample: Sample, spec: Spec) -> ValidationResult:
        """Validate semantic similarity and bidirectional entailment."""
        # Extract original text from task_input using type guard
        if is_ml_augmentation_dict(spec.task_input):
            original_text = spec.task_input["original_input"]
        else:
            original_text = str(spec.task_input)

        paraphrase_text = sample.content

        # Semantic similarity check
        original_embedding = self.embedding_client.get_embedding(original_text)
        paraphrase_embedding = self.embedding_client.get_embedding(paraphrase_text)
        similarity_score = self.embedding_client.cosine_similarity(
            original_embedding, paraphrase_embedding
        )

        # Bidirectional entailment check
        entailment_passed, entailment_score = self._check_bidirectional_entailment(
            original_text, paraphrase_text
        )

        # Combined validation: both similarity and entailment must pass
        passed = (similarity_score >= self.threshold) and entailment_passed

        return ValidationResult(
            score=similarity_score,
            passed=passed,
            metadata={
                "similarity_score": similarity_score,
                "entailment_score": entailment_score,
                "entailment_passed": entailment_passed,
            },
        )
