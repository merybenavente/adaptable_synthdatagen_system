from src.core.base_validator import BaseValidator, ValidationResult
from src.core.models import Domain, Sample, Spec
from src.core.type_guards import is_ml_augmentation_dict
from src.utils import DeBERTaClient


class EntailmentValidator(BaseValidator):
    """Validates bidirectional entailment between generated and original text."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.threshold = config.get("threshold", 0.5)
        self.nli_client = DeBERTaClient()

    def is_sample_level(self) -> bool:
        """Return True - this validator operates on individual samples."""
        return True

    def is_batch_level(self) -> bool:
        """Return False - this validator does not operate on batches."""
        return False

    def _extract_original_text(self, spec: Spec) -> str:
        """Extract original text from spec based on domain and input type."""
        # For ML augmentation, extract from dict
        if is_ml_augmentation_dict(spec.task_input):
            return spec.task_input["original_input"]

        # For task_rewrite domain, check examples first
        if spec.domain == Domain.TASK_REWRITE:
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
        """Validate bidirectional entailment using NLI."""
        original_text = self._extract_original_text(spec)
        generated_text = sample.content

        # Forward entailment: original → generated
        forward_scores = self.nli_client.classify(
            premise=original_text, hypothesis=generated_text, return_probabilities=False
        )
        forward_entailment = forward_scores["entailment"]

        # Backward entailment: generated → original
        backward_scores = self.nli_client.classify(
            premise=generated_text, hypothesis=original_text, return_probabilities=False
        )
        backward_entailment = backward_scores["entailment"]

        # Bidirectional entailment score (minimum of both directions)
        bidirectional_entailment = min(forward_entailment, backward_entailment)

        # Both directions should show entailment for semantic preservation
        passed = (
            forward_entailment >= self.threshold
            and backward_entailment >= self.threshold
        )

        return ValidationResult(
            score=bidirectional_entailment,
            passed=passed,
            metadata={
                "forward_entailment": forward_entailment,
                "backward_entailment": backward_entailment,
                "bidirectional_entailment": bidirectional_entailment,
            },
        )
