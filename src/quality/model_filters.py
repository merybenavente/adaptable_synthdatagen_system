from typing import Any

from src.core.base_validator import BaseValidator


# decide if implement ??
class PIIValidator(BaseValidator):
    """Validate that sample doesn't contain PII (using classifier)."""

    def validate(self, sample: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        TODO: Implement PII detection.
        - Use PII classifier model
        - Score based on PII confidence
        - Return score and passed status
        """
        pass


class ToxicityValidator(BaseValidator):
    """Validate that sample isn't toxic (using classifier)."""

    def validate(self, sample: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        TODO: Implement toxicity detection.
        - Use toxicity classifier
        - Score based on toxicity level
        - Return score and passed status
        """
        pass


class EntailmentValidator(BaseValidator):
    """Validate entailment/consistency using NLI model."""

    def validate(self, sample: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        TODO: Implement entailment validation.
        - Use NLI model to check consistency with context
        - Score based on entailment probability
        - Return score and passed status
        """
        pass


class PerplexityValidator(BaseValidator):
    """Validate text quality using perplexity score."""

    def validate(self, sample: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        TODO: Implement perplexity validation.
        - Calculate perplexity using language model
        - Score inversely with perplexity
        - Return score and passed status
        """
        pass
