from typing import Any, Dict

from src.core.spec import Spec


class ContextExtractor:
    """
    Extract routing context from generation request.

    MVP: Only reads directly from Spec without inference.
    Future: Will infer complexity, KB requirements, etc.
    """

    # Define required features for routing decisions
    REQUIRED_FEATURES = {
        "domain_type": str,
        "output_format": str,
        "num_samples": int,
    }

    def extract(self, spec: Spec) -> Dict[str, Any]:
        """
        Extract routing context from Spec.

        Currently only reads explicit fields from spec.

        Args:
            spec: Generation request specification

        Returns:
            Dictionary of routing context features
        """
        context = {
            "domain_type": spec.domain.value,
            "output_format": spec.output_format,
            "num_samples": spec.num_samples,
        }

        # TODO: Add inference for missing features
        # TODO: Extract complexity_level from constraints/input
        # TODO: Infer has_knowledge_base from constraints or routing_hints
        # TODO: Add constraint_count as feature

        # Validate all required features are present
        self._validate_context(context)

        return context

    def _validate_context(self, context: Dict[str, Any]) -> None:
        """
        Ensure all required features are present with correct types.

        Args:
            context: Extracted context dictionary

        Raises:
            ValueError: If required feature is missing
            TypeError: If feature has incorrect type
        """
        for feature, expected_type in self.REQUIRED_FEATURES.items():
            if feature not in context:
                raise ValueError(f"Missing required context feature: {feature}")
            if not isinstance(context[feature], expected_type):
                raise TypeError(
                    f"Feature '{feature}' has type {type(context[feature])}, "
                    f"expected {expected_type}"
                )
