from typing import Any

from src.core.generator_types import GeneratorType
from src.core.spec import Domain, GenerationPlan, LocalFeedbackState


class Router:
    """Router that produces GenerationPlans based on Context + LocalFeedbackState."""

    def __init__(self, default_batch_size: int = 5):
        """Initialize router."""
        self.default_batch_size = default_batch_size

    def plan_next_batch(
        self,
        context: dict[str, Any],
        state: LocalFeedbackState,
    ) -> GenerationPlan:
        """
        Generate a GenerationPlan for the next batch.

        Router reads context and current feedback state, then produces
        next batch configuration. Does not look at samples or spec directly.

        Args:
            context: Extracted context features (domain_type, num_samples, etc.)
            state: Current LocalFeedbackState

        Returns:
            GenerationPlan describing batch configuration
        """
        # Select generator arm based on context
        domain_type = context.get("domain_type", "task_rewrite")
        selected_arm = self._select_arm(domain_type)

        # Use default batch size (Pipeline will cap if needed)
        batch_size = self.default_batch_size

        # Use current temperature from feedback state
        parameters = {
            "temperature": state.current_temperature,
            "domain": domain_type,
        }

        return GenerationPlan(
            batch_size=batch_size,
            generator_arm=selected_arm,
            parameters=parameters,
            iteration=state.iteration,
        )

    def _select_arm(self, domain_type: str) -> GeneratorType:
        """Select generator arm based on domain type from context."""
        # Simple routing based on domain
        if domain_type == Domain.TASK_REWRITE.value:
            return GeneratorType.NAIVE
        elif domain_type == Domain.QA_PAIRS.value:
            return GeneratorType.NAIVE
        elif domain_type == Domain.CODE_SNIPPETS.value:
            return GeneratorType.NAIVE
        else:
            return GeneratorType.NAIVE

    # Legacy method for backwards compatibility
    def route(self, spec) -> GeneratorType:
        """Route request to generator based on domain type (legacy)."""
        domain_type = spec.domain.value if hasattr(spec, 'domain') else "task_rewrite"
        return self._select_arm(domain_type)

    def log_feedback(
        self, generator: str | GeneratorType, reward: float, context: dict[str, Any]
    ) -> None:
        """Log feedback (no-op for simple router)."""
        pass
