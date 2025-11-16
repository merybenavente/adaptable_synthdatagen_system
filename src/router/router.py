from typing import Any

from src.core.generator_types import GeneratorType
from src.core.spec import Domain, GenerationPlan, LocalFeedbackState, Spec
from src.router.context_extractor import ContextExtractor


class Router:
    """Router that produces GenerationPlans based on Context + LocalFeedbackState."""

    def __init__(self, default_batch_size: int = 5):
        """Initialize router."""
        self.default_batch_size = default_batch_size
        self.context_extractor = ContextExtractor()

    def plan_next_batch(
        self,
        spec: Spec,
        state: LocalFeedbackState,
    ) -> GenerationPlan:
        """
        Generate a GenerationPlan for the next batch.

        Router reads current feedback state (temperature, etc.) and produces
        next batch configuration. Does not look at samples.

        Args:
            spec: Input specification
            state: Current LocalFeedbackState

        Returns:
            GenerationPlan describing batch configuration
        """
        # Extract context from spec
        context = self.context_extractor.extract(spec)

        # Select generator arm based on domain
        selected_arm = self._select_arm(spec.domain)

        # Determine batch size based on remaining samples
        remaining = spec.num_samples - state.generated_so_far
        batch_size = min(self.default_batch_size, remaining)

        # Use current temperature from feedback state
        parameters = {
            "temperature": state.current_temperature,
            "domain": spec.domain.value,
        }

        # Add constraints from spec
        if spec.constraints:
            parameters.update(spec.constraints)

        return GenerationPlan(
            batch_size=batch_size,
            generator_arm=selected_arm,
            parameters=parameters,
            iteration=state.iteration,
        )

    def _select_arm(self, domain: Domain) -> GeneratorType:
        """Select generator arm based on domain."""
        match domain:
            case Domain.TASK_REWRITE:
                return GeneratorType.NAIVE
            case Domain.QA_PAIRS:
                return GeneratorType.NAIVE
            case Domain.CODE_SNIPPETS:
                return GeneratorType.NAIVE
            case _:
                return GeneratorType.NAIVE

    # Legacy method for backwards compatibility
    def route(self, spec: Spec) -> GeneratorType:
        """Route request to generator based on domain type."""
        return self._select_arm(spec.domain)

    def log_feedback(
        self, generator: str | GeneratorType, reward: float, context: dict[str, Any]
    ) -> None:
        """Log feedback (no-op for simple router)."""
        pass
