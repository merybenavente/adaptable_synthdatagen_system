from typing import Any

from src.core.generator_types import GeneratorType
from src.core.spec import Domain, GenerationPlan, LocalFeedbackState


class Router:
    """Router that produces GenerationPlans based on context, progress, and feedback state."""

    def __init__(self, default_batch_size: int = 5):
        self.default_batch_size = default_batch_size

    def route(
        self,
        context: dict[str, Any],
        state: LocalFeedbackState,
        progress: dict[str, Any],
    ) -> GenerationPlan:
        """Route to generator and produce GenerationPlan."""
        domain_type = context.get("domain_type", "task_rewrite")
        selected_arm = self._select_arm(domain_type)

        # Compute batch size based on remaining samples
        remaining = progress.get("remaining_samples", 0)
        batch_size = min(self.default_batch_size, remaining) if remaining > 0 else self.default_batch_size

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
        """Select generator arm based on domain type."""
        if domain_type == Domain.TASK_REWRITE.value:
            return GeneratorType.NAIVE
        elif domain_type == Domain.QA_PAIRS.value:
            return GeneratorType.NAIVE
        elif domain_type == Domain.CODE_SNIPPETS.value:
            return GeneratorType.NAIVE
        else:
            return GeneratorType.NAIVE

    def log_feedback(
        self, generator: str | GeneratorType, reward: float, context: dict[str, Any]
    ) -> None:
        """Log feedback (no-op for simple router)."""
        pass
