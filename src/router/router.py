from typing import Any

from src.core.generator_types import GeneratorType
from src.core.spec import Domain, Spec


class Router:
    """Simple domain-based router with deterministic rules."""

    def __init__(self):
        """Initialize router."""
        pass

    def route(self, spec: Spec) -> GeneratorType:
        """Route request to generator based on domain type."""
        match spec.domain:
            case Domain.TASK_REWRITE:
                return GeneratorType.NAIVE
            case Domain.QA_PAIRS:
                return GeneratorType.NAIVE
            case Domain.CODE_SNIPPETS:
                return GeneratorType.NAIVE
            case _:
                return GeneratorType.NAIVE

    def log_feedback(
        self, generator: str | GeneratorType, reward: float, context: dict[str, Any]
    ) -> None:
        """Log feedback (no-op for simple router)."""
        pass
