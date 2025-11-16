from typing import Any

from src.core.spec import Domain, Spec


class Router:
    """Simple domain-based router with deterministic rules."""

    def __init__(self):
        """Initialize router."""
        pass

    def route(self, spec: Spec) -> str:
        """Route request to generator based on domain type."""
        match spec.domain:
            case Domain.TASK_REWRITE:
                return "naive"
            case Domain.QA_PAIRS:
                return "naive"
            case Domain.CODE_SNIPPETS:
                return "naive"
            case _:
                return "naive"

    def log_feedback(
        self, generator: str, reward: float, context: dict[str, Any]
    ) -> None:
        """Log feedback (no-op for simple router)."""
        pass
