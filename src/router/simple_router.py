from typing import Dict, Any

from src.core.spec import Spec, Domain


class SimpleRouter:
    """
    Simple domain-based router (V0 - MVP).

    Routes requests to generators based solely on domain type.
    No learning, no context inference - just deterministic routing rules.
    """

    def __init__(self):
        """Initialize simple router."""
        pass

    def route(self, spec: Spec) -> str:
        """
        Route request to generator based on domain.

        Args:
            spec: Generation request specification

        Returns:
            Generator name to use
        """
        match spec.domain:
            case Domain.TASK_REWRITE:
                return "naive"
            case Domain.QA_PAIRS:
                return "naive"
            case Domain.CODE_SNIPPETS:
                return "naive"
            case _:
                # Fallback for any future domains
                return "naive"

    def log_feedback(
        self, generator: str, reward: float, context: Dict[str, Any]
    ) -> None:
        """
        Log feedback (no-op for simple router).

        Simple router doesn't learn, but we keep this method
        for interface compatibility with learning routers.

        Args:
            generator: Generator that was used
            reward: Quality score or reward signal
            context: Request context features
        """
        # No-op: simple router doesn't learn
        pass
