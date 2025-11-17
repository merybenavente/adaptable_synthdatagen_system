"""Adaptation policies for adjusting generation parameters based on feedback."""

from src.core.spec import BatchMetrics, LocalFeedbackState


class AdaptationPolicy:
    """Base class for adaptation policies."""

    def adapt_exploration(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> float:
        """Adapt exploration rate based on batch metrics."""
        raise NotImplementedError


class DefaultAdaptationPolicy(AdaptationPolicy):
    """Default policy: decay exploration over time."""

    def __init__(self, decay_factor: float = 0.95):
        self.decay_factor = decay_factor

    def adapt_exploration(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> float:
        """Adapt exploration: decay over time."""
        return max(0.01, state.exploration_rate * self.decay_factor)


class StaticAdaptationPolicy(AdaptationPolicy):
    """Static policy: no adaptation."""

    def adapt_exploration(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> float:
        """Keep exploration rate constant."""
        return state.exploration_rate
