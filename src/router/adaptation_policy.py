"""Adaptation policies for adjusting generation parameters based on feedback."""

from src.core.spec import BatchMetrics, LocalFeedbackState


class AdaptationPolicy:
    """Base class for adaptation policies."""

    def adapt_temperature(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> float:
        """Adapt temperature based on batch metrics."""
        raise NotImplementedError

    def adapt_exploration(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> float:
        """Adapt exploration rate based on batch metrics."""
        raise NotImplementedError


class DefaultAdaptationPolicy(AdaptationPolicy):
    """Default policy: adapt temperature based on quality, decay exploration."""

    def __init__(
        self,
        temperature_adaptation: bool = True,
        exploration_decay: bool = True,
        temp_step: float = 0.05,
        decay_factor: float = 0.95,
    ):
        self.temperature_adaptation = temperature_adaptation
        self.exploration_decay = exploration_decay
        self.temp_step = temp_step
        self.decay_factor = decay_factor

    def adapt_temperature(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> float:
        """Adapt temperature: decrease if low quality, increase if high quality."""
        if not self.temperature_adaptation or metrics.mean_quality is None:
            return state.current_temperature

        current_temp = state.current_temperature
        quality = metrics.mean_quality
        pass_rate = metrics.pass_rate

        # Thresholds
        LOW_QUALITY = 0.6
        HIGH_QUALITY = 0.8
        LOW_PASS_RATE = 0.5

        # Adapt temperature
        if pass_rate < LOW_PASS_RATE or quality < LOW_QUALITY:
            new_temp = current_temp - self.temp_step
        elif quality > HIGH_QUALITY and pass_rate > 0.8:
            new_temp = current_temp + self.temp_step * 0.5
        else:
            new_temp = current_temp

        # Clamp to reasonable range
        return max(0.3, min(1.2, new_temp))

    def adapt_exploration(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> float:
        """Adapt exploration: decay over time."""
        if not self.exploration_decay:
            return state.exploration_rate

        return max(0.01, state.exploration_rate * self.decay_factor)


class StaticAdaptationPolicy(AdaptationPolicy):
    """Static policy: no adaptation."""

    def adapt_temperature(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> float:
        """Keep temperature constant."""
        return state.current_temperature

    def adapt_exploration(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> float:
        """Keep exploration rate constant."""
        return state.exploration_rate
