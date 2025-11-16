"""
Feedback Engine for adaptive generation pipeline.

Computes metrics from generated batches and updates LocalFeedbackState
to enable Router to make adaptive decisions.
"""

from typing import Any

import numpy as np

from src.core.spec import BatchMetrics, GenerationPlan, LocalFeedbackState, Sample


class FeedbackEngine:
    """
    Feedback Engine that computes metrics and updates LocalFeedbackState.

    The Feedback Engine does NOT trigger generation or decide batch count.
    It only:
    1. Computes metrics from generated samples
    2. Updates LocalFeedbackState based on those metrics
    """

    def __init__(
        self,
        temperature_adaptation: bool = True,
        exploration_decay: bool = True,
        max_history_length: int = 10,
    ):
        """
        Initialize FeedbackEngine.

        Args:
            temperature_adaptation: Whether to adapt temperature based on quality
            exploration_decay: Whether to decay exploration rate over time
            max_history_length: Maximum number of recent batches to keep in history
        """
        self.temperature_adaptation = temperature_adaptation
        self.exploration_decay = exploration_decay
        self.max_history_length = max_history_length

    def compute_batch_metrics(
        self,
        samples: list[Sample],
        total_generated: int,
    ) -> BatchMetrics:
        """
        Compute metrics for a batch of samples.

        Args:
            samples: List of generated samples (already scored)
            total_generated: Total number of samples attempted in this batch

        Returns:
            BatchMetrics with computed statistics
        """
        if not samples:
            # Empty batch - all samples filtered out
            return BatchMetrics(
                mean_similarity=None,
                diversity_score=None,
                mean_quality=None,
                pass_rate=0.0,
                num_samples=0,
            )

        # Extract quality scores
        similarity_scores = []
        diversity_scores = []
        all_quality_scores = []

        for sample in samples:
            if "similarity" in sample.quality_scores:
                similarity_scores.append(sample.quality_scores["similarity"])
            if "diversity" in sample.quality_scores:
                diversity_scores.append(sample.quality_scores["diversity"])

            # Collect all quality scores for mean quality
            all_quality_scores.extend(sample.quality_scores.values())

        # Compute metrics
        mean_similarity = float(np.mean(similarity_scores)) if similarity_scores else None
        diversity_score = float(np.mean(diversity_scores)) if diversity_scores else None
        mean_quality = float(np.mean(all_quality_scores)) if all_quality_scores else None
        pass_rate = len(samples) / total_generated if total_generated > 0 else 0.0

        return BatchMetrics(
            mean_similarity=mean_similarity,
            diversity_score=diversity_score,
            mean_quality=mean_quality,
            pass_rate=pass_rate,
            num_samples=len(samples),
        )

    def update_feedback_state(
        self,
        state: LocalFeedbackState,
        plan: GenerationPlan,
        batch_metrics: BatchMetrics,
        samples: list[Sample],
    ) -> LocalFeedbackState:
        """
        Update LocalFeedbackState based on batch results.

        This is the core feedback loop logic:
        - Updates arm counts and rewards
        - Adapts temperature based on quality
        - Decays exploration rate
        - Tracks recent metrics

        Args:
            state: Current LocalFeedbackState
            plan: GenerationPlan that was executed
            batch_metrics: Computed metrics for the batch
            samples: Generated samples

        Returns:
            Updated LocalFeedbackState
        """
        # Update iteration and generated count
        state.iteration += 1
        state.generated_so_far += batch_metrics.num_samples

        # Update arm statistics
        arm_name = str(plan.generator_arm)
        state.arm_counts[arm_name] = state.arm_counts.get(arm_name, 0) + 1

        # Use mean_quality as reward (or pass_rate if quality not available)
        reward = batch_metrics.mean_quality if batch_metrics.mean_quality is not None else batch_metrics.pass_rate

        if arm_name not in state.arm_rewards:
            state.arm_rewards[arm_name] = []
        state.arm_rewards[arm_name].append(reward)

        # Add to recent metrics history (keep only last N)
        state.recent_metrics.append(batch_metrics)
        if len(state.recent_metrics) > self.max_history_length:
            state.recent_metrics = state.recent_metrics[-self.max_history_length:]

        # Adaptive temperature adjustment
        if self.temperature_adaptation and batch_metrics.mean_quality is not None:
            state.current_temperature = self._adapt_temperature(
                current_temp=state.current_temperature,
                quality=batch_metrics.mean_quality,
                pass_rate=batch_metrics.pass_rate,
            )

        # Exploration rate decay
        if self.exploration_decay:
            # Linear decay: reduce exploration as we generate more samples
            decay_factor = 0.95
            state.exploration_rate = max(0.01, state.exploration_rate * decay_factor)

        return state

    def _adapt_temperature(
        self,
        current_temp: float,
        quality: float,
        pass_rate: float,
    ) -> float:
        """
        Adapt temperature based on quality and pass rate.

        Strategy:
        - If quality is low or pass rate is low, decrease temperature (more conservative)
        - If quality is high and pass rate is high, slightly increase temperature (more diversity)

        Args:
            current_temp: Current temperature
            quality: Mean quality score [0, 1]
            pass_rate: Fraction of samples that passed validation [0, 1]

        Returns:
            Adjusted temperature
        """
        # Define thresholds
        LOW_QUALITY_THRESHOLD = 0.6
        HIGH_QUALITY_THRESHOLD = 0.8
        LOW_PASS_RATE_THRESHOLD = 0.5

        # Adjustment step size
        TEMP_STEP = 0.05

        # If pass rate is very low, decrease temperature
        if pass_rate < LOW_PASS_RATE_THRESHOLD:
            new_temp = current_temp - TEMP_STEP
        # If quality is low, decrease temperature
        elif quality < LOW_QUALITY_THRESHOLD:
            new_temp = current_temp - TEMP_STEP
        # If quality is high, slightly increase temperature for diversity
        elif quality > HIGH_QUALITY_THRESHOLD and pass_rate > 0.8:
            new_temp = current_temp + TEMP_STEP * 0.5
        else:
            # Keep current temperature
            new_temp = current_temp

        # Clamp to reasonable range
        return max(0.3, min(1.2, new_temp))

    def get_arm_statistics(self, state: LocalFeedbackState) -> dict[str, dict[str, float]]:
        """
        Get summary statistics for each arm (for debugging/monitoring).

        Args:
            state: LocalFeedbackState

        Returns:
            Dict mapping arm name to statistics (mean_reward, std_reward, count)
        """
        stats = {}
        for arm, rewards in state.arm_rewards.items():
            if rewards:
                stats[arm] = {
                    "mean_reward": float(np.mean(rewards)),
                    "std_reward": float(np.std(rewards)),
                    "count": state.arm_counts.get(arm, 0),
                }
        return stats


# Legacy stub for backwards compatibility
class FeedbackAggregator:
    """
    DEPRECATED: Use FeedbackEngine instead.

    Legacy aggregator for bandit learning.
    """

    def log_feedback(self, arm: str, reward: float, context: dict[str, Any]) -> None:
        """Log feedback entry (stub)."""
        pass

    def aggregate_batch(self) -> dict[str, Any]:
        """Aggregate logged feedback (stub)."""
        pass
