"""Feedback Engine for adaptive generation pipeline."""

from typing import Any

import numpy as np

from src.core.spec import BatchMetrics, GenerationPlan, LocalFeedbackState, Sample


class FeedbackEngine:
    """Feedback Engine that computes metrics and updates LocalFeedbackState."""

    def __init__(
        self,
        temperature_adaptation: bool = True,
        exploration_decay: bool = True,
        max_history_length: int = 10,
    ):
        """Initialize FeedbackEngine with adaptation settings."""
        self.temperature_adaptation = temperature_adaptation
        self.exploration_decay = exploration_decay
        self.max_history_length = max_history_length

    def compute_batch_metrics(
        self,
        samples: list[Sample],
        total_generated: int,
    ) -> BatchMetrics:
        """Compute metrics for a batch of samples."""
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
        """Update LocalFeedbackState based on batch results."""
        # Update iteration and generated count
        state.iteration += 1
        state.generated_so_far += batch_metrics.num_samples

        # Update arm statistics
        arm_name = str(plan.generator_arm)
        state.arm_counts[arm_name] = state.arm_counts.get(arm_name, 0) + 1

        # Use mean_quality as reward (or pass_rate if quality not available)
        reward = (
            batch_metrics.mean_quality
            if batch_metrics.mean_quality is not None
            else batch_metrics.pass_rate
        )

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
        """Adapt temperature based on quality and pass rate."""
        # Define thresholds
        low_quality_threshold = 0.6
        high_quality_threshold = 0.8
        low_pass_rate_threshold = 0.5

        # Adjustment step size
        temp_step = 0.05

        # If pass rate is very low, decrease temperature
        if pass_rate < low_pass_rate_threshold:
            new_temp = current_temp - temp_step
        # If quality is low, decrease temperature
        elif quality < low_quality_threshold:
            new_temp = current_temp - temp_step
        # If quality is high, slightly increase temperature for diversity
        elif quality > high_quality_threshold and pass_rate > 0.8:
            new_temp = current_temp + temp_step * 0.5
        else:
            # Keep current temperature
            new_temp = current_temp

        # Clamp to reasonable range
        return max(0.3, min(1.2, new_temp))

    def get_arm_statistics(self, state: LocalFeedbackState) -> dict[str, dict[str, float]]:
        """Get summary statistics for each arm."""
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
    """DEPRECATED: Use FeedbackEngine instead."""

    def log_feedback(self, arm: str, reward: float, context: dict[str, Any]) -> None:
        """Log feedback entry (stub)."""
        pass

    def aggregate_batch(self) -> dict[str, Any]:
        """Aggregate logged feedback (stub)."""
        pass
