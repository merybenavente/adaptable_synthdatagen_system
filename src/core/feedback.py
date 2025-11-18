"""Feedback Engine for adaptive generation pipeline."""

import numpy as np

from src.core.models import BatchMetrics, GenerationPlan, LocalFeedbackState, Sample


class FeedbackEngine:
    """Feedback Engine that computes metrics and updates LocalFeedbackState."""

    def __init__(self, quality_reward_validator: str | None = "llm_judge"):
        """Initialize FeedbackEngine."""
        self.quality_reward_validator = quality_reward_validator

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

        for sample in samples:
            if "similarity" in sample.quality_scores:
                similarity_scores.append(sample.quality_scores["similarity"])
            if "diversity" in sample.quality_scores:
                diversity_scores.append(sample.quality_scores["diversity"])

        # Compute metrics
        mean_similarity = float(np.mean(similarity_scores)) if similarity_scores else None
        diversity_score = float(np.mean(diversity_scores)) if diversity_scores else None
        quality_scores = self._collect_validator_scores(samples, self.quality_reward_validator)
        mean_quality = float(np.mean(quality_scores)) if quality_scores else None
        pass_rate = len(samples) / total_generated if total_generated > 0 else 0.0

        return BatchMetrics(
            mean_similarity=mean_similarity,
            diversity_score=diversity_score,
            mean_quality=mean_quality,
            pass_rate=pass_rate,
            num_samples=len(samples),
        )

    def _collect_validator_scores(
        self,
        samples: list[Sample],
        validator_name: str | None,
    ) -> list[float]:
        """Collect non-skipped scores produced by a specific validator."""
        if not validator_name:
            return []

        scores: list[float] = []
        for sample in samples:
            validation_results = sample.metadata.get("validation_results", {})
            result = validation_results.get(validator_name)
            if result is None:
                continue

            metadata = result.get("metadata") or {}
            if metadata.get("skipped", False):
                continue

            try:
                scores.append(float(result.get("score", 0.0)))
            except (TypeError, ValueError):
                continue
        return scores

    def compute_reward(self, batch_metrics: BatchMetrics) -> float:
        """Compute composite reward combining pass rate with quality score."""
        quality_component = batch_metrics.mean_quality or 0.0
        return batch_metrics.pass_rate * quality_component

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

        # Use composite reward = pass_rate * quality_score (quality defaults to 1.0 if missing)
        reward = self.compute_reward(batch_metrics)

        if arm_name not in state.arm_rewards:
            state.arm_rewards[arm_name] = []
        state.arm_rewards[arm_name].append(reward)

        return state

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
