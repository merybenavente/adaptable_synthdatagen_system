"""Feedback Engine for adaptive generation pipeline."""

import numpy as np

from src.core.models import BatchMetrics, GenerationPlan, LocalFeedbackState, Sample


class FeedbackEngine:
    """Feedback Engine that computes metrics and updates LocalFeedbackState."""

    def __init__(self):
        """Initialize FeedbackEngine."""
        pass

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
            # Exclude scores from skipped validators (score=0.0 with skipped metadata)
            validation_results = sample.metadata.get("validation_results", {})
            for validator_name, score in sample.quality_scores.items():
                result = validation_results.get(validator_name, {})
                metadata = result.get("metadata", {})
                # Skip if validator was skipped (has skipped=True in metadata)
                if not metadata.get("skipped", False):
                    all_quality_scores.append(score)

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
