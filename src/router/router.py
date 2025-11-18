import random

import numpy as np

from src.core.generator_types import GeneratorType
from src.core.models import BatchMetrics, GenerationContext, GenerationPlan, LocalFeedbackState
from src.router.adaptation_policy import AdaptationPolicy, DefaultAdaptationPolicy


# TODO: Extend router capabilities with context-aware bandits - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/25
# TODO: Add cost-aware routing logic - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/9
class Router:
    """Router that produces GenerationPlans based on context, progress, and feedback state."""

    def __init__(
        self,
        default_batch_size: int = 5,
        adaptation_policy: AdaptationPolicy | None = None,
    ):
        self.default_batch_size = default_batch_size
        self.adaptation_policy = adaptation_policy or DefaultAdaptationPolicy()

        # TODO: Replace hardcoded arms with centralized registry - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/15
        self.arms = {
            "naive_conservative": {
                "generator": GeneratorType.NAIVE,
                "temperature": 0.5,
                "top_p": 0.8,
            },
            "naive_balanced": {
                "generator": GeneratorType.NAIVE,
                "temperature": 0.7,
                "top_p": 1.0,
            },
            "naive_creative": {
                "generator": GeneratorType.NAIVE,
                "temperature": 0.9,
                "top_p": 1.0,
            },
        }

    def route(
        self,
        context: GenerationContext,
        state: LocalFeedbackState,
    ) -> GenerationPlan:
        """Route to generator and produce GenerationPlan."""
        # Select arm using epsilon-greedy with reasoning
        selected_arm_name, reasoning = self._select_arm(state)
        arm_config = self.arms[selected_arm_name]

        # Compute batch size based on remaining samples from context.progress
        remaining = context.progress.remaining_samples
        batch_size = (
            min(self.default_batch_size, remaining)
            if remaining > 0
            else self.default_batch_size
        )

        # Merge arm config with additional parameters
        parameters = {
            "temperature": arm_config["temperature"],
            "top_p": arm_config["top_p"],
            "domain": context.domain.value,
        }

        return GenerationPlan(
            batch_size=batch_size,
            generator_arm=selected_arm_name,
            parameters=parameters,
            iteration=state.iteration,
            reasoning=reasoning,
        )

    def _select_arm(self, state: LocalFeedbackState) -> tuple[str, str]:
        """Select arm using epsilon-greedy strategy and return reasoning."""
        epsilon = state.exploration_rate
        arm_names = list(self.arms.keys())
        random_value = random.random()

        # Epsilon-greedy selection
        if random_value < epsilon:
            # Explore: random arm
            selected = random.choice(arm_names)
            reasoning = (
                f"EXPLORATION: Random value ({random_value:.3f}) < ε ({epsilon:.3f}). "
                f"Randomly selected '{selected}' to explore different generation strategies"
            )
            return selected, reasoning

        # Exploit: best arm based on mean reward
        best_arm = None
        best_reward = float('-inf')
        arm_rewards_summary = {}

        for arm_name in arm_names:
            rewards = state.arm_rewards.get(arm_name, [])
            if rewards:
                mean_reward = float(np.mean(rewards))
                arm_rewards_summary[arm_name] = mean_reward
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_arm = arm_name

        # If no arm has been tried yet, pick random
        if best_arm is None:
            selected = random.choice(arm_names)
            reasoning = (
                f"INITIAL: Random value ({random_value:.3f}) >= ε ({epsilon:.3f}), "
                f"but no arm has been tried yet.\n"
                f"ARM SELECTION: Randomly selected '{selected}' to start exploration"
            )
            return selected, reasoning

        # Build reasoning string with arm comparison
        rewards_str = ", ".join(
            f"{arm}={reward:.3f}" for arm, reward in sorted(
                arm_rewards_summary.items(), key=lambda x: x[1], reverse=True
            )
        )
        reasoning = (
            f"EXPLOITATION: Random value ({random_value:.3f}) >= ε ({epsilon:.3f}). "
            f"Selected '{best_arm}' with best mean reward ({best_reward:.3f}). "
            f"Arm rewards: [{rewards_str}]"
        )
        return best_arm, reasoning

    def adapt(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> LocalFeedbackState:
        """Adapt generation parameters based on batch metrics."""
        new_exploration = self.adaptation_policy.adapt_exploration(state, metrics)

        return state.model_copy(
            update={
                "exploration_rate": new_exploration,
            }
        )
