import random
from typing import Any

import numpy as np

from src.core.generator_types import GeneratorType
from src.core.spec import BatchMetrics, GenerationPlan, LocalFeedbackState
from src.router.adaptation_policy import AdaptationPolicy, DefaultAdaptationPolicy


class Router:
    """Router that produces GenerationPlans based on context, progress, and feedback state."""

    def __init__(
        self,
        default_batch_size: int = 5,
        adaptation_policy: AdaptationPolicy | None = None,
    ):
        self.default_batch_size = default_batch_size
        self.adaptation_policy = adaptation_policy or DefaultAdaptationPolicy()

        # Define arms as NAIVE configurations
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
        context: dict[str, Any],
        state: LocalFeedbackState,
        progress: dict[str, Any],
    ) -> GenerationPlan:
        """Route to generator and produce GenerationPlan."""
        # Select arm using epsilon-greedy
        selected_arm_name = self._select_arm(state)
        arm_config = self.arms[selected_arm_name]

        # Compute batch size based on remaining samples
        remaining = progress.get("remaining_samples", 0)
        batch_size = min(self.default_batch_size, remaining) if remaining > 0 else self.default_batch_size

        # Merge arm config with additional parameters
        parameters = {
            "temperature": arm_config["temperature"],
            "top_p": arm_config["top_p"],
            "domain": context["domain_type"],
        }

        return GenerationPlan(
            batch_size=batch_size,
            generator_arm=selected_arm_name,
            parameters=parameters,
            iteration=state.iteration,
        )

    def _select_arm(self, state: LocalFeedbackState) -> str:
        """Select arm using epsilon-greedy strategy."""
        epsilon = state.exploration_rate
        arm_names = list(self.arms.keys())

        # Epsilon-greedy selection
        if random.random() < epsilon:
            # Explore: random arm
            return random.choice(arm_names)
        else:
            # Exploit: best arm based on mean reward
            best_arm = None
            best_reward = float('-inf')

            for arm_name in arm_names:
                rewards = state.arm_rewards.get(arm_name, [])
                if rewards:
                    mean_reward = float(np.mean(rewards))
                    if mean_reward > best_reward:
                        best_reward = mean_reward
                        best_arm = arm_name

            # If no arm has been tried yet, pick random
            if best_arm is None:
                return random.choice(arm_names)

            return best_arm

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
