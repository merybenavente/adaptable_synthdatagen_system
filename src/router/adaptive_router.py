"""
Adaptive Router for generation pipeline with bandit-based arm selection.

The Router takes Context + LocalFeedbackState and outputs a GenerationPlan
describing the next batch configuration (batch size, generator arm, parameters).
"""

import random
from typing import Any

import numpy as np

from src.core.generator_types import GeneratorType
from src.core.spec import GenerationPlan, LocalFeedbackState, Spec
from src.router.context_extractor import ContextExtractor


class AdaptiveRouter:
    """
    Adaptive Router that selects generator arms based on LocalFeedbackState.

    Implements epsilon-greedy strategy for exploration/exploitation:
    - With probability epsilon, explore by selecting a random arm
    - With probability 1-epsilon, exploit by selecting the best-performing arm
    """

    def __init__(
        self,
        available_arms: list[GeneratorType] | None = None,
        default_batch_size: int = 5,
        min_batch_size: int = 3,
        max_batch_size: int = 10,
        strategy: str = "epsilon_greedy",
    ):
        """
        Initialize AdaptiveRouter.

        Args:
            available_arms: List of available generator arms (defaults to all)
            default_batch_size: Default batch size for generation
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
            strategy: Routing strategy ("epsilon_greedy", "thompson_sampling", "ucb")
        """
        self.available_arms = available_arms or [
            GeneratorType.NAIVE,
            GeneratorType.TEMPLATER,
            GeneratorType.RAG_LLM,
        ]
        self.default_batch_size = default_batch_size
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.strategy = strategy
        self.context_extractor = ContextExtractor()

    def plan_next_batch(
        self,
        spec: Spec,
        state: LocalFeedbackState,
    ) -> GenerationPlan:
        """
        Generate a GenerationPlan for the next batch.

        Args:
            spec: Input specification
            state: Current LocalFeedbackState with arm performance history

        Returns:
            GenerationPlan describing batch configuration
        """
        # Extract context from spec
        context = self.context_extractor.extract(spec)

        # Select generator arm based on strategy
        if self.strategy == "epsilon_greedy":
            selected_arm, reasoning = self._epsilon_greedy(state)
        elif self.strategy == "thompson_sampling":
            # TODO: Implement Thompson sampling
            selected_arm, reasoning = self._epsilon_greedy(state)
        elif self.strategy == "ucb":
            # TODO: Implement UCB
            selected_arm, reasoning = self._epsilon_greedy(state)
        else:
            # Default to epsilon-greedy
            selected_arm, reasoning = self._epsilon_greedy(state)

        # Determine batch size based on remaining samples
        remaining = spec.num_samples - state.generated_so_far
        batch_size = min(self.default_batch_size, remaining, self.max_batch_size)
        batch_size = max(batch_size, self.min_batch_size) if remaining >= self.min_batch_size else remaining

        # Build parameters dict
        parameters = {
            "temperature": state.current_temperature,
            "domain": spec.domain.value,
        }

        # Add any additional parameters from spec constraints
        if spec.constraints:
            parameters.update(spec.constraints)

        return GenerationPlan(
            batch_size=batch_size,
            generator_arm=selected_arm,
            parameters=parameters,
            iteration=state.iteration,
            reasoning=reasoning,
        )

    def _epsilon_greedy(self, state: LocalFeedbackState) -> tuple[GeneratorType, str]:
        """
        Epsilon-greedy arm selection.

        Args:
            state: LocalFeedbackState with arm performance history

        Returns:
            Tuple of (selected_arm, reasoning)
        """
        epsilon = state.exploration_rate

        # Exploration: select random arm
        if random.random() < epsilon:
            selected_arm = random.choice(self.available_arms)
            reasoning = f"Exploring: randomly selected {selected_arm.value} (ε={epsilon:.3f})"
            return selected_arm, reasoning

        # Exploitation: select best-performing arm
        if not state.arm_rewards:
            # No history yet, select random arm
            selected_arm = random.choice(self.available_arms)
            reasoning = f"No history: randomly selected {selected_arm.value}"
            return selected_arm, reasoning

        # Compute mean rewards for each arm
        arm_means = {}
        for arm_name, rewards in state.arm_rewards.items():
            if rewards:
                arm_means[arm_name] = np.mean(rewards)

        if not arm_means:
            # No rewards yet, select random arm
            selected_arm = random.choice(self.available_arms)
            reasoning = f"No rewards: randomly selected {selected_arm.value}"
            return selected_arm, reasoning

        # Select arm with highest mean reward
        best_arm_name = max(arm_means, key=arm_means.get)
        best_reward = arm_means[best_arm_name]

        # Convert string back to GeneratorType
        try:
            selected_arm = GeneratorType(best_arm_name)
        except ValueError:
            # If the arm name is not a valid GeneratorType, fall back to random
            selected_arm = random.choice(self.available_arms)
            reasoning = f"Invalid arm name '{best_arm_name}': randomly selected {selected_arm.value}"
            return selected_arm, reasoning

        reasoning = (
            f"Exploiting: selected {selected_arm.value} with mean reward {best_reward:.3f} "
            f"(count={state.arm_counts.get(best_arm_name, 0)})"
        )
        return selected_arm, reasoning

    def _thompson_sampling(self, state: LocalFeedbackState) -> tuple[GeneratorType, str]:
        """
        Thompson sampling arm selection (TODO).

        Args:
            state: LocalFeedbackState with arm performance history

        Returns:
            Tuple of (selected_arm, reasoning)
        """
        # TODO: Implement Beta-Bernoulli Thompson sampling
        # For now, fall back to epsilon-greedy
        return self._epsilon_greedy(state)

    def _ucb(self, state: LocalFeedbackState) -> tuple[GeneratorType, str]:
        """
        Upper Confidence Bound (UCB) arm selection (TODO).

        Args:
            state: LocalFeedbackState with arm performance history

        Returns:
            Tuple of (selected_arm, reasoning)
        """
        # TODO: Implement UCB1 algorithm
        # For now, fall back to epsilon-greedy
        return self._epsilon_greedy(state)


# Simple Router for backwards compatibility
class Router:
    """Simple domain-based router with deterministic rules."""

    def __init__(self):
        """Initialize router."""
        pass

    def route(self, spec: Spec) -> GeneratorType:
        """Route request to generator based on domain type."""
        # Always route to NAIVE for now
        return GeneratorType.NAIVE

    def log_feedback(
        self, generator: str | GeneratorType, reward: float, context: dict[str, Any]
    ) -> None:
        """Log feedback (no-op for simple router)."""
        pass
