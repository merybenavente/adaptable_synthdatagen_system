import logging
import random

import numpy as np

from src.core.generator_types import GeneratorType
from src.core.models import BatchMetrics, GenerationContext, GenerationPlan, LocalFeedbackState
from src.router.adaptation_policy import AdaptationPolicy, DefaultAdaptationPolicy
from src.utils.logger import Colors

logger = logging.getLogger(__name__)


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
            # NAIVE arms
            "naive_conservative": {
                "generator": GeneratorType.NAIVE,
                "temperature": 0.3,
                "top_p": 0.8,
            },
            "naive_balanced": {
                "generator": GeneratorType.NAIVE,
                "temperature": 0.7,
                "top_p": 1.0,
            },
            "naive_creative": {
                "generator": GeneratorType.NAIVE,
                "temperature": 1.2,
                "top_p": 1.0,
            },
            # TEMPLATER arms
            "templater_conservative": {
                "generator": GeneratorType.TEMPLATER,
                "temperature": 0.7,
                "max_depth": 8,
                "deduplication": False,
            },
            "templater_exploratory": {
                "generator": GeneratorType.TEMPLATER,
                "temperature": 1.2,
                "max_depth": 12,
                "deduplication": False,
            },
            "templater_dedup": {
                "generator": GeneratorType.TEMPLATER,
                "temperature": 0.9,
                "max_depth": 10,
                "deduplication": True,
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

        # Always generate full batch size (selection happens in pipeline)
        batch_size = self.default_batch_size

        # Merge arm config with additional parameters
        parameters = {
            "temperature": arm_config["temperature"],
            "top_p": arm_config["top_p"],
            "domain": context.domain,
        }

        logger.info(
            f"{reasoning}\n🎯 Router decision for iteration {state.iteration + 1}: "
            f"Selected arm {Colors.CYAN}'{selected_arm_name}'{Colors.RESET}"
        )

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

        # Compute best arm based on mean reward (for exploitation)
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

        # Decision order: (INIT) -> EXPLORE? -> EXPLOIT!
        # 1. Initial state: no arms tried yet, pick random
        if best_arm is None:
            selected = random.choice(arm_names)
            reasoning = (
                f"{Colors.CYAN}[INITIAL]{Colors.RESET} No arm has been tried yet. "
                f"Randomly selected {Colors.CYAN}'{selected}'{Colors.RESET} to start exploration"
            )
            return selected, reasoning

        # 2. Epsilon-greedy exploration: random arm
        random_value = random.random()
        if random_value < epsilon:
            selected = random.choice(arm_names)
            reasoning = (
                f"{Colors.CYAN}[EXPLORATION]{Colors.RESET} "
                f"Random value ({random_value:.3f}) < ε ({epsilon:.3f}). "
                f"Randomly selected {Colors.CYAN}'{selected}'{Colors.RESET} "
                f"to explore different generation strategies"
            )
            return selected, reasoning

        # 3. Exploitation: best performing arm
        rewards_str = ", ".join(
            f"{arm}={reward:.3f}" for arm, reward in sorted(
                arm_rewards_summary.items(), key=lambda x: x[1], reverse=True
            )
        )
        reasoning = (
            f"{Colors.CYAN}[EXPLOITATION]{Colors.RESET} "
            f"Random value ({random_value:.3f}) >= ε ({epsilon:.3f}). "
            f"Selected {Colors.CYAN}'{best_arm}'{Colors.RESET} "
            f"with best mean reward ({best_reward:.3f}). "
            f"Arm rewards: [{rewards_str}]"
        )
        return best_arm, reasoning

    def adapt(
        self,
        state: LocalFeedbackState,
        metrics: BatchMetrics,
    ) -> LocalFeedbackState:
        """Adapt generation parameters based on batch metrics."""
        old_exploration = state.exploration_rate
        new_exploration = self.adaptation_policy.adapt_exploration(state, metrics)

        if abs(new_exploration - old_exploration) > 0.001:
            logger.info(
                f"📊 Adaptation: Exploration rate adjusted "
                f"from {old_exploration:.3f} to {new_exploration:.3f} "
                f"(pass_rate={metrics.pass_rate:.2%})"
            )

        return state.model_copy(
            update={
                "exploration_rate": new_exploration,
            }
        )
