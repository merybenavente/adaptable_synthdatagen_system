import random
from typing import Any, Dict, List


class EpsilonGreedy:
    """
    Epsilon-greedy contextual bandit algorithm.

    Balances exploration (random arm selection) and exploitation
    (selecting arm with highest average reward) using epsilon parameter.

    With probability epsilon: explore (random arm)
    With probability 1-epsilon: exploit (best arm based on history)
    """

    def __init__(self, arms: List[str], epsilon: float = 0.1):
        """
        Initialize epsilon-greedy bandit.

        Args:
            arms: List of arm names (generator names)
            epsilon: Exploration probability (0.0 = pure exploitation, 1.0 = pure exploration)
        """
        self.arms = arms
        self.epsilon = epsilon

        # Statistics: {context_key: {arm: {"pulls": int, "total_reward": float}}}
        self.stats: Dict[str, Dict[str, Dict[str, float]]] = {}

    def select_arm(self, context_key: str) -> str:
        """
        Select an arm using epsilon-greedy strategy.

        Args:
            context_key: String representation of context (e.g., "qa_pairs_json")

        Returns:
            Selected arm name
        """
        # Explore with probability epsilon
        if random.random() < self.epsilon:
            return random.choice(self.arms)

        # Exploit: choose arm with highest average reward
        if context_key not in self.stats:
            # No data for this context yet, explore randomly
            return random.choice(self.arms)

        # Find best arm based on average reward
        best_arm = max(self.arms, key=lambda arm: self._avg_reward(context_key, arm))
        return best_arm

    def update(self, context_key: str, arm: str, reward: float) -> None:
        """
        Update statistics after observing reward.

        Args:
            context_key: String representation of context
            arm: Arm that was selected
            reward: Observed reward (e.g., quality score)
        """
        # Initialize context if not seen before
        if context_key not in self.stats:
            self.stats[context_key] = {}

        # Initialize arm stats if not seen in this context
        if arm not in self.stats[context_key]:
            self.stats[context_key][arm] = {"pulls": 0, "total_reward": 0.0}

        # Update statistics
        self.stats[context_key][arm]["pulls"] += 1
        self.stats[context_key][arm]["total_reward"] += reward

    def _avg_reward(self, context_key: str, arm: str) -> float:
        """
        Calculate average reward for an arm in a given context.

        Args:
            context_key: String representation of context
            arm: Arm name

        Returns:
            Average reward (0.0 if arm never tried in this context)
        """
        if context_key not in self.stats:
            return 0.0

        if arm not in self.stats[context_key]:
            return 0.0

        arm_stats = self.stats[context_key][arm]
        pulls = arm_stats["pulls"]

        if pulls == 0:
            return 0.0

        return arm_stats["total_reward"] / pulls

    def get_stats(self, context_key: str) -> Dict[str, Dict[str, Any]]:
        """
        Get statistics for all arms in a given context.

        Args:
            context_key: String representation of context

        Returns:
            Dictionary mapping arm names to their statistics
        """
        return self.stats.get(context_key, {})

    def get_all_stats(self) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """
        Get all statistics across all contexts.

        Returns:
            Full statistics dictionary
        """
        return self.stats
