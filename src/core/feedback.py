from typing import Any, Dict, List


class FeedbackAggregator:
    """
    Aggregate rewards and feedback for bandit learning.

    Logs feedback to artifacts/bandit/ for later batch updates.
    """

    def log_feedback(self, arm: str, reward: float, context: Dict[str, Any]) -> None:
        """
        TODO: Log feedback entry.
        - arm: generator name
        - reward: quality score or downstream metric
        - context: request features
        """
        pass

    def aggregate_batch(self) -> Dict[str, Any]:
        """
        TODO: Aggregate logged feedback for batch update.
        """
        pass
