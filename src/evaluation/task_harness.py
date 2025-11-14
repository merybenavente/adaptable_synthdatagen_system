from pathlib import Path
from typing import Any


# TODO: this is a P2 for now. I would like to run these on tinker
class TaskHarness:
    """
    Tiny downstream task evaluation harness.

    Train/eval small model on generated data to measure utility.
    """

    def run_evaluation(
        self,
        dataset_path: Path,
        task_config: dict[str, Any]
    ) -> dict[str, float]:
        """
        TODO: Run downstream task evaluation.
        - Load generated dataset
        - Train small model (or use existing)
        - Evaluate on test set
        - Return metrics (accuracy, F1, etc.)
        """
        pass

    def train_model(self, train_data: list[dict[str, Any]], config: dict[str, Any]) -> Any:
        """
        TODO: Train a small model.
        - Simple classifier or model for the task
        - Return trained model
        """
        pass

    def evaluate_model(self, model: Any, test_data: list[dict[str, Any]]) -> dict[str, float]:
        """
        TODO: Evaluate model on test data.
        - Calculate metrics
        - Return results dict
        """
        pass
