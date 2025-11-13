from typing import Any, Dict, List
from pathlib import Path


class TaskHarness:
    """
    Tiny downstream task evaluation harness.

    Train/eval small model on generated data to measure utility.
    """

    def run_evaluation(
        self,
        dataset_path: Path,
        task_config: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        TODO: Run downstream task evaluation.
        - Load generated dataset
        - Train small model (or use existing)
        - Evaluate on test set
        - Return metrics (accuracy, F1, etc.)
        """
        pass

    def train_model(self, train_data: List[Dict[str, Any]], config: Dict[str, Any]) -> Any:
        """
        TODO: Train a small model.
        - Simple classifier or model for the task
        - Return trained model
        """
        pass

    def evaluate_model(self, model: Any, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        TODO: Evaluate model on test data.
        - Calculate metrics
        - Return results dict
        """
        pass
