from abc import ABC, abstractmethod
from typing import Any


class BaseValidator(ABC):
    """
    Abstract base class for validators.

    Validators return scores that are compared against thresholds.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.threshold = config.get("threshold", 0.0)

    @abstractmethod
    def validate(self, sample: dict[str, Any], context: dict[str, Any]) -> dict[str, Any]:
        """
        Validate a sample.

        Returns:
            Dict with 'score' (float) and 'passed' (bool based on threshold)
        """
        pass
