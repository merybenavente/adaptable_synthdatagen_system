from abc import ABC
from typing import Any, Dict, List, TypedDict

from src.core.spec import Sample, Spec


class ValidationResult(TypedDict):
    """Validation result with score and pass/fail status."""
    score: float
    passed: bool


class BaseValidator(ABC):
    """Base class for validators that score samples against thresholds."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.threshold = config.get("threshold", 0.7)

    def validate(self, sample: Sample, spec: Spec) -> ValidationResult:
        """Validate a single sample; returns ValidationResult with score and passed status."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support unit-level validation. "
            "Implement validate_batch() for batch-level validation."
        )

    def validate_batch(self, samples: List[Sample], spec: Spec) -> ValidationResult:
        """Validate entire batch; returns ValidationResult with single batch score and passed status."""
        pass
