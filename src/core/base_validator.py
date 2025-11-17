from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from src.core.models import Sample, Spec


class ValidationResult(BaseModel):
    """Validation result with score and pass/fail status."""

    score: float = Field(..., description="Validation score")
    passed: bool = Field(..., description="Whether validation passed")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata for additional validation details"
    )


# TODO: Extract embedding client creation to shared factory - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/17
class BaseValidator(ABC):
    """Base class for validators that score samples against thresholds."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.threshold = config.get("threshold", 0.7)

    @abstractmethod
    def is_sample_level(self) -> bool:
        """Return True if this validator operates on individual samples."""
        raise NotImplementedError

    @abstractmethod
    def is_batch_level(self) -> bool:
        """Return True if this validator operates on batches of samples."""
        raise NotImplementedError

    def validate(self, sample: Sample, spec: Spec) -> ValidationResult:
        """Validate a single sample; returns ValidationResult with score and passed status."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support sample-level validation. "
            "Implement validate() for sample-level validation."
        )

    def validate_batch(self, samples: list[Sample], spec: Spec) -> ValidationResult:
        """Validate entire batch; returns ValidationResult with single batch score and passed."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batch-level validation. "
            "Implement validate_batch() for batch-level validation."
        )
