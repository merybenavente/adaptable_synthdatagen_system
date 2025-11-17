from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from src.core.spec import Sample, Spec


class ValidationResult(BaseModel):
    """Validation result with score, pass/fail status, and optional metadata."""

    score: float = Field(..., description="Numeric quality score (typically 0.0 to 1.0)")
    passed: bool = Field(..., description="Whether sample passed validation threshold")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Optional additional validation details"
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
