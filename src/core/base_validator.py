from abc import ABC
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

    def validate(self, sample: Sample, spec: Spec) -> ValidationResult:
        """Validate a single sample; returns ValidationResult with score and passed status."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support unit-level validation. "
            "Implement validate_batch() for batch-level validation."
        )

    def validate_batch(self, samples: list[Sample], spec: Spec) -> ValidationResult:
        """Validate entire batch; returns ValidationResult with single batch score and passed."""
        pass
