from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from src.core.generator_types import GeneratorType


class Domain(str, Enum):
    """Supported generation domains."""
    TASK_REWRITE = "task_rewrite"
    QA_PAIRS = "qa_pairs"
    CODE_SNIPPETS = "code_snippets"


class Spec(BaseModel):
    """Input specification for data generation request."""

    domain: Domain = Field(..., description="Type of data generation task")
    task_input: str | dict[str, Any] = Field(..., description="Input content")
    num_samples: int = Field(..., gt=0, description="Number of samples to generate")
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific constraints"
    )
    output_format: str = Field(default="text", description="Output format")

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v: Domain | str) -> Domain:
        if isinstance(v, str):
            try:
                return Domain(v)
            except ValueError:
                valid_domains = [d.value for d in Domain]
                raise ValueError(
                    f"Invalid domain '{v}'. Must be one of: {', '.join(valid_domains)}"
                )
        return v

    @field_validator("num_samples")
    @classmethod
    def validate_num_samples(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_samples must be positive")
        return v


# TODO: Lineage will make more sense once we implement evolving methods - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/20
class Lineage(BaseModel):
    """Provenance tracking for generated samples."""

    original_sample: UUID | None = Field(
        None,
        description="UUID of root ancestor (None for initial generation)"
    )
    num_of_evolutions: int = Field(
        default=0,
        ge=0,
        description="Number of evolution steps from original"
    )
    parent_id: UUID | None = Field(
        None,
        description="UUID of immediate parent sample (None for initial generation)"
    )
    generator: str | GeneratorType = Field(..., description="Generator used to create this sample")
    generator_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used by generator"
    )


class Sample(BaseModel):
    """Individual generated sample with content and metadata."""

    id: UUID = Field(default_factory=uuid4, description="Unique sample identifier")
    content: str | dict[str, Any] = Field(..., description="Generated content")
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": datetime.utcnow().isoformat()},
        description="Operational metadata"
    )
    lineage: Lineage = Field(..., description="Generation provenance")
    quality_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Quality assessment scores by validator name"
    )


class GenerationPlan(BaseModel):
    """Plan for a single batch generation, output by Router."""

    batch_size: int = Field(..., gt=0, description="Number of samples to generate in this batch")
    generator_arm: str | GeneratorType = Field(..., description="Which generator to use")
    parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Generator-specific parameters (e.g., temperature, top_p)"
    )
    iteration: int = Field(..., ge=0, description="Batch iteration number")
    reasoning: str | None = Field(
        None,
        description="Optional explanation for why this plan was chosen"
    )


class BatchMetrics(BaseModel):
    """Computed metrics for a generated batch."""

    mean_similarity: float | None = Field(None, description="Mean semantic similarity score")
    diversity_score: float | None = Field(None, description="Batch diversity score")
    mean_quality: float | None = Field(None, description="Mean overall quality score")
    pass_rate: float = Field(
        ..., ge=0.0, le=1.0, description="Fraction of samples that passed validation"
    )
    num_samples: int = Field(..., ge=0, description="Number of samples in batch")
    custom_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Domain-specific or validator-specific metrics"
    )


class LocalFeedbackState(BaseModel):
    """State container for adaptive feedback loop within a single pipeline run."""

    generated_so_far: int = Field(default=0, ge=0, description="Total samples generated so far")
    iteration: int = Field(default=0, ge=0, description="Current iteration/batch number")

    # Arm performance tracking (for bandit)
    arm_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Number of times each generator arm has been used"
    )
    arm_rewards: dict[str, list[float]] = Field(
        default_factory=dict,
        description="Reward history for each arm (e.g., quality scores)"
    )

    # Adaptive hyperparameters
    exploration_rate: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Exploration rate for epsilon-greedy or similar strategies"
    )

    # Additional state
    state_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible storage for router-specific or domain-specific state"
    )


# Legacy alias for backwards compatibility
class Plan(BaseModel):
    """Execution plan with sequence of generation steps/arms."""
    pass
