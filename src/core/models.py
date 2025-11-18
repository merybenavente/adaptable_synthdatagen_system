from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, field_validator

from src.core.generator_types import GeneratorType


class ProgressState(BaseModel):
    """Dynamic progress tracking within a generation run."""

    remaining_samples: int = Field(..., ge=0, description="Samples remaining to generate")
    collected_samples: int = Field(..., ge=0, description="Samples collected so far")
    rejected_samples: int = Field(default=0, ge=0, description="Samples rejected so far")
    iteration: int = Field(..., ge=0, description="Current iteration number")


class GenerationContext(BaseModel):
    """Complete context for routing decisions with intelligent feature extraction."""

    # Core fields from Spec
    domain: str = Field(..., description="Generation domain (optional metadata)")
    task_input: str | dict[str, Any] = Field(..., description="Input content")
    num_samples: int = Field(..., gt=0, description="Total samples to generate")
    constraints: dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific constraints"
    )
    grammar_path: str | None = Field(
        None,
        description="Path to PCFG grammar file for templater generator"
    )

    # Intelligent extracted features
    complexity_level: str = Field(
        default="medium",
        description="Inferred complexity: simple|medium|complex"
    )
    constraint_count: int = Field(default=0, ge=0, description="Number of constraints")
    has_examples: bool = Field(default=False, description="Whether examples are provided")
    has_knowledge_base: bool = Field(
        default=False,
        description="Whether knowledge base is referenced"
    )

    # Dynamic progress (updated each iteration)
    progress: ProgressState = Field(
        ...,
        description="Current generation progress"
    )

    def update_progress(
        self,
        collected: int,
        rejected: int,
        iteration: int
    ) -> GenerationContext:
        """Return updated context with new progress."""
        return self.model_copy(update={
            "progress": ProgressState(
                remaining_samples=self.num_samples - collected,
                collected_samples=collected,
                rejected_samples=rejected,
                iteration=iteration,
            )
        })


class Spec(BaseModel):
    """Input specification for data generation request (user's job specification)."""

    domain: str = Field(..., description="Type of data generation task")
    task_input: str | dict[str, Any] = Field(..., description="Input content")
    num_samples: int = Field(..., gt=0, description="Number of samples to generate")
    constraints: dict[str, Any] = Field(
        default_factory=dict, description="Domain-specific constraints"
    )
    output_format: str = Field(default="text", description="Output format")
    output_path: str | None = Field(
        None, description="Output file path (required for batch formats)"
    )

    @field_validator("num_samples")
    @classmethod
    def validate_num_samples(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_samples must be positive")
        return v


# TODO: Lineage will make more sense once we implement evolving methods - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/20
class Lineage(BaseModel):
    """Provenance tracking for generated samples.

    This class is only present in Samples that are generated from other samples.
    Spontaneously generated samples (created without a parent) have lineage=None.
    """

    original_sample: str | None = Field(
        None, description="Content of root ancestor (only for evolved samples)"
    )
    original_sample_id: UUID | None = Field(
        None, description="UUID of root ancestor (only for evolved samples)"
    )
    num_of_evolutions: int = Field(
        default=0, ge=0, description="Number of evolution steps from original"
    )
    parent_id: UUID | None = Field(None, description="UUID of immediate parent sample")
    generator: str | GeneratorType = Field(..., description="Generator used to create this sample")
    generator_parameters: dict[str, Any] = Field(
        default_factory=dict, description="Parameters used by generator"
    )


class Sample(BaseModel):
    """Individual generated sample with content and metadata."""

    id: UUID = Field(default_factory=uuid4, description="Unique sample identifier")
    content: str | dict[str, Any] = Field(..., description="Generated content")
    metadata: dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": datetime.utcnow().isoformat()},
        description="Operational metadata",
    )
    lineage: Lineage | None = Field(
        None,
        description="Generation provenance (None for spontaneously generated or input samples)"
    )
    quality_scores: dict[str, float] = Field(
        default_factory=dict, description="Quality assessment scores by validator name"
    )


class GenerationPlan(BaseModel):
    """Router's decision for a single batch generation."""

    batch_size: int = Field(..., gt=0, description="Number of samples to generate in this batch")
    generator_arm: str | GeneratorType = Field(..., description="Which generator to use")
    parameters: dict[str, Any] = Field(
        default_factory=dict, description="Generator-specific parameters (e.g., temperature, top_p)"
    )
    iteration: int = Field(..., ge=0, description="Batch iteration number")
    reasoning: str | None = Field(
        None, description="Optional explanation for why this plan was chosen"
    )


# TODO: Add cost tracking to feedback loop - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/9
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
        default_factory=dict, description="Domain-specific or validator-specific metrics"
    )


# TODO: Add cost tracking to feedback loop - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/9
class LocalFeedbackState(BaseModel):
    """State container for adaptive feedback loop within a single pipeline run."""

    generated_so_far: int = Field(default=0, ge=0, description="Total samples generated so far")
    iteration: int = Field(default=0, ge=0, description="Current iteration/batch number")

    # Arm performance tracking (for bandit)
    arm_counts: dict[str, int] = Field(
        default_factory=dict, description="Number of times each generator arm has been used"
    )
    arm_rewards: dict[str, list[float]] = Field(
        default_factory=dict, description="Reward history for each arm (e.g., quality scores)"
    )

    # Adaptive hyperparameters
    exploration_rate: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Exploration rate for epsilon-greedy or similar strategies",
    )

    # Additional state
    state_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible storage for router-specific or domain-specific state",
    )
