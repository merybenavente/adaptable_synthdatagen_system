from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, field_validator


class Domain(str, Enum):
    """Supported generation domains."""
    TASK_REWRITE = "task_rewrite"
    QA_PAIRS = "qa_pairs"
    CODE_SNIPPETS = "code_snippets"


class Spec(BaseModel):
    """Input specification for data generation request."""

    domain: Domain = Field(..., description="Type of data generation task")
    task_input: Union[str, Dict[str, Any]] = Field(..., description="Input content")
    num_samples: int = Field(..., gt=0, description="Number of samples to generate")
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Domain-specific constraints"
    )
    output_format: str = Field(default="text", description="Output format")

    @field_validator("num_samples")
    @classmethod
    def validate_num_samples(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("num_samples must be positive")
        return v


class Lineage(BaseModel):
    """Provenance tracking for generated samples."""

    original_sample: Optional[UUID] = Field(
        None,
        description="UUID of root ancestor (None for initial generation)"
    )
    num_of_evolutions: int = Field(
        default=0,
        ge=0,
        description="Number of evolution steps from original"
    )
    parent_id: Optional[UUID] = Field(
        None,
        description="UUID of immediate parent sample (None for initial generation)"
    )
    generator: str = Field(..., description="Generator used to create this sample")
    generator_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters used by generator"
    )


class Sample(BaseModel):
    """Individual generated sample with content and metadata."""

    id: UUID = Field(default_factory=uuid4, description="Unique sample identifier")
    content: Union[str, Dict[str, Any]] = Field(..., description="Generated content")
    metadata: Dict[str, Any] = Field(
        default_factory=lambda: {"timestamp": datetime.utcnow().isoformat()},
        description="Operational metadata"
    )
    lineage: Lineage = Field(..., description="Generation provenance")
    quality_scores: Dict[str, float] = Field(
        default_factory=dict,
        description="Quality assessment scores by validator name"
    )


# TODO: review all the Plan logic
class Plan(BaseModel):
    """
    Execution plan with sequence of generation steps/arms.

    TODO: Define fields (e.g., steps, selected_generators, routing_decisions, etc.)
    """
    pass
