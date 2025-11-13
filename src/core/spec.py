from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Spec(BaseModel):
    """
    Input specification for data generation request.

    TODO: Define fields (e.g., domain, num_samples, constraints, etc.)
    """
    pass


class Plan(BaseModel):
    """
    Execution plan with sequence of generation steps/arms.

    TODO: Define fields (e.g., steps, selected_generators, routing_decisions, etc.)
    """
    pass


class Sample(BaseModel):
    """
    Individual generated sample with content and metadata.

    TODO: Define fields (e.g., content, metadata, quality_scores, etc.)
    """
    pass


class Lineage(BaseModel):
    """
    Provenance tracking for generated samples.

    TODO: Define fields (e.g., generator_used, timestamps, parent_samples, etc.)
    """
    pass
