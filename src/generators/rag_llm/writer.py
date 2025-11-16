from typing import Any

from src.core.base_generator import BaseGenerator
from src.core.generator_types import GeneratorType


class RAGLLMGenerator(BaseGenerator):
    """
    RAG + LLM generator with constrained decoding.

    Retrieves relevant knowledge, plans, then generates with
    optional JSON schema or regex constraints.
    """

    def generate(self) -> list[dict[str, Any]]:
        """
        TODO: Implement RAG-based generation.
        - Retrieve relevant documents
        - Build plan-then-write prompt
        - Generate with constraints (JSON schema/regex)
        - Include attributions in output
        """
        pass

    def get_capabilities(self) -> dict[str, Any]:
        """Return RAG-LLM generator capabilities."""
        return {
            "name": GeneratorType.RAG_LLM,
            "domain": "knowledge_grounded",
            "method": "retrieval_augmented",
            "complexity": "high",
            "supports_constraints": True
        }
