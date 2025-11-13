from typing import Any, Dict, List

from src.core.base_generator import BaseGenerator


class RAGLLMGenerator(BaseGenerator):
    """
    RAG + LLM generator with constrained decoding.

    Retrieves relevant knowledge, plans, then generates with
    optional JSON schema or regex constraints.
    """

    def generate(self) -> List[Dict[str, Any]]:
        """
        TODO: Implement RAG-based generation.
        - Retrieve relevant documents
        - Build plan-then-write prompt
        - Generate with constraints (JSON schema/regex)
        - Include attributions in output
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Return RAG-LLM generator capabilities."""
        return {
            "name": "rag_llm",
            "domain": "knowledge_grounded",
            "method": "retrieval_augmented",
            "complexity": "high",
            "supports_constraints": True
        }
