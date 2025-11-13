from typing import Any, Dict, List

from src.core.base_generator import BaseGenerator


class NaiveGenerator(BaseGenerator):
    """
    Naive generator that directly calls LLM for generation.

    Simple baseline: takes prompt, calls LLM, returns response.
    No evolution, no RAG, no templates - just straightforward generation.
    """

    def generate(self) -> List[Dict[str, Any]]:
        """
        TODO: Implement naive LLM generation.
        - Take input prompt/instruction
        - Call LLM directly
        - Return generated samples
        """
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        """Return naive generator capabilities."""
        return {
            "name": "naive",
            "domain": "open",
            "method": "direct_llm",
            "complexity": "low"
        }
