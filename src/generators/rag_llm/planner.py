from typing import Any


class RAGPlanner:
    """
    Build plan-then-write prompts for RAG-based generation.

    Creates structured prompts that guide LLM through planning
    then writing with retrieved context.
    """

    def build_prompt(
        self,
        query: str,
        retrieved_docs: list[dict[str, Any]],
        spec: dict[str, Any]
    ) -> str:
        """
        TODO: Build prompt with retrieved context.
        - Format retrieved documents
        - Add planning instructions
        - Structure for constrained generation
        """
        pass
