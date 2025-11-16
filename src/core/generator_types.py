from enum import Enum


class GeneratorType(str, Enum):
    """Enumeration of available generator types."""

    NAIVE = "naive"
    WIZARDLM = "wizardlm"
    TEMPLATER = "templater"
    RAG_LLM = "rag_llm"
