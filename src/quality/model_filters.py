from typing import Any, Dict, List

from src.quality.base_filter import BaseFilter


class EmbeddingSimilarityFilter(BaseFilter):
    """Filter based on embedding similarity to seed data."""
    pass


class PerplexityFilter(BaseFilter):
    """Filter based on perplexity scores."""
    pass
