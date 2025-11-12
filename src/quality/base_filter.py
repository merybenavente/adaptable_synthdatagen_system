from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseFilter(ABC):
    """Abstract base class for quality filters."""

    @abstractmethod
    def filter(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter data based on quality criteria."""
        pass
