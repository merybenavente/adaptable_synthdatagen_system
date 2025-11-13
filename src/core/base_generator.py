from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseGenerator(ABC):
    """Abstract base class for all synthetic data generators."""

    @abstractmethod
    def generate(self) -> List[Dict[str, Any]]:
        """Generate synthetic data."""
        pass

    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return generator capabilities and requirements."""
        pass
