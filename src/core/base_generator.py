from abc import ABC, abstractmethod
from typing import Any


class BaseGenerator(ABC):
    """Abstract base class for all synthetic data generators."""

    @abstractmethod
    def generate(self) -> list[dict[str, Any]]:
        """Generate synthetic data."""
        pass

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Return generator capabilities and requirements."""
        pass
