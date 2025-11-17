from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.core.models import Sample


# TODO: Track cost trends across batches - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/10
class BaseGenerator(ABC):
    """Abstract base class for all synthetic data generators."""

    @abstractmethod
    def generate(self) -> list[Sample]:
        """Generate synthetic data."""
        pass

    @abstractmethod
    def get_capabilities(self) -> dict[str, Any]:
        """Return generator capabilities and requirements."""
        pass
