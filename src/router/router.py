from typing import Dict, Any

from src.core.spec import Spec
from src.core.config_loader import load_yaml
from src.router.simple_router import SimpleRouter
from src.router.context_extractor import ContextExtractor
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class Router:
    """
    Main router facade for generator selection.

    Loads configuration and delegates to appropriate routing strategy:
    - SimpleRouter: Static domain-based routing (V0)
    - BanditRouter: Learning-based routing (future)
    """

    def __init__(self, config_path: str = "config/routing.yaml"):
        """
        Initialize router with configuration.

        Args:
            config_path: Path to routing configuration YAML file
        """
        self.config = load_yaml(config_path)
        self.context_extractor = ContextExtractor()

        # For MVP, always use SimpleRouter
        # TODO: Support BanditRouter when bandit.enabled = true
        self._initialize_router()

        logger.info(f"Router initialized with strategy: {self.strategy}")

    def _initialize_router(self) -> None:
        """Initialize routing strategy based on configuration."""
        # Check if bandit is enabled in config
        bandit_enabled = self.config.get("bandit", {}).get("enabled", False)

        if bandit_enabled:
            # TODO: Implement BanditRouter
            logger.warning(
                "Bandit routing requested but not yet implemented. "
                "Falling back to SimpleRouter."
            )
            self.strategy = "simple"
            self.router = SimpleRouter()
        else:
            # Use simple static routing
            self.strategy = "simple"
            self.router = SimpleRouter()

    def route(self, spec: Spec) -> str:
        """
        Select generator for given specification.

        Args:
            spec: Generation request specification

        Returns:
            Generator name to use
        """
        # Extract context (currently just reads from spec)
        context = self.context_extractor.extract(spec)

        # Route using selected strategy
        generator = self.router.route(spec)

        logger.info(
            f"Routed request: domain={spec.domain.value} -> generator={generator}"
        )

        return generator

    def log_feedback(self, spec: Spec, generator: str, reward: float) -> None:
        """
        Log feedback for routing decision.

        Args:
            spec: Original generation request
            generator: Generator that was used
            reward: Quality score or reward signal (e.g., avg quality score)
        """
        # Extract context
        context = self.context_extractor.extract(spec)

        # Log feedback to router
        self.router.log_feedback(generator, reward, context)

        logger.debug(
            f"Logged feedback: generator={generator}, reward={reward:.3f}, "
            f"context={context}"
        )
