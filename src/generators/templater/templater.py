from typing import Any

from src.core.base_generator import BaseGenerator
from src.core.generator_types import GeneratorType
from src.core.models import GenerationContext, GenerationPlan, Lineage, Sample
from src.generators.templater.grammar import Grammar
from src.generators.templater.sampler import GrammarSampler
from src.utils.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TemplaterGenerator(BaseGenerator):
    """Template-based generator using PCFG with LLM content filling."""

    def __init__(self, context: GenerationContext, plan: GenerationPlan):
        self.context = context
        self.plan = plan

        if not context.grammar_path:
            raise ValueError(
                "TemplaterGenerator requires grammar_path in context. "
                "Specify 'grammar_path' in constraints."
            )

        # Extract adaptive parameters from plan (preferred) and constraints (fallback)
        self.temperature = (
            plan.parameters.get('temperature')
            or context.constraints.get('temperature', 1.0)
        )
        self.max_depth = (
            plan.parameters.get('max_depth')
            or context.constraints.get('max_depth', 10)
        )
        self.model = plan.parameters.get('model') or context.constraints.get('model', 'gpt-4o-mini')

        # Use grammar path from context
        grammar_path = context.grammar_path

        # Load grammar and initialize sampler
        self.grammar = Grammar.from_yaml(grammar_path)
        self.llm_client = LLMClient(model=self.model)
        self.sampler = GrammarSampler(
            grammar=self.grammar,
            llm_client=self.llm_client,
            temperature=self.temperature,
            max_depth=self.max_depth
        )

        logger.info(
            f"TemplaterGenerator initialized: temp={self.temperature}, "
            f"max_depth={self.max_depth}, grammar={grammar_path}"
        )

    def generate(self) -> list[Sample]:
        """Generate samples by sampling from grammar."""
        samples = []
        attempts = 0
        batch_size = self.plan.batch_size
        max_attempts = batch_size * 5  # Allow retries for dedup

        while len(samples) < batch_size and attempts < max_attempts:
            attempts += 1

            try:
                # Sample from grammar
                content = self.sampler.sample()

                # Create sample with lineage
                sample = Sample(
                    content=content,
                    lineage=Lineage(
                        generator=GeneratorType.TEMPLATER,
                        generator_parameters={
                            "model": self.model,
                            "temperature": self.temperature,
                            "max_depth": self.max_depth,
                        }
                    )
                )
                samples.append(sample)

            except RecursionError as e:
                logger.warning(f"Recursion error during sampling: {e}")
                continue
            except Exception as e:
                logger.error(f"Error during sampling: {e}")
                continue

        if len(samples) < batch_size:
            logger.warning(
                f"Only generated {len(samples)}/{batch_size} samples "
                f"after {attempts} attempts"
            )

        return samples

    def get_capabilities(self) -> dict[str, Any]:
        """Return templater capabilities."""
        return {
            "name": GeneratorType.TEMPLATER,
            "domain": self.context.domain,
            "method": "grammar_based",
            "complexity": "medium"
        }
