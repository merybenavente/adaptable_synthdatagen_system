from typing import Any

from src.core.base_generator import BaseGenerator
from src.core.generator_types import GeneratorType
from src.core.spec import Domain, Lineage, Sample, Spec
from src.generators.templater.grammar import Grammar
from src.generators.templater.sampler import GrammarSampler
from src.utils.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class TemplaterGenerator(BaseGenerator):
    """Template-based generator using PCFG with LLM content filling."""

    DOMAIN_TO_RECIPE = {
        Domain.QA_PAIRS: "config/recipes/qa_pairs_example.yaml",
        Domain.TASK_REWRITE: "config/recipes/task_rewrite_example.yaml",
    }

    def __init__(self, spec: Spec):
        self.spec = spec

        # Extract adaptive parameters from constraints
        constraints = spec.constraints or {}
        self.temperature = constraints.get('temperature', 1.0)
        self.max_depth = constraints.get('max_depth', 10)
        self.enable_dedup = constraints.get('deduplication', False)
        self.model = constraints.get('model', 'gpt-4o-mini')

        # Determine grammar file path
        grammar_path = constraints.get('grammar_path')
        if not grammar_path:
            grammar_path = self.DOMAIN_TO_RECIPE.get(spec.domain)
            if not grammar_path:
                raise ValueError(f"No default grammar for domain: {spec.domain}")

        # Load grammar and initialize sampler
        self.grammar = Grammar.from_yaml(grammar_path)
        self.llm_client = LLMClient(model=self.model)
        self.sampler = GrammarSampler(
            grammar=self.grammar,
            llm_client=self.llm_client,
            temperature=self.temperature,
            max_depth=self.max_depth
        )

        # Track generated samples for deduplication
        self.generated_hashes = set()

        logger.info(
            f"TemplaterGenerator initialized: temp={self.temperature}, "
            f"max_depth={self.max_depth}, dedup={self.enable_dedup}, "
            f"grammar={grammar_path}"
        )

    def generate(self) -> list[Sample]:
        """Generate samples by sampling from grammar."""
        samples = []
        attempts = 0
        max_attempts = self.spec.num_samples * 5  # Allow retries for dedup

        while len(samples) < self.spec.num_samples and attempts < max_attempts:
            attempts += 1

            try:
                # Sample from grammar
                content = self.sampler.sample()

                # Check for duplicates if deduplication enabled
                if self.enable_dedup:
                    content_hash = hash(content)
                    if content_hash in self.generated_hashes:
                        logger.debug(f"Skipping duplicate sample (attempt {attempts})")
                        continue
                    self.generated_hashes.add(content_hash)

                # Create sample with lineage
                sample = Sample(
                    content=content,
                    lineage=Lineage(
                        generator=GeneratorType.TEMPLATER,
                        generator_parameters={
                            "model": self.model,
                            "temperature": self.temperature,
                            "max_depth": self.max_depth,
                            "deduplication": self.enable_dedup,
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

        if len(samples) < self.spec.num_samples:
            logger.warning(
                f"Only generated {len(samples)}/{self.spec.num_samples} samples "
                f"after {attempts} attempts"
            )

        return samples

    def get_capabilities(self) -> dict[str, Any]:
        """Return templater capabilities."""
        return {
            "name": GeneratorType.TEMPLATER,
            "domain": self.spec.domain.value,
            "method": "grammar_based",
            "complexity": "medium"
        }
