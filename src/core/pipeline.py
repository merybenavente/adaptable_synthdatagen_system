from src.core.generator_types import GeneratorType
from src.core.spec import Sample, Spec
from src.generators.naive_generator import NaiveGenerator
from src.router import Router


class Pipeline:
    """Main orchestration pipeline for synthetic data generation."""

    def __init__(self, routing_config_path: str | None = None):
        self.router = Router()
        # Map generator types to classes
        self.generators = {
            GeneratorType.NAIVE: NaiveGenerator,
            # GeneratorType.WIZARDLM: WizardLMGenerator,
            # GeneratorType.TEMPLATER: TemplaterGenerator,
        }
        # TODO: Use routing_config_path when Router supports configuration

    def run(self, spec: Spec) -> list[Sample]:
        """Execute generation pipeline: route → generate → return."""
        # 1. Route to appropriate generator
        generator_type = self.router.route(spec)

        # 2. Instantiate selected generator
        generator_class = self.generators.get(generator_type)
        if not generator_class:
            raise ValueError(f"Unknown generator: {generator_type}")

        generator = generator_class(spec)

        # 3. Generate samples
        samples = generator.generate()

        return samples
