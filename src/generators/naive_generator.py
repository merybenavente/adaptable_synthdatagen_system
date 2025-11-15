
from src.core.base_generator import BaseGenerator
from src.core.spec import Domain, Lineage, Sample, Spec
from src.utils.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class NaiveGenerator(BaseGenerator):
    """Naive generator that directly calls LLM for generation."""

    DOMAIN_TEMPLATES = {
        Domain.TASK_REWRITE: """Generate {num_samples} variants of the following task instruction:

Task: {task_input}

{constraints_instructions}

Return only the variants, one per line, numbered 1-{num_samples}.""",
        Domain.QA_PAIRS: """Generate {num_samples} question-answer pairs based on:

Topic: {task_input}

{constraints_instructions}

Return in JSON format with 'question' and 'answer' fields.""",
    }

    CONSTRAINTS_BUILDER_PROMPT = """Given these generation constraints as a dictionary:

{constraints_dict}

Convert them into clear, natural language instructions for a data generation task
in the {domain} domain. Be specific and actionable. Return only the instructions, no preamble."""

    def __init__(self, spec: Spec, model: str = "gpt-4o-mini", temperature: float = 0.7):
        self.spec = spec
        self.model = model
        self.temperature = temperature
        self.llm_client = LLMClient(model=model, temperature=temperature)
        self.prompt = self._build_prompt()
        logger.info(f"\n{'='*60}\nGeneration Prompt:\n{'='*60}\n{self.prompt}\n{'='*60}\n")

    def _build_prompt(self) -> str:
        """Build final generation prompt using LLM to interpret constraints."""
        template = self.DOMAIN_TEMPLATES.get(self.spec.domain)
        if not template:
            raise ValueError(f"No template for domain: {self.spec.domain}")

        # Step 1: Use LLM to convert constraints dict to natural language
        if self.spec.constraints:
            constraints_dict_str = "\n".join(
                f"- {key}: {value}" for key, value in self.spec.constraints.items()
            )

            builder_prompt = self.CONSTRAINTS_BUILDER_PROMPT.format(
                constraints_dict=constraints_dict_str,
                domain=self.spec.domain.value
            )

            constraints_instructions = self.llm_client.generate(builder_prompt)
        else:
            constraints_instructions = ""

        # Step 2: Build final prompt with natural language constraints
        task_input_str = (
            str(self.spec.task_input)
            if isinstance(self.spec.task_input, str)
            else "\n".join(f"{k}: {v}" for k, v in self.spec.task_input.items())
        )

        return template.format(
            num_samples=self.spec.num_samples,
            task_input=task_input_str,
            constraints_instructions=constraints_instructions,
        )

    def generate(self) -> list[Sample]:
        """Generate samples using stored prompt."""
        raw_output = self.llm_client.generate(self.prompt)

        # Parse output into individual samples
        lines = [line.strip() for line in raw_output.strip().split("\n") if line.strip()]

        samples = []
        for i, line in enumerate(lines[:self.spec.num_samples]):
            # Remove numbering if present (e.g., "1. ", "1) ")
            content = line.lstrip("0123456789.-) ")

            sample = Sample(
                content=content,
                lineage=Lineage(
                    generator="naive",
                    generator_parameters={
                        "model": self.model,
                        "temperature": self.temperature,
                        "prompt": self.prompt,
                    }
                )
            )
            samples.append(sample)

        return samples

    def get_capabilities(self) -> dict[str, str]:
        """Return naive generator capabilities."""
        return {
            "name": "naive",
            "domain": self.spec.domain.value,
            "method": "direct_llm",
            "complexity": "low"
        }
