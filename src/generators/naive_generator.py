from src.core.base_generator import BaseGenerator
from src.core.generator_types import GeneratorType
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

    def __init__(self, spec: Spec):
        self.spec = spec

        # Extract adaptive parameters from constraints
        constraints = spec.constraints or {}
        self.temperature = constraints.get("temperature", 0.7)
        self.top_p = constraints.get("top_p", 1.0)
        self.max_tokens = constraints.get("max_tokens", None)
        self.model = constraints.get("model", "gpt-4o-mini")

        self.llm_client = LLMClient(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        self.prompt = self._build_prompt()
        logger.info(f"\n{'=' * 60}\nGeneration Prompt:\n{'=' * 60}\n{self.prompt}\n{'=' * 60}\n")

    def _build_prompt(self) -> str:
        """Build final generation prompt using LLM to interpret constraints."""
        template = self.DOMAIN_TEMPLATES.get(self.spec.domain)
        if not template:
            raise ValueError(f"No template for domain: {self.spec.domain}")

        # Step 1: Filter out technical/LLM parameters from constraints
        technical_params = {"temperature", "top_p", "max_tokens", "model", "domain"}
        content_constraints = {
            k: v for k, v in (self.spec.constraints or {}).items() if k not in technical_params
        }

        # Step 2: Use LLM to convert content constraints to natural language
        if content_constraints:
            constraints_dict_str = "\n".join(
                f"- {key}: {value}" for key, value in content_constraints.items()
            )

            builder_prompt = self.CONSTRAINTS_BUILDER_PROMPT.format(
                constraints_dict=constraints_dict_str, domain=self.spec.domain.value
            )

            try:
                constraints_instructions = self.llm_client.generate(builder_prompt)
            except Exception as e:
                logger.warning(
                    f"Failed to generate constraint instructions: {e}. Using empty constraints."
                )
                constraints_instructions = ""
        else:
            constraints_instructions = ""

        # Step 3: Build final prompt with natural language constraints
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
        # Create original sample from input task (generation 0)
        task_input_str = (
            str(self.spec.task_input)
            if isinstance(self.spec.task_input, str)
            else str(self.spec.task_input)
        )

        original_sample = Sample(
            content=task_input_str,
            lineage=Lineage(
                original_sample=None,  # Root has no ancestor
                num_of_evolutions=0,  # This is generation 0
                parent_id=None,  # Root has no parent
                generator=GeneratorType.NAIVE,
                generator_parameters={"source": "input_task"},
            ),
        )

        # Generate evolved variants
        raw_output = self.llm_client.generate(self.prompt)

        # Parse output into individual samples
        lines = [line.strip() for line in raw_output.strip().split("\n") if line.strip()]

        samples = []
        for i, line in enumerate(lines[: self.spec.num_samples]):
            # Remove numbering if present (e.g., "1. ", "1) ")
            content = line.lstrip("0123456789.-) ")

            sample = Sample(
                content=content,
                lineage=Lineage(
                    original_sample=original_sample.id,  # Track root ancestor
                    num_of_evolutions=1,  # First evolution from original
                    parent_id=original_sample.id,  # Immediate parent is original
                    generator=GeneratorType.NAIVE,
                    generator_parameters={
                        "model": self.model,
                        "temperature": self.temperature,
                        "top_p": self.top_p,
                        "max_tokens": self.max_tokens,
                        "prompt": self.prompt,
                    },
                ),
            )
            samples.append(sample)

        return samples

    def get_capabilities(self) -> dict[str, str]:
        """Return naive generator capabilities."""
        return {
            "name": GeneratorType.NAIVE,
            "domain": self.spec.domain.value,
            "method": "direct_llm",
            "complexity": "low",
        }
