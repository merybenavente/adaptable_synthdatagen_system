from src.core.base_generator import BaseGenerator
from src.core.generator_types import GeneratorType
from src.core.models import Domain, GenerationContext, GenerationPlan, Lineage, Sample
from src.core.type_guards import is_ml_augmentation_dict
from src.utils.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# TODO: fix this hack for the demo once we address https://github.com/merybenavente/adaptable_synthdatagen_system/issues/21)
# Track which generator configurations have already printed their template
_printed_templates: set[str] = set()


class NaiveGenerator(BaseGenerator):
    """Naive generator that directly calls LLM for generation."""

    DOMAIN_TEMPLATES = {
        Domain.TASK_REWRITE: """Generate {num_samples} variants of a task instruction:

{constraints_instructions}

Return only the variants, one per line, numbered 1-{num_samples}.

Task: \"{task_input}\"""",
        Domain.QA_PAIRS: """Generate {num_samples} question-answer pairs based on:

Topic: {task_input}

{constraints_instructions}

Return in JSON format with 'question' and 'answer' fields.""",
    }

    CONSTRAINTS_BUILDER_PROMPT = """Given these generation constraints as a dictionary:

{constraints_dict}

Convert them into clear, natural language instructions for a data generation task
in the {domain} domain. Be specific and actionable. Return only the instructions, no preamble."""

    def __init__(self, context: GenerationContext, plan: GenerationPlan):
        self.context = context
        self.plan = plan

        # Extract adaptive parameters from plan
        self.temperature = plan.parameters.get("temperature", 0.7)
        self.top_p = plan.parameters.get("top_p", 1.0)
        self.max_tokens = plan.parameters.get("max_tokens", None)
        self.model = plan.parameters.get("model", "gpt-4o-mini")

        self.llm_client = LLMClient(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )
        self.prompt = self._build_prompt()
        logger.info(f"\n{'=' * 60}\nGeneration Prompt:\n{'=' * 60}\n{self.prompt}\n{'=' * 60}\n")

        self.task_input = self._format_task_input(self.context.task_input)

        # Print prompt template only once per unique configuration
        # Key based on static config (domain + constraints from context only, not plan params)
        constraints_key = (
            hash(tuple(sorted(self.context.constraints.items())))
            if self.context.constraints
            else 0
        )
        template_key = f"{self.context.domain.value}_{constraints_key}"

        if template_key not in _printed_templates:
            # Create canonical template for display (replace batch_size with placeholder)
            display_template = self.prompt.replace(str(self.plan.batch_size), "{num_samples}")
            sep = "=" * 60
            logger.debug(
                f"\n\n{sep}\nPrompt Template:\n{sep}\n{display_template}\n{sep}\n"
            )
            _printed_templates.add(template_key)

    def _format_task_input(self, task_input) -> str:
        """Format task_input for display or prompt generation."""
        if is_ml_augmentation_dict(task_input):
            return task_input.get("original_input", "")

        if isinstance(task_input, str):
            return task_input

        return "\n".join(f"{k}: {v}" for k, v in task_input.items())

    def _build_prompt(self) -> str:
        """Build final generation prompt using LLM to interpret constraints."""
        # Step 1: Merge context constraints and plan parameters, filter technical params
        technical_params = {'temperature', 'top_p', 'max_tokens', 'model', 'domain'}
        all_constraints = {**self.context.constraints, **self.plan.parameters}
        content_constraints = {
            k: v for k, v in all_constraints.items()
            if k not in technical_params
        }

        # Step 2: Use LLM to convert content constraints to natural language
        if content_constraints:
            constraints_dict_str = "\n".join(
                f"- {key}: {value}" for key, value in content_constraints.items()
            )

            builder_prompt = self.CONSTRAINTS_BUILDER_PROMPT.format(
                constraints_dict=constraints_dict_str,
                domain=self.context.domain.value
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

        # Step 3: Build prompt based on task_input structure
        if self.context.domain == Domain.TASK_REWRITE:
            return self._build_task_rewrite_prompt(constraints_instructions)

        # For other domains, use standard template
        template = self.DOMAIN_TEMPLATES.get(self.context.domain)
        if not template:
            raise ValueError(f"No template for domain: {self.context.domain}")

        task_input_str = self._format_task_input(self.context.task_input)

        return template.format(
            num_samples=self.plan.batch_size,
            task_input=task_input_str,
            constraints_instructions=constraints_instructions,
        )

    def _build_task_rewrite_prompt(self, constraints_instructions: str) -> str:
        """Build task_rewrite prompt dynamically based on task_input structure."""
        task_input = self.context.task_input

        # ML augmentation mode: dict with expected_output (using type guard)
        if is_ml_augmentation_dict(task_input):
            original_input = task_input.get("original_input", "")
            expected_output = task_input.get("expected_output", "")
            task_description = task_input.get("task_description", "")
            context = task_input.get("context", "")
            examples = task_input.get("examples", [])

            # Build context section
            context_section = f"\nContext: {context}" if context else ""

            # Build examples section (few-shot)
            examples_section = ""
            if examples:
                examples_section = "\n\nExamples:"
                for ex in examples:
                    ex_input = ex.get("input", "")
                    ex_output = ex.get("output", "")
                    examples_section += f"\n  Input: {ex_input} → Output: {ex_output}"

            return f"""Generate {self.plan.batch_size} paraphrases for ML data augmentation.

Original Input: {original_input}
Expected Output: {expected_output}
Task: {task_description}{context_section}{examples_section}

{constraints_instructions}

Return only the paraphrased inputs, one per line, numbered 1-{self.plan.batch_size}."""

        # Simple paraphrase mode: string or dict without expected_output
        task_input_str = self._format_task_input(task_input)

        template = self.DOMAIN_TEMPLATES.get(Domain.TASK_REWRITE)
        return template.format(
            num_samples=self.plan.batch_size,
            task_input=task_input_str,
            constraints_instructions=constraints_instructions,
        )

    def generate(self) -> list[Sample]:
        """Generate samples using stored prompt."""
        # Create original sample from input task (no lineage as it's not generated)
        original_sample = Sample(
            content=self._format_task_input(self.context.task_input),
            lineage=None,
        )

        # Generate evolved variants
        raw_output = self.llm_client.generate(self.prompt)

        # Parse output into individual samples
        lines = [line.strip() for line in raw_output.strip().split("\n") if line.strip()]

        samples = []
        for i, line in enumerate(lines[ :self.plan.batch_size]):
            # Remove numbering if present (e.g., "1. ", "1) ")
            content = line.lstrip("0123456789.-) ")

            sample = Sample(
                content=content,
                lineage=Lineage(
                    original_sample=original_sample.content,
                    num_of_evolutions=1,
                    parent_id=original_sample.id,
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
            "domain": self.context.domain.value,
            "method": "direct_llm",
            "complexity": "low",
        }
