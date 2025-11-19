from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

from src.core.base_generator import BaseGenerator
from src.core.generator_types import GeneratorType
from src.core.models import GenerationContext, GenerationPlan, Lineage, Sample
from src.core.type_guards import is_ml_augmentation_dict
from src.utils.llm_client import LLMClient
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# TODO: fix this hack for the demo once we address https://github.com/merybenavente/adaptable_synthdatagen_system/issues/21)
# Cache criteria to avoid regenerating and printing duplicates for same spec configuration
_logged_planner_prompts: set[str] = set()
_logged_generation_prompts: set[str] = set()


@dataclass
class PromptPlan:
    """Structured prompt and parsing instructions derived from context/spec."""

    system_prompt: str
    user_prompt: str
    parsing_strategy: str = "list"
    schema: dict[str, Any] | None = None
    notes: str | None = None

    @classmethod
    def from_dict(cls, payload: dict[str, Any], fallback_system: str) -> PromptPlan:
        return cls(
            system_prompt=payload.get("system_prompt") or fallback_system,
            user_prompt=payload.get("user_prompt") or payload.get("prompt", ""),
            parsing_strategy=payload.get("parsing_strategy", "list"),
            schema=payload.get("schema"),
            notes=payload.get("notes"),
        )


class NaiveGenerator(BaseGenerator):
    """Auto-planning generator that derives prompts/parsers from the spec."""

    PROMPT_PLANNER_SYSTEM_PROMPT = (
        "You are an expert prompt engineer for synthetic data generation pipelines. "
        "Given a structured specification, you design the optimal system prompt, user prompt, "
        "expected output format, and parsing strategy. Always return VALID JSON with the fields: "
        "system_prompt (string), user_prompt (string), parsing_strategy (json|jsonl|list|text), "
        "schema (optional JSON schema), notes (optional string)."
    )

    PROMPT_PLANNER_TEMPLATE = """You will receive a JSON specification describing the \
generation context.
- Study the input carefully.
- Decide on the minimal yet complete instructions needed to generate {batch_size} samples.
- If the spec requests structured data, prefer json parsing_strategy and include schema.
- For unstructured text, use list parsing_strategy with clear formatting requirements.
- The output must be a valid JSON array, no matter how complex the inner format.
- Encourage on the prompt to be imaginative for data diversity.

OUTPUT FORMAT EXAMPLES (structure only - generate NEW content, do NOT copy):

Example 1 - Structured objects:
[
    {{"review_text": "These headphones are amazing!", "sentiment": "positive"}},
    {{"review_text": "Terrible sound quality", "sentiment": "negative"}}
]

Example 2 - Simple strings:
[
    "First generated sample",
    "Second generated sample"
]

- Describe in the prompt the desired format of each output, mimicking the structure shown above.
- Make it EXPLICIT that examples show structure only and the LLM must generate completely new
  content.
- Include a format example in the prompt with a clear note that it's illustrative.

Specification:
{spec_json}
"""

    DEFAULT_SYSTEM_PROMPT = (
        "You are a meticulous synthetic data generator. Follow constraints exactly, "
        "avoid preambles, and return only the requested structured output."
        "Generate only the examples, nothing else, following the schema."
    )

    def __init__(self, context: GenerationContext, plan: GenerationPlan):
        self.context = context
        self.plan = plan

        self.temperature = plan.parameters.get("temperature", 0.7)
        self.top_p = plan.parameters.get("top_p", 1.0)
        self.max_tokens = plan.parameters.get("max_tokens")
        self.model = plan.parameters.get("model", "gpt-4o-mini")
        self.constraints_max_tokens = plan.parameters.get("constraints_max_tokens", 768)

        self.llm_client = LLMClient(
            model=self.model,
            temperature=self.temperature,
            top_p=self.top_p,
            max_tokens=self.max_tokens,
        )

        self.task_input_text = self._format_task_input(self.context.task_input)
        self.prompt_spec = self._build_prompt_spec()
        self.plan_signature = self._compute_plan_signature()
        self.prompt_plan = self._build_prompt_plan()

        logger.debug(
            "Prompt plan derived for domain=%s | batch=%s | strategy=%s\n%s",
            self.context.domain,
            self.plan.batch_size,
            self.prompt_plan.parsing_strategy,
            json.dumps(
                {
                    "system_prompt": self.prompt_plan.system_prompt,
                    "user_prompt": self.prompt_plan.user_prompt,
                    "parsing_strategy": self.prompt_plan.parsing_strategy,
                    "schema": self.prompt_plan.schema,
                    "notes": self.prompt_plan.notes,
                },
                indent=2,
            ),
        )

    # -------------------------------------------------------------------------
    # Planning helpers
    # -------------------------------------------------------------------------
    def _build_prompt_spec(self) -> dict[str, Any]:
        """Assemble serializable spec for the planner."""
        sanitized_constraints = self._sanitize_constraints(self.context.constraints)
        plan_parameters = {
            k: v
            for k, v in self.plan.parameters.items()
            if k not in {"temperature", "top_p", "max_tokens", "model", "constraints_max_tokens"}
        }

        return {
            "domain": self.context.domain,
            "goal": self.task_input_text,
            "raw_task_input": self.context.task_input,
            "batch_size": self.plan.batch_size,
            "constraints": sanitized_constraints,
            "plan_parameters": plan_parameters,
            "progress": self.context.progress.model_dump(),
        }

    def _compute_plan_signature(self) -> str:
        """Stable signature to avoid logging duplicate prompts."""
        # Exclude progress from signature as it changes every iteration
        spec_without_progress = {
            k: v for k, v in self.prompt_spec.items()
            if k not in ["progress", "goal"]
        }
        signature_payload = {
            "spec": spec_without_progress,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        signature_str = json.dumps(signature_payload, sort_keys=True, default=str)
        return hashlib.sha256(signature_str.encode("utf-8")).hexdigest()

    def _build_prompt_plan(self) -> PromptPlan:
        """Call planner LLM once to derive prompts + parsing instructions."""
        spec_json = json.dumps(
            {k: v for k, v in self.prompt_spec.items() if k not in ["progress", "goal"]},
            indent=2, default=str
        )
        planner_prompt = self.PROMPT_PLANNER_TEMPLATE.format(
            spec_json=spec_json, batch_size=self.plan.batch_size
        )

        should_log = self.plan_signature not in _logged_planner_prompts

        if should_log:
            logger.info(
                "Prompt planner request | domain=%s | batch=%s\n"
                "System Prompt:\n%s\n\nPlanner Input:\n%s\n",
                self.context.domain,
                self.plan.batch_size,
                self.PROMPT_PLANNER_SYSTEM_PROMPT,
                planner_prompt,
            )

        try:
            raw_plan = self.llm_client.generate(
                prompt=planner_prompt,
                system_prompt=self.PROMPT_PLANNER_SYSTEM_PROMPT,
                max_tokens=self.constraints_max_tokens,
            )
            if should_log:
                logger.info("Prompt planner raw response:\n%s\n", raw_plan)
                _logged_planner_prompts.add(self.plan_signature)
            plan_payload = self._coerce_json(raw_plan)
            return PromptPlan.from_dict(plan_payload, fallback_system=self.DEFAULT_SYSTEM_PROMPT)
        except Exception as exc:
            logger.warning("Prompt planner failed (%s). Falling back to default plan.", exc)
            return self._default_prompt_plan()

    def _default_prompt_plan(self) -> PromptPlan:
        """Fallback deterministic plan if the planner cannot respond."""
        constraints_text = "\n".join(
            f"- {key}: {value}" for key, value in self.prompt_spec.get("constraints", {}).items()
        ) or "None provided."

        user_prompt = (
            f"Goal:\n{self.task_input_text}\n\nConstraints:\n{constraints_text}\n\n"
            f"Generate {self.plan.batch_size} distinct samples. "
            "Return a JSON array named samples where each element is an object with a "
            "'content' field."
        )

        return PromptPlan(
            system_prompt=self.DEFAULT_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            parsing_strategy="json",
            schema={
                "type": "object",
                "properties": {
                    "samples": {
                        "type": "array",
                        "items": {"type": "object"},
                        "minItems": self.plan.batch_size,
                    }
                },
                "required": ["samples"],
            },
            notes="Auto fallback plan",
        )

    # TODO: move shared method to base class
    @staticmethod
    def _sanitize_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
        """Filter out technical validation constraints and ensure JSON-serializable."""
        # TODO: Replace manual filtering with intelligent AI-based parsing
        # https://github.com/merybenavente/adaptable_synthdatagen_system/issues/30

        # Technical constraints that should NOT go in the generation prompt
        technical_constraints = {
            # Sample-level validation
            "semantic_similarity",
            "semantic_similarity_min",
            "semantic_similarity_max",
            "similarity_threshold",
            "min_length",
            "max_length",
            "min_tokens",
            "max_tokens",
            "embedding_model",
            "quality_threshold",
            "uniqueness_threshold",
            # Dataset-level validation
            "maintain_label_distribution",
            "label_distribution",
            "min_diversity",
            "max_repetition_rate",
            # Generator-specific (not prompt content)
            "grammar_path",
        }

        sanitized = {}
        for key, value in (constraints or {}).items():
            # Skip technical validation constraints
            if key in technical_constraints:
                continue

            if isinstance(value, (str, int, float, bool)) or value is None:
                sanitized[key] = value
            else:
                sanitized[key] = json.loads(json.dumps(value, default=str))
        return sanitized

    @staticmethod
    def _coerce_json(text: str) -> dict[str, Any]:
        """Parse JSON response, handling stray code fences and extra text."""
        cleaned = text.strip()

        # Handle code fences
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?", "", cleaned, count=1).strip()
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3].strip()

        # Try direct parsing first
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        # Try to find JSON object in the text
        match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        raise ValueError(f"Could not extract valid JSON from response: {text[:200]}")

    # -------------------------------------------------------------------------
    # Generation + parsing
    # -------------------------------------------------------------------------
    def generate(self) -> list[Sample]:
        """Generate samples using the derived prompt plan."""
        original_sample = Sample(content=self.task_input_text, lineage=None)

        if self.plan_signature not in _logged_generation_prompts:
            logger.info(
                "Executing prompt plan | domain=%s | strategy=%s\n"
                "System Prompt:\n%s\n\nUser Prompt:\n%s\n",
                self.context.domain,
                self.prompt_plan.parsing_strategy,
                self.prompt_plan.system_prompt,
                self.prompt_plan.user_prompt,
            )
            _logged_generation_prompts.add(self.plan_signature)

        raw_output = self.llm_client.generate(
            prompt=self.prompt_plan.user_prompt,
            system_prompt=self.prompt_plan.system_prompt,
        )

        parsed_payloads = self._parse_output(raw_output)
        if len(parsed_payloads) < self.plan.batch_size:
            raise ValueError(
                f"Only parsed {len(parsed_payloads)} samples, expected {self.plan.batch_size}"
            )

        samples: list[Sample] = []
        for payload in parsed_payloads[: self.plan.batch_size]:
            # Populate metadata with content fields if content is a dict
            # (enables field-based deduplication like check_field: "plant")
            metadata = {"timestamp": datetime.utcnow().isoformat()}
            if isinstance(payload, dict):
                metadata.update(payload)

            sample = Sample(
                content=payload,
                metadata=metadata,
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
                        "prompt_plan": asdict(self.prompt_plan),
                    },
                ),
            )
            samples.append(sample)

        return samples

    def _parse_output(self, raw_output: str) -> list[str | dict[str, Any]]:
        """Parse LLM output - tries JSON first, falls back to line split."""
        cleaned = self._strip_code_fences(raw_output)

        # Try JSON parsing first (standard double-quoted JSON)
        try:
            payload = json.loads(cleaned)
            if isinstance(payload, list):
                return payload
            if isinstance(payload, dict):
                # Extract "samples" key if present (common LLM pattern), otherwise wrap dict
                if "samples" in payload and isinstance(payload["samples"], list):
                    return payload["samples"]
                return [payload]
            # Handle primitives (string, number, bool, null) - wrap in array
            return [payload]
        except json.JSONDecodeError:
            # Try parsing as Python literal (handles single-quoted JSON from LLM)
            try:
                import ast
                payload = ast.literal_eval(cleaned)
                if isinstance(payload, list):
                    return payload
                if isinstance(payload, dict):
                    if "samples" in payload and isinstance(payload["samples"], list):
                        return payload["samples"]
                    return [payload]
                return [payload]
            except (ValueError, SyntaxError):
                pass

        # Fallback: split by lines, preserve content as-is (no numbered prefix removal)
        lines = [line.strip() for line in cleaned.strip().splitlines() if line.strip()]
        return lines

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        if text.strip().startswith("```"):
            inner = re.sub(r"^```(?:json|JSON)?", "", text.strip(), count=1)
            if inner.endswith("```"):
                inner = inner[:-3]
            return inner.strip()
        return text.strip()

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    def _format_task_input(self, task_input: Any) -> str:
        """Format task_input for planner readability."""
        if is_ml_augmentation_dict(task_input):
            # Preserve full ML augmentation context so the planner can design
            # task-aware prompts (e.g., generate new input-output pairs instead
            # of just paraphrasing the original input).
            payload: dict[str, Any] = {
                "task_type": "ml_augmentation",
                "original_example": {
                    "input": task_input.get("original_input"),
                    "output": task_input.get("expected_output"),
                },
                "task_description": task_input.get("task_description"),
                "context": task_input.get("context"),
                "examples": task_input.get("examples"),
                "augmentation_goal": (
                    "Generate new input-output training examples following the same "
                    "structure as the original_example and examples. Do not merely "
                    "paraphrase the input text; instead, create new, diverse "
                    "examples that fit the described task."
                ),
            }

            return json.dumps(payload, indent=2, ensure_ascii=False)

        if isinstance(task_input, str):
            return task_input

        try:
            return json.dumps(task_input, indent=2, ensure_ascii=False)
        except TypeError:
            return "\n".join(f"{k}: {v}" for k, v in task_input.items())

    def get_capabilities(self) -> dict[str, str]:
        return {
            "name": GeneratorType.NAIVE,
            "domain": self.context.domain,
            "method": "direct_llm_auto_prompt",
            "complexity": "medium",
        }
