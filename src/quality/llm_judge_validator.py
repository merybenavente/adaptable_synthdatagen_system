import json

from src.core.base_validator import BaseValidator, ValidationResult
from src.core.spec import Sample, Spec
from src.utils.llm_client import LLMClient


class LLMJudgeValidator(BaseValidator):
    """Batch-level LLM judge that evaluates sample quality with 5-level scoring."""

    QUALITY_LEVEL_DESCRIPTIONS = {
        5: "Excellent - Exceptional quality, exceeds all requirements",
        4: "Good - High quality, exceeds most requirements",
        3: "Average - Acceptable quality, meets requirements adequately",
        2: "Below Average - Meets some requirements but has significant issues",
        1: "Poor - Fails to meet basic requirements",
    }

    CRITERIA_BUILDER_PROMPT = """You are helping define quality criteria for evaluating synthetic data.

Given this generation specification:
Domain: {domain}
Task Input: {task_input}
Constraints: {constraints}
User Guidance: {user_guidance}

Write a concise list (1-3 bullet points) of the most important quality criteria for evaluating generated samples in this domain.
Focus on what makes a sample high-quality vs low-quality for this specific task.
Be specific and actionable. Return only the criteria list, no preamble."""

    EVALUATION_SYSTEM_PROMPT = """You are a quality judge for synthetic data generation.
Evaluate samples based on the criteria below and assign a quality level (1-5) to each.

QUALITY LEVELS:
{quality_levels}

EVALUATION CRITERIA:
{criteria}

Return your evaluation as valid JSON with this exact structure:
{{
  "reasoning": "Brief overall assessment of batch quality patterns",
  "aggregate_quality_level": <average quality level as float>,
  "samples": [
    {{"sample_index": 0, "quality_level": <1-5>, "justification": "Brief reason"}},
    {{"sample_index": 1, "quality_level": <1-5>, "justification": "Brief reason"}}
  ]
}}"""

    EVALUATION_USER_PROMPT = """Domain: {domain}
Original Task: {task_input}
{constraints_text}

Evaluate these {num_samples} generated samples:
{samples_text}

Provide your evaluation in the JSON format specified."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.model = config.get("model", "gpt-4o-mini")
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 2000)
        self.user_guidance = config.get("quality_criteria", "")

        self.llm_client = LLMClient(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    def is_sample_level(self) -> bool:
        """Return False - this validator does not operate on individual samples."""
        return False

    def is_batch_level(self) -> bool:
        """Return True - this validator operates on batches."""
        return True

    def validate_batch(self, samples: list[Sample], spec: Spec) -> ValidationResult:
        """Evaluate batch quality using LLM judge with 5-level scoring."""
        if not samples:
            return ValidationResult(score=0.0, passed=False, metadata={"error": "Empty batch"})

        # Build quality criteria dynamically from spec
        criteria = self._build_quality_criteria(spec)

        # Build evaluation prompt
        evaluation_prompt = self._build_evaluation_prompt(samples, spec, criteria)

        # Call LLM judge (with retry)
        response = self._call_llm_with_retry(evaluation_prompt)

        if response is None:
            # LLM call failed after retry
            return ValidationResult(
                score=0.0,
                passed=False,
                metadata={"error": "LLM judge failed to return valid response"}
            )

        # Parse quality levels and convert to scores
        aggregate_score, metadata = self._parse_response(response, len(samples))

        # Determine if batch passes threshold
        passed = aggregate_score >= self.threshold

        return ValidationResult(score=aggregate_score, passed=passed, metadata=metadata)

    def _build_quality_criteria(self, spec: Spec) -> str:
        """Generate domain-specific quality criteria from spec using LLM."""
        # Format constraints for prompt
        constraints_str = self._format_constraints(spec.constraints)

        # Format task input
        task_input_str = self._format_task_input(spec.task_input)

        # Build criteria using LLM
        prompt = self.CRITERIA_BUILDER_PROMPT.format(
            domain=spec.domain.value,
            task_input=task_input_str,
            constraints=constraints_str if constraints_str else "None specified",
            user_guidance=self.user_guidance if self.user_guidance else "None provided",
        )

        try:
            criteria = self.llm_client.generate(prompt)
            return criteria
        except Exception:
            # Fallback to user guidance if LLM fails
            return self.user_guidance if self.user_guidance else "General quality and coherence"

    def _build_evaluation_prompt(self, samples: list[Sample], spec: Spec, criteria: str) -> str:
        """Build complete evaluation prompt for LLM judge."""
        # Format quality levels for system prompt
        quality_levels_text = "\n".join(
            f"- Level {level}: {desc}"
            for level, desc in sorted(self.QUALITY_LEVEL_DESCRIPTIONS.items(), reverse=True)
        )

        # Build system prompt
        system_prompt = self.EVALUATION_SYSTEM_PROMPT.format(
            quality_levels=quality_levels_text,
            criteria=criteria,
        )

        # Format samples for user prompt
        samples_text = "\n".join(
            f"{i+1}. {sample.content}" for i, sample in enumerate(samples)
        )

        # Format constraints
        constraints_str = self._format_constraints(spec.constraints)
        constraints_text = f"Constraints: {constraints_str}" if constraints_str else ""

        # Format task input
        task_input_str = self._format_task_input(spec.task_input)

        # Build user prompt
        user_prompt = self.EVALUATION_USER_PROMPT.format(
            domain=spec.domain.value,
            task_input=task_input_str,
            constraints_text=constraints_text,
            num_samples=len(samples),
            samples_text=samples_text,
        )

        # Combine into full prompt (LLMClient will handle system/user split)
        # For now, return as single prompt with clear sections
        return f"{system_prompt}\n\n{user_prompt}"

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 1) -> dict | None:
        """Call LLM and parse JSON response with retry on failure."""
        for attempt in range(max_retries + 1):
            try:
                # Extract system prompt (everything before the evaluation section)
                parts = prompt.split("\n\nDomain:")
                if len(parts) == 2:
                    system_prompt = parts[0]
                    user_prompt = "Domain:" + parts[1]
                else:
                    system_prompt = None
                    user_prompt = prompt

                # Call LLM
                response_text = self.llm_client.generate(
                    user_prompt,
                    system_prompt=system_prompt,
                )

                # Parse JSON
                # Try to extract JSON if wrapped in markdown code blocks
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    response_text = response_text[json_start:json_end].strip()

                response = json.loads(response_text)
                return response

            except (json.JSONDecodeError, KeyError, ValueError) as e:
                if attempt < max_retries:
                    # Retry with stricter prompt
                    continue
                else:
                    # Final attempt failed
                    return None

        return None

    def _parse_response(self, response: dict, num_samples: int) -> tuple[float, dict]:
        """Parse LLM response and convert quality levels to scores."""
        # Extract aggregate quality level
        aggregate_level = response.get("aggregate_quality_level", 3.0)

        # Convert quality level (1-5) to score (0.2-1.0)
        aggregate_score = aggregate_level / 5.0

        # Build metadata with full response
        metadata = {
            "reasoning": response.get("reasoning", ""),
            "aggregate_quality_level": aggregate_level,
            "per_sample_evaluations": response.get("samples", []),
            "criteria_used": "Dynamic criteria based on spec",
        }

        return aggregate_score, metadata

    def _format_constraints(self, constraints: dict) -> str:
        """Format constraints dict as readable text."""
        if not constraints:
            return ""
        return "\n".join(f"  - {key}: {value}" for key, value in constraints.items())

    def _format_task_input(self, task_input: str | dict) -> str:
        """Format task_input for display in prompt."""
        if isinstance(task_input, str):
            return task_input
        elif isinstance(task_input, dict):
            # Handle ML augmentation format
            if "original_input" in task_input:
                original = task_input.get("original_input", "")
                expected = task_input.get("expected_output", "")
                return f"{original} (expected output: {expected})"
            # Handle generic dict
            return "\n".join(f"  {k}: {v}" for k, v in task_input.items())
        return str(task_input)
