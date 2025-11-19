from typing import Any

from src.core.models import Sample, Spec
from src.quality.deduplication_validator import DeduplicationValidator
from src.quality.diversity_validator import DiversityValidator
from src.quality.entailment_validator import EntailmentValidator
from src.quality.llm_judge_validator import LLMJudgeValidator
from src.quality.rule_filters import JSONSchemaValidator
from src.quality.similarity_validator import SimilarityValidator


class QualityAssessmentOrchestrator:
    """Orchestrates quality validation across multiple validators."""

    def __init__(self, config: dict[str, Any] | None = None):
        """Initialize orchestrator with an explicit validator configuration.

        Args:
            config: Validator configuration dictionary, typically coming from
                `Spec.validators`. If None or empty, no validators are enabled.
        """
        # Default to empty config (all validators disabled) when not provided
        self.config: dict[str, Any] = config or {}
        self.validators = self._initialize_validators()

    def _initialize_validators(self) -> dict[str, Any]:
        """Initialize enabled validators from config."""
        validators = {}

        # Deduplication validator (sample-level) - run first to catch duplicates early
        if self.config.get("deduplication", {}).get("enabled", False):
            validators["deduplication"] = DeduplicationValidator(self.config["deduplication"])

        # Similarity validator (sample-level)
        if self.config.get("similarity", {}).get("enabled", False):
            validators["similarity"] = SimilarityValidator(self.config["similarity"])

        # Entailment validator (sample-level)
        if self.config.get("entailment", {}).get("enabled", False):
            validators["entailment"] = EntailmentValidator(self.config["entailment"])

        # JSON schema validator (sample-level)
        if self.config.get("json_schema", {}).get("enabled", False):
            validators["json_schema"] = JSONSchemaValidator(self.config["json_schema"])

        # Diversity validator (batch-level)
        if self.config.get("diversity", {}).get("enabled", False):
            validators["diversity"] = DiversityValidator(self.config["diversity"])

        # LLM judge validator (batch-level)
        if self.config.get("llm_judge", {}).get("enabled", False):
            validators["llm_judge"] = LLMJudgeValidator(self.config["llm_judge"])

        return validators

    def _sample_passed_sample_level_validators(self, sample: Sample) -> bool:
        """Check if sample passed all sample-level validators."""
        validation_results = sample.metadata.get("validation_results", {})
        for validator_name, validator in self.validators.items():
            if validator.is_sample_level():
                result = validation_results.get(validator_name)
                if result is not None and not result["passed"]:
                    return False
        return True

    def assess(self, samples: list[Sample], spec: Spec) -> list[Sample]:
        """Run all validators and populate quality_scores for each sample."""
        if not samples:
            return samples

        # Initialize validation_results in metadata if not present
        for sample in samples:
            if "validation_results" not in sample.metadata:
                sample.metadata["validation_results"] = {}

        # TODO: Optimize embedding computation with caching - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/23
        # Run sample-level validators
        for validator_name, validator in self.validators.items():
            if validator.is_sample_level():
                for sample in samples:
                    result = validator.validate(sample, spec)
                    # Store score for backward compatibility
                    sample.quality_scores[validator_name] = result.score
                    # Store full ValidationResult for proper pass/fail tracking
                    sample.metadata["validation_results"][validator_name] = result.model_dump()

        # Filter samples that passed sample-level validation before running batch-level validators
        samples_passed_sample_level = [
            sample for sample in samples if self._sample_passed_sample_level_validators(sample)
        ]

        # Run batch-level validators only on samples that passed sample-level validation
        for validator_name, validator in self.validators.items():
            if validator.is_batch_level():
                # Only compute batch-level metrics (e.g., diversity) on samples that passed
                # sample-level validation. This ensures batch metrics reflect only samples
                # that will be kept, not rejected samples.
                # Note: len(samples_passed_sample_level) should be <= len(samples)
                result = validator.validate_batch(samples_passed_sample_level, spec)

                per_sample_data = {}
                metadata = result.metadata or {}
                if "per_sample_evaluations" in metadata:
                    per_sample_data = {
                        entry.get("sample_index"): entry
                        for entry in metadata["per_sample_evaluations"]
                        if isinstance(entry, dict) and "sample_index" in entry
                    }

                # Store batch-level result only in samples that passed sample-level validation,
                # but prefer per-sample quality scores when the validator provides them.
                for idx, sample in enumerate(samples_passed_sample_level):
                    sample.metadata["validation_results"][validator_name] = result.model_dump()

                    per_sample_entry = per_sample_data.get(idx)
                    if per_sample_entry and "quality_level" in per_sample_entry:
                        # Normalize 1-5 quality level into 0-1 range for consistency
                        level = per_sample_entry["quality_level"]
                        sample.quality_scores[validator_name] = max(0.0, min(1.0, level / 5.0))
                    else:
                        sample.quality_scores[validator_name] = result.score

        return samples

    def filter_failing_samples(self, samples: list[Sample]) -> list[Sample]:
        """Filter out samples that failed validation (passed=False)."""
        filtered = []
        for sample in samples:
            # Check if all validators passed using stored ValidationResult.passed
            all_passed = True
            validation_results = sample.metadata.get("validation_results", {})

            for validator_name in self.validators.keys():
                result = validation_results.get(validator_name)
                if result is not None and not result["passed"]:
                    all_passed = False
                    break

            if all_passed:
                filtered.append(sample)
        return filtered
