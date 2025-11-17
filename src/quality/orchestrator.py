from pathlib import Path
from typing import Any

import yaml

from src.core.spec import Sample, Spec
from src.quality.diversity_validator import DiversityValidator
from src.quality.llm_judge_validator import LLMJudgeValidator
from src.quality.semantic_validator import SemanticSimilarityValidator


class QualityAssessmentOrchestrator:
    """Orchestrates quality validation across multiple validators."""

    def __init__(self, config_path: str = "config/validators.yaml"):
        self.config = self._load_config(config_path)
        self.validators = self._initialize_validators()

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load validator configuration from YAML file."""
        path = Path(config_path)
        with open(path) as f:
            return yaml.safe_load(f)

    def _initialize_validators(self) -> dict[str, Any]:
        """Initialize enabled validators from config."""
        validators = {}

        # Semantic similarity validator (sample-level)
        if self.config.get("semantic_similarity", {}).get("enabled", False):
            validators["semantic_similarity"] = SemanticSimilarityValidator(
                self.config["semantic_similarity"]
            )

        # Diversity validator (batch-level)
        if self.config.get("diversity", {}).get("enabled", False):
            validators["diversity"] = DiversityValidator(
                self.config["diversity"]
            )

        # LLM judge validator (batch-level)
        if self.config.get("llm_judge", {}).get("enabled", False):
            validators["llm_judge"] = LLMJudgeValidator(self.config["llm_judge"])

        return validators

    def assess(self, samples: list[Sample], spec: Spec) -> list[Sample]:
        """Run all validators and populate quality_scores for each sample."""
        if not samples:
            return samples

        # Run sample-level validators
        for validator_name, validator in self.validators.items():
            if hasattr(validator, 'validate') and validator_name != "diversity":
                for sample in samples:
                    result = validator.validate(sample, spec)
                    sample.quality_scores[validator_name] = result["score"]

        # Run batch-level validators
        for validator_name, validator in self.validators.items():
            if hasattr(validator, 'validate_batch'):
                result = validator.validate_batch(samples, spec)
                # Only store if result is not None (validator actually implements batch validation)
                if result is not None:
                    # Store batch-level score in each sample
                    for sample in samples:
                        sample.quality_scores[f"{validator_name}_batch"] = result["score"]

        return samples

    def filter_failing_samples(self, samples: list[Sample]) -> list[Sample]:
        """Filter out samples that failed validation (passed=False)."""
        filtered = []
        for sample in samples:
            # Check if all validators passed
            passed = True
            for validator_name, validator in self.validators.items():
                score = sample.quality_scores.get(validator_name)
                if score is not None and score < validator.threshold:
                    passed = False
                    break
            if passed:
                filtered.append(sample)
        return filtered
