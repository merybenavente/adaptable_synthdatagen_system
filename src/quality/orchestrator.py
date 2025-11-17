from pathlib import Path
from typing import Any

import yaml

from src.core.models import Sample, Spec
from src.quality.diversity_validator import DiversityValidator
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
            validators["diversity"] = DiversityValidator(self.config["diversity"])

        return validators

    def assess(self, samples: list[Sample], spec: Spec) -> list[Sample]:
        """Run all validators and populate quality_scores for each sample."""
        if not samples:
            return samples

        # Initialize validation_results in metadata if not present
        for sample in samples:
            if "validation_results" not in sample.metadata:
                sample.metadata["validation_results"] = {}

        # Run sample-level validators
        for validator_name, validator in self.validators.items():
            if validator.is_sample_level():
                for sample in samples:
                    result = validator.validate(sample, spec)
                    # Store score for backward compatibility
                    sample.quality_scores[validator_name] = result.score
                    # Store full ValidationResult for proper pass/fail tracking
                    sample.metadata["validation_results"][validator_name] = result.model_dump()

        # Run batch-level validators
        for validator_name, validator in self.validators.items():
            if validator.is_batch_level():
                result = validator.validate_batch(samples, spec)
                # Store batch-level result in each sample
                for sample in samples:
                    sample.quality_scores[validator_name] = result.score
                    sample.metadata["validation_results"][validator_name] = result.model_dump()

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
