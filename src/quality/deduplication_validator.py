import hashlib
import json
from pathlib import Path

from src.core.base_validator import BaseValidator, ValidationResult
from src.core.models import Sample, Spec


class DeduplicationValidator(BaseValidator):
    """Validates that samples are unique by detecting and rejecting duplicates."""

    def __init__(self, config: dict):
        super().__init__(config)
        self.seen_hashes: set[str] = set()
        self.reference_file = config.get("reference_file")
        self.check_field = config.get("check_field", "content")

        # Load existing samples from reference file if provided
        if self.reference_file:
            self._load_reference_samples()

    def is_sample_level(self) -> bool:
        """Return True - this validator operates on individual samples."""
        return True

    def is_batch_level(self) -> bool:
        """Return False - this validator does not operate on batches."""
        return False

    def _load_reference_samples(self) -> None:
        """Load previously accepted samples to check against."""
        if not self.reference_file:
            return

        ref_path = Path(self.reference_file)
        if not ref_path.exists():
            return

        with open(ref_path) as f:
            for line in f:
                if line.strip():
                    try:
                        sample_data = json.loads(line)
                        content = sample_data.get(self.check_field, "")
                        if content:
                            content_hash = self._hash_content(content)
                            self.seen_hashes.add(content_hash)
                    except json.JSONDecodeError:
                        continue

    def _hash_content(self, content: str | dict) -> str:
        """Generate hash for content to efficiently detect duplicates."""
        if isinstance(content, dict):
            content = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def validate(self, sample: Sample, spec: Spec) -> ValidationResult:
        """Check if sample is a duplicate."""
        # Extract content to check for duplication
        if self.check_field == "content":
            check_value = sample.content
        else:
            check_value = sample.metadata.get(self.check_field, sample.content)

        content_hash = self._hash_content(check_value)

        # Check if we've seen this content before
        is_duplicate = content_hash in self.seen_hashes

        if not is_duplicate:
            # Mark as seen for future samples in the batch
            self.seen_hashes.add(content_hash)

        # Score: 1.0 for unique, 0.0 for duplicate
        score = 0.0 if is_duplicate else 1.0
        passed = not is_duplicate

        return ValidationResult(
            score=score,
            passed=passed,
            metadata={
                "is_duplicate": is_duplicate,
                "content_hash": content_hash
            },
        )
