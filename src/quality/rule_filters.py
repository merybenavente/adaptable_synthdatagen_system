import json

import jsonschema
from jsonschema import ValidationError

from src.core.base_validator import BaseValidator, ValidationResult
from src.core.models import Sample, Spec


class JSONSchemaValidator(BaseValidator):
    """Validate samples against JSON schema."""

    def is_sample_level(self) -> bool:
        return True

    def is_batch_level(self) -> bool:
        return False

    def validate(self, sample: Sample, spec: Spec) -> ValidationResult:
        """Validate sample content against JSON schema from spec.constraints.schema."""
        # Get schema from spec constraints
        schema = spec.constraints.get("schema")
        if not schema:
            # No schema provided - skip validation (pass by default)
            return ValidationResult(
                score=1.0,
                passed=True,
                metadata={"error": "No schema provided in spec.constraints.schema"},
            )

        # Extract content to validate
        content = sample.content

        # Handle string content - try to parse as JSON
        if isinstance(content, str):
            try:
                content = json.loads(content)
            except json.JSONDecodeError as e:
                return ValidationResult(
                    score=0.0,
                    passed=False,
                    metadata={"error": f"Content is not valid JSON: {str(e)}"},
                )

        # Validate content against schema
        try:
            jsonschema.validate(instance=content, schema=schema)
            # Validation passed
            return ValidationResult(
                score=1.0,
                passed=True,
                metadata={"validated": True},
            )
        except ValidationError as e:
            # Validation failed - extract error details
            error_message = e.message
            error_path = ".".join(str(p) for p in e.path) if e.path else "root"
            return ValidationResult(
                score=0.0,
                passed=False,
                metadata={
                    "error": error_message,
                    "error_path": error_path,
                    "validator": e.validator,
                },
            )
        except jsonschema.SchemaError as e:
            # Schema itself is invalid
            return ValidationResult(
                score=0.0,
                passed=False,
                metadata={"error": f"Invalid schema: {str(e)}"},
            )
