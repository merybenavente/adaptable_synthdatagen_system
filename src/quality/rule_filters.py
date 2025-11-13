from typing import Any, Dict

from src.core.base_validator import BaseValidator


class LengthValidator(BaseValidator):
    """Validate text length against min/max requirements."""

    def validate(self, sample: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Implement length validation.
        - Check text length against thresholds from config/validators.yaml
        - Return score and passed status
        """
        pass


class FormatValidator(BaseValidator):
    """Validate format (JSON schema, regex, etc.)."""

    def validate(self, sample: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Implement format validation.
        - Validate against JSON schema or regex pattern
        - Return score and passed status
        """
        pass


class JSONSchemaValidator(BaseValidator):
    """Validate samples against JSON schema."""

    def validate(self, sample: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        TODO: Implement JSON schema validation.
        - Load schema from config
        - Validate sample structure
        - Return score and passed status
        """
        pass
