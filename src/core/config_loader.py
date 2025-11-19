from pathlib import Path
from typing import Any

import yaml

from src.generators.templater import GrammarSchemaDeriver
from src.generators.templater.grammar import Grammar
from src.utils.logger import setup_logger

from .models import Spec

logger = setup_logger(__name__)


class ConfigLoader:
    """Load and validate YAML configuration files."""

    @staticmethod
    def load_spec(path: str | Path) -> Spec:
        """Load and validate a Spec from YAML file."""
        file_path = Path(path).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"Spec file not found: {file_path}")

        with open(file_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML dict, got {type(data)}")

        # If recipe has a 'grammar' section, auto-add grammar_path to constraints
        if "grammar" in data:
            if "constraints" not in data:
                data["constraints"] = {}
            # Only set grammar_path if not already specified
            if "grammar_path" not in data["constraints"]:
                data["constraints"]["grammar_path"] = str(file_path)

            # Auto-derive JSON schema from grammar if not provided (Opción 3)
            if "schema" not in data["constraints"]:
                try:
                    grammar = Grammar(data["grammar"])
                    deriver = GrammarSchemaDeriver()
                    derived_schema = deriver.derive_schema(grammar)

                    if derived_schema:
                        # Mark schema as auto-derived for logging
                        derived_schema["_derived_from_grammar"] = True
                        data["constraints"]["schema"] = derived_schema
                        logger.info(
                            f"✓ Auto-derived JSON schema from grammar:\n"
                            f"  Required fields: {derived_schema.get('required', [])}\n"
                            f"  Properties: {list(derived_schema.get('properties', {}).keys())}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to auto-derive schema from grammar: {e}")

        return Spec(**data)

    @staticmethod
    def load_yaml(path: str | Path) -> dict[str, Any]:
        """Load raw YAML data without validation."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path) as f:
            return yaml.safe_load(f)
