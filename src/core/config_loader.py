from pathlib import Path
from typing import Any

import yaml

from .models import Spec


class ConfigLoader:
    """Load and validate YAML configuration files."""

    @staticmethod
    def load_spec(path: str | Path) -> Spec:
        """Load and validate a Spec from YAML file."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"Spec file not found: {file_path}")

        with open(file_path) as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Expected YAML dict, got {type(data)}")

        return Spec(**data)

    @staticmethod
    def load_yaml(path: str | Path) -> dict[str, Any]:
        """Load raw YAML data without validation."""
        file_path = Path(path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path) as f:
            return yaml.safe_load(f)
