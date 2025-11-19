"""Grammar loader and validator for template-based generation."""

from typing import Any

import yaml


class Grammar:
    """PCFG supporting template rules and LLM-generated content."""

    def __init__(self, grammar_dict: dict[str, Any]):
        """Initialize and validate grammar from dict with 'start' and 'rules'."""
        self.start = grammar_dict.get('start')
        self.rules = grammar_dict.get('rules', {})
        self._validate()

    def _validate(self) -> None:
        """Validate grammar structure and rule format."""
        if not self.start:
            raise ValueError("Grammar must have a 'start' symbol")

        if self.start not in self.rules:
            raise ValueError(f"Start symbol '{self.start}' not found in rules")

        # Validate each rule
        for rule_name, options in self.rules.items():
            if not isinstance(options, list):
                raise ValueError(f"Rule '{rule_name}' must be a list of options")

            if len(options) == 0:
                raise ValueError(f"Rule '{rule_name}' must have at least one option")

            for i, option in enumerate(options):
                if not isinstance(option, dict):
                    raise ValueError(f"Rule '{rule_name}' option {i} must be a dict")

                # Must have either 'template' or 'llm_fill'
                has_template = 'template' in option
                has_llm = 'llm_fill' in option

                if not has_template and not has_llm:
                    raise ValueError(
                        f"Rule '{rule_name}' option {i} must have 'template' or 'llm_fill'"
                    )

                if has_template and has_llm:
                    raise ValueError(
                        f"Rule '{rule_name}' option {i} cannot have both 'template' and 'llm_fill'"
                    )

                # Must have weight
                if 'weight' not in option:
                    raise ValueError(f"Rule '{rule_name}' option {i} must have 'weight'")

                if not isinstance(option['weight'], (int, float)) or option['weight'] <= 0:
                    raise ValueError(
                        f"Rule '{rule_name}' option {i} weight must be a positive number"
                    )

    def get_rule_options(self, rule_name: str) -> list[dict[str, Any]]:
        """Return all options for the given rule."""
        if rule_name not in self.rules:
            raise ValueError(f"Rule '{rule_name}' not found in grammar")
        return self.rules[rule_name]

    def is_llm_rule(self, option: dict[str, Any]) -> bool:
        """Return True if option requires LLM generation."""
        return 'llm_fill' in option

    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Grammar':
        """Load grammar from YAML recipe file."""
        with open(yaml_path) as f:
            recipe = yaml.safe_load(f)

        if 'grammar' not in recipe:
            raise ValueError(f"Recipe file {yaml_path} must contain 'grammar' section")

        return cls(recipe['grammar'])
