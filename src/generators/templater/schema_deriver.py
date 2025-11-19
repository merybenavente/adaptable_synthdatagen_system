"""Auto-derive JSON schema from PCFG grammar templates."""

import json
import re
from typing import Any

from src.generators.templater.grammar import Grammar
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GrammarSchemaDeriver:
    """Derives JSON schema from PCFG grammar templates."""

    def derive_schema(self, grammar: Grammar) -> dict[str, Any]:
        """Derive JSON schema from grammar start symbol templates."""
        # Get all templates from the start symbol
        start_options = grammar.get_rule_options(grammar.start)

        # Parse templates to extract structure
        schemas = []
        for option in start_options:
            if "template" in option:
                template_schema = self._parse_template(option["template"])
                if template_schema:
                    schemas.append(template_schema)

        if not schemas:
            logger.warning(f"No valid templates found in start symbol '{grammar.start}'")
            return {}

        # Merge all schemas to find common structure and derive enums
        merged_schema = self._merge_schemas(schemas)

        logger.debug(f"Auto-derived schema from grammar: {merged_schema}")
        return merged_schema

    def _parse_template(self, template: str) -> dict[str, Any] | None:
        """Parse a template string to extract JSON structure."""
        # Replace placeholders with dummy string values to make valid JSON
        # "<positive_review>" → "PLACEHOLDER" (including the quotes around placeholder)
        dummy_template = re.sub(r'"<\w+>"', '"PLACEHOLDER"', template)

        try:
            parsed = json.loads(dummy_template)
        except json.JSONDecodeError as e:
            logger.debug(f"Template is not valid JSON, skipping: {template[:100]}... Error: {e}")
            return None

        # Only support object templates for now
        if not isinstance(parsed, dict):
            logger.debug(f"Template is not a JSON object, skipping: {template[:100]}")
            return None

        # Extract structure
        schema = {"type": "object", "properties": {}, "required": []}

        for key, value in parsed.items():
            schema["required"].append(key)

            if isinstance(value, str):
                if value == "PLACEHOLDER":
                    # This is a placeholder - type is string
                    schema["properties"][key] = {"type": "string"}
                else:
                    # Literal value - record it for potential enum
                    schema["properties"][key] = {"type": "string", "const": value}
            elif isinstance(value, int):
                schema["properties"][key] = {"type": "integer", "const": value}
            elif isinstance(value, float):
                schema["properties"][key] = {"type": "number", "const": value}
            elif isinstance(value, bool):
                schema["properties"][key] = {"type": "boolean", "const": value}
            elif isinstance(value, list):
                # Arrays - just set type, can't infer items without more analysis
                schema["properties"][key] = {"type": "array"}
            elif isinstance(value, dict):
                # Nested objects - just set type for now
                schema["properties"][key] = {"type": "object"}

        return schema

    def _merge_schemas(self, schemas: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge multiple schemas to find common structure and derive enums."""
        if not schemas:
            return {}

        if len(schemas) == 1:
            # Single schema - just clean up const values
            schema = schemas[0].copy()
            for field, prop in schema.get("properties", {}).items():
                if "const" in prop:
                    # Single value, keep as const
                    pass
            return schema

        # Multiple schemas - find common structure
        merged = {"type": "object", "properties": {}, "required": []}

        # Find common required fields across all templates
        all_required = [set(s.get("required", [])) for s in schemas]
        common_required = set.intersection(*all_required) if all_required else set()
        merged["required"] = sorted(common_required)

        # Collect all properties by field name
        all_properties = {}
        for schema in schemas:
            for field, prop in schema.get("properties", {}).items():
                if field not in all_properties:
                    all_properties[field] = []
                all_properties[field].append(prop)

        # Derive property schemas
        for field, props in all_properties.items():
            # Check if all props have 'const' → derive enum
            const_values = [p.get("const") for p in props if "const" in p]

            if const_values and len(const_values) == len(props):
                # All are const values → create enum
                unique_values = sorted(set(const_values), key=str)
                base_type = props[0].get("type", "string")

                merged["properties"][field] = {"type": base_type, "enum": unique_values}
            else:
                # Mixed or no const values - use type from first prop
                base_prop = props[0].copy()
                # Remove const if present (not consistent across templates)
                base_prop.pop("const", None)
                merged["properties"][field] = base_prop

        return merged
