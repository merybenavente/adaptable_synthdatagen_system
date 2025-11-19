"""Auto-derive JSON schema from PCFG grammar templates."""

import json
import re
from typing import Any

from src.generators.templater.grammar import Grammar
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class GrammarSchemaDeriver:
    """Derives JSON schema from PCFG grammar templates."""

    def __init__(self):
        self.grammar = None

    def derive_schema(self, grammar: Grammar) -> dict[str, Any]:
        """Derive JSON schema from grammar start symbol templates."""
        self.grammar = grammar  # Store for type inference

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

        # Infer types from grammar rules for placeholders
        merged_schema = self._infer_types_from_rules(merged_schema)

        logger.debug(f"Auto-derived schema from grammar: {merged_schema}")
        return merged_schema

    def _parse_template(self, template: str) -> dict[str, Any] | None:
        """Parse a template string to extract JSON structure."""
        # Replace placeholders with dummy string values to make valid JSON
        # "<positive_review>" → "PLACEHOLDER" (including the quotes around placeholder)
        dummy_template = re.sub(r'"<\w+>"', '"PLACEHOLDER"', template)
        # Also handle unquoted placeholders (for numbers/booleans): <rating> → "PLACEHOLDER_NUM"
        dummy_template = re.sub(r'<\w+>', '"PLACEHOLDER_NUM"', dummy_template)

        try:
            parsed = json.loads(dummy_template)
        except json.JSONDecodeError as e:
            logger.debug(f"Template is not valid JSON, skipping: {template[:100]}... Error: {e}")
            return None

        # Only support object templates for now
        if not isinstance(parsed, dict):
            logger.debug(f"Template is not a JSON object, skipping: {template[:100]}")
            return None

        # Extract structure recursively
        schema = self._extract_schema_from_value(parsed)
        return schema

    def _extract_schema_from_value(self, value: Any) -> dict[str, Any]:
        """Recursively extract schema from a parsed JSON value."""
        if isinstance(value, dict):
            # Nested object - recurse into it
            schema = {"type": "object", "properties": {}, "required": []}
            for key, val in value.items():
                schema["required"].append(key)
                schema["properties"][key] = self._extract_schema_from_value(val)
            return schema

        elif isinstance(value, list):
            # Array - try to infer item type from first element
            if value:
                item_schema = self._extract_schema_from_value(value[0])
                return {"type": "array", "items": item_schema}
            else:
                return {"type": "array"}

        elif isinstance(value, str):
            if value == "PLACEHOLDER":
                # String placeholder
                return {"type": "string"}
            elif value == "PLACEHOLDER_NUM":
                # Unquoted placeholder - could be number, boolean, or string
                # Default to string, let merge logic handle it
                return {"type": "string"}
            else:
                # Literal value - record it for potential enum
                return {"type": "string", "const": value}

        elif isinstance(value, bool):
            # Check before int since bool is subclass of int in Python
            return {"type": "boolean", "const": value}

        elif isinstance(value, int):
            return {"type": "integer", "const": value}

        elif isinstance(value, float):
            return {"type": "number", "const": value}

        else:
            # Fallback
            return {"type": "string"}

    def _merge_schemas(self, schemas: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge multiple schemas to find common structure and derive enums."""
        if not schemas:
            return {}

        if len(schemas) == 1:
            # Single schema - just clean up and return
            return self._cleanup_schema(schemas[0])

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
            merged["properties"][field] = self._merge_property_schemas(props)

        return merged

    def _merge_property_schemas(self, props: list[dict[str, Any]]) -> dict[str, Any]:
        """Merge multiple property schemas, handling nested objects and enums."""
        if not props:
            return {"type": "string"}

        # Check if all are the same type
        types = [p.get("type") for p in props]
        if len(set(types)) == 1:
            prop_type = types[0]

            # Handle nested objects recursively
            if prop_type == "object":
                # All are objects - merge them recursively
                return self._merge_schemas(props)

            # Handle arrays
            elif prop_type == "array":
                # Try to merge item schemas if present
                item_schemas = [p.get("items") for p in props if "items" in p]
                if item_schemas:
                    merged_items = self._merge_property_schemas(item_schemas)
                    return {"type": "array", "items": merged_items}
                else:
                    return {"type": "array"}

            # Handle enums for primitive types
            else:
                const_values = [p.get("const") for p in props if "const" in p]
                if const_values and len(const_values) == len(props):
                    # All have const values → create enum
                    unique_values = sorted(set(const_values), key=str)
                    return {"type": prop_type, "enum": unique_values}
                else:
                    # Mixed or no const - return base type
                    base_prop = props[0].copy()
                    base_prop.pop("const", None)
                    return base_prop

        else:
            # Mixed types - use first one and remove const
            base_prop = props[0].copy()
            base_prop.pop("const", None)
            return base_prop

    def _cleanup_schema(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Recursively clean up a schema (remove single const values, etc)."""
        if schema.get("type") == "object":
            cleaned = {
                "type": "object",
                "properties": {},
                "required": schema.get("required", []),
            }
            for field, prop in schema.get("properties", {}).items():
                cleaned["properties"][field] = self._cleanup_schema(prop)
            return cleaned

        elif schema.get("type") == "array" and "items" in schema:
            return {"type": "array", "items": self._cleanup_schema(schema["items"])}

        else:
            # For primitives, keep const if present (will be useful info)
            return schema.copy()

    def _infer_types_from_rules(self, schema: dict[str, Any]) -> dict[str, Any]:
        """Infer actual types by looking at grammar rule values."""
        if not self.grammar or schema.get("type") != "object":
            return schema

        # Process properties recursively
        for field, prop in schema.get("properties", {}).items():
            if prop.get("type") == "object":
                # Recurse into nested objects
                schema["properties"][field] = self._infer_types_from_rules(prop)
            elif prop.get("type") == "string" and "enum" not in prop:
                # Try to infer type from grammar rule with same name as field
                inferred_type = self._infer_type_from_rule(field)
                if inferred_type:
                    schema["properties"][field] = inferred_type

        return schema

    def _infer_type_from_rule(self, rule_name: str) -> dict[str, Any] | None:
        """Infer type and possible enum from a grammar rule."""
        if not self.grammar or rule_name not in self.grammar.rules:
            return None

        options = self.grammar.get_rule_options(rule_name)
        if not options:
            return None

        # Collect all template values
        values = []
        for option in options:
            if "template" in option:
                template_val = option["template"].strip()
                # Try to parse as JSON literal
                try:
                    parsed_val = json.loads(template_val)
                    values.append(parsed_val)
                except json.JSONDecodeError:
                    # Not a JSON literal, treat as string
                    values.append(template_val)

        if not values:
            return None

        # Infer type from collected values
        types = {type(v).__name__ for v in values}

        # All same type
        if len(types) == 1:
            first_val = values[0]

            if isinstance(first_val, bool):
                if len(values) > 1:
                    return {"type": "boolean", "enum": sorted(set(values))}
                else:
                    return {"type": "boolean"}
            elif isinstance(first_val, int):
                if len(values) > 1:
                    return {"type": "integer", "enum": sorted(set(values))}
                else:
                    return {"type": "integer"}
            elif isinstance(first_val, float):
                if len(values) > 1:
                    return {"type": "number", "enum": sorted(set(values))}
                else:
                    return {"type": "number"}
            elif isinstance(first_val, str):
                unique_vals = sorted(set(values))
                if len(unique_vals) > 1:
                    return {"type": "string", "enum": unique_vals}
                else:
                    return {"type": "string"}

        # Mixed types - default to string
        return {"type": "string"}
