"""Type guard functions for runtime type checking of dict structures."""

from typing import Any, TypeGuard

from src.core.generator_types import GeneratorType


def is_batch_input_dict(task_input: str | dict[str, Any]) -> TypeGuard[dict[str, Any]]:
    """Type guard: Check if task_input is a batch input dict with 'input_file'."""
    return isinstance(task_input, dict) and "input_file" in task_input


def is_ml_augmentation_dict(task_input: str | dict[str, Any]) -> TypeGuard[dict[str, Any]]:
    """Type guard: Check if task_input is an ML augmentation dict with expected_output."""
    return isinstance(task_input, dict) and "expected_output" in task_input


def is_valid_generator_arm_string(arm: str | GeneratorType) -> TypeGuard[str]:
    """Type guard: Check if generator_arm is a valid string (not GeneratorType enum)."""
    return isinstance(arm, str)


def is_valid_generator_type_string(arm: str) -> bool:
    """Check if string is a valid GeneratorType enum value."""
    try:
        GeneratorType(arm)
        return True
    except ValueError:
        return False

