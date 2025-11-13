from typing import Any, Callable, Dict, Type


class Registry:
    """
    Registry for generators and validators by name.

    Allows dynamic lookup and instantiation of components.
    """

    def __init__(self):
        self._generators: Dict[str, Type] = {}
        self._validators: Dict[str, Type] = {}

    def register_generator(self, name: str, generator_class: Type) -> None:
        """Register a generator class."""
        pass

    def register_validator(self, name: str, validator_class: Type) -> None:
        """Register a validator class."""
        pass

    def get_generator(self, name: str) -> Type:
        """Get generator class by name."""
        pass

    def get_validator(self, name: str) -> Type:
        """Get validator class by name."""
        pass
