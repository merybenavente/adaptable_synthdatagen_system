from src.core.generator_types import GeneratorType


class Registry:
    """
    Registry for generators and validators by name.

    Allows dynamic lookup and instantiation of components.
    """

    def __init__(self):
        self._generators: dict[str | GeneratorType, type] = {}
        self._validators: dict[str, type] = {}

    def register_generator(self, name: str | GeneratorType, generator_class: type) -> None:
        """Register a generator class."""
        pass

    def register_validator(self, name: str, validator_class: type) -> None:
        """Register a validator class."""
        pass

    def get_generator(self, name: str | GeneratorType) -> type:
        """Get generator class by name."""
        pass

    def get_validator(self, name: str) -> type:
        """Get validator class by name."""
        pass
