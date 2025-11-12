from typing import Any, Dict, List

from src.core.base_generator import BaseGenerator


class WizardLMGenerator(BaseGenerator):
    """Generator using WizardLM-style Evol-Instruct approach."""

    def generate(self) -> List[Dict[str, Any]]:
        pass

    def get_capabilities(self) -> Dict[str, Any]:
        pass
