"""Grammar sampler with temperature-controlled diversity."""

import random
import re
from typing import Any

import numpy as np

from src.generators.templater.grammar import Grammar
from src.utils.llm_client import LLMClient


class GrammarSampler:
    """Samples from PCFG with temperature-controlled diversity."""

    def __init__(
        self,
        grammar: Grammar,
        llm_client: LLMClient,
        temperature: float = 1.0,
        max_depth: int = 10
    ):
        """Initialize sampler with grammar, LLM client, and sampling params."""
        self.grammar = grammar
        self.llm_client = llm_client
        self.temperature = temperature
        self.max_depth = max_depth
        self.current_depth = 0

    def sample(self) -> str:
        """Generate a sample by expanding from the start symbol."""
        self.current_depth = 0
        return self._expand(self.grammar.start)

    def _expand(self, symbol: str) -> str:
        """Recursively expand a grammar symbol."""
        # Check max depth to prevent infinite recursion
        self.current_depth += 1
        if self.current_depth > self.max_depth:
            raise RecursionError(f"Max derivation depth {self.max_depth} exceeded")

        # Get rule options for this symbol
        options = self.grammar.get_rule_options(symbol)

        # Sample one option using temperature-adjusted weights
        chosen_option = self._sample_option(options)

        # Generate content based on option type
        if self.grammar.is_llm_rule(chosen_option):
            content = self._generate_llm_content(chosen_option)
        else:
            content = chosen_option['template']

        # Recursively expand any placeholders in the content
        content = self._expand_placeholders(content)

        self.current_depth -= 1
        return content

    def _sample_option(self, options: list[dict[str, Any]]) -> dict[str, Any]:
        """Sample one option from list using temperature-adjusted weights."""
        weights = [opt['weight'] for opt in options]

        # Apply temperature to weights for diversity control
        if self.temperature != 1.0:
            weights = self._apply_temperature(weights)

        # Normalize weights to probabilities
        total = sum(weights)
        probabilities = [w / total for w in weights]

        # Sample
        return random.choices(options, weights=probabilities)[0]

    def _apply_temperature(self, weights: list[float]) -> list[float]:
        """Apply temperature scaling to weights (higher temp = more uniform)."""
        log_weights = np.log(np.array(weights) + 1e-10)  # Add small epsilon to avoid log(0)
        scaled = log_weights / self.temperature
        exp_scaled = np.exp(scaled - np.max(scaled))  # Subtract max for numerical stability
        return exp_scaled.tolist()

    def _generate_llm_content(self, option: dict[str, Any]) -> str:
        """Generate content using LLM based on llm_fill specification."""
        llm_spec = option['llm_fill']
        prompt = llm_spec.get('prompt', '')

        # Generate using LLM
        content = self.llm_client.generate(prompt)
        return content.strip()

    def _expand_placeholders(self, text: str) -> str:
        """Find and expand all <symbol> placeholders in text."""
        # Pattern matches <symbol_name>
        pattern = r'<(\w+)>'

        def replace_placeholder(match):
            symbol = match.group(1)
            return self._expand(symbol)

        # Recursively replace all placeholders
        return re.sub(pattern, replace_placeholder, text)
