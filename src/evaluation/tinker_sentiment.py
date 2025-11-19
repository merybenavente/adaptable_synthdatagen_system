from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class TinkerSentimentConfig:
    """Minimal config for using sentiment data with Tinker chat_sl."""

    train_path: Path
    eval_path: Path | None = None
    input_field: str = "text"
    label_field: str = "label"
    recipe: str = "chat_sl"
    task: str = "sentiment_classification"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable config."""
        return {
            "recipe": self.recipe,
            "task": self.task,
            "train_file": str(self.train_path),
            "eval_file": str(self.eval_path) if self.eval_path else None,
            "input_field": self.input_field,
            "label_field": self.label_field,
        }


def build_default_tinker_sentiment_config(dataset_path: Path) -> TinkerSentimentConfig:
    """Build default config for a single sentiment dataset."""
    return TinkerSentimentConfig(train_path=dataset_path)


