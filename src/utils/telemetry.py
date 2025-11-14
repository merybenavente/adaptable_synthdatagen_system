from pathlib import Path
from typing import Any


class Telemetry:
    """
    Minimal metrics tracking.

    Outputs to stdout/CSV for demo. In production would integrate
    with Prometheus, DataDog, etc.
    """

    def __init__(self, output_path: Path | None = None):
        self.output_path = output_path
        self.metrics: dict[str, Any] = {}

    def track(self, metric_name: str, value: Any, tags: dict[str, str] | None = None) -> None:
        """
        TODO: Track a metric.
        - Store in memory
        - Optionally write to CSV
        - Print to stdout for demo
        """
        pass

    def flush(self) -> None:
        """
        TODO: Flush metrics to storage.
        """
        pass
