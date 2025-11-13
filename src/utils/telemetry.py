from typing import Any, Dict, Optional
from pathlib import Path


class Telemetry:
    """
    Minimal metrics tracking.

    Outputs to stdout/CSV for demo. In production would integrate
    with Prometheus, DataDog, etc.
    """

    def __init__(self, output_path: Optional[Path] = None):
        self.output_path = output_path
        self.metrics: Dict[str, Any] = {}

    def track(self, metric_name: str, value: Any, tags: Optional[Dict[str, str]] = None) -> None:
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
