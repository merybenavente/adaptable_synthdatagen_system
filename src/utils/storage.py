from pathlib import Path
from typing import Any, Dict, List
import json


class StorageUtils:
    """
    Local filesystem / S3-like IO utilities.

    Provides atomic writes and consistent interface for storage.
    """

    @staticmethod
    def write_json(data: Any, path: Path, atomic: bool = True) -> None:
        """
        TODO: Write JSON data to file.
        - Support atomic writes (write to temp, then rename)
        - Handle errors gracefully
        """
        pass

    @staticmethod
    def read_json(path: Path) -> Any:
        """
        TODO: Read JSON data from file.
        """
        pass

    @staticmethod
    def write_jsonl(data: List[Dict], path: Path, atomic: bool = True) -> None:
        """
        TODO: Write JSONL data to file.
        - One JSON object per line
        - Support atomic writes
        """
        pass

    @staticmethod
    def read_jsonl(path: Path) -> List[Dict]:
        """
        TODO: Read JSONL data from file.
        """
        pass
