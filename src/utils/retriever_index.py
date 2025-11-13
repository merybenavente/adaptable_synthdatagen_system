from pathlib import Path
from typing import Any, Dict, List


class IndexBuilder:
    """
    Build and manage FAISS/ScaNN indexes for retrieval.

    Builds indexes from data/kb/ and saves to artifacts/retriever/.
    """

    def build_index(self, kb_path: Path, output_path: Path, config: Dict[str, Any]) -> None:
        """
        TODO: Build retrieval index.
        - Load documents from kb_path
        - Generate embeddings
        - Build FAISS/ScaNN index
        - Save to output_path
        """
        pass

    def load_index(self, index_path: Path) -> Any:
        """
        TODO: Load existing index.
        - Load FAISS/ScaNN index from disk
        - Return index object
        """
        pass
