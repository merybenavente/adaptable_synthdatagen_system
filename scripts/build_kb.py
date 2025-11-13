#!/usr/bin/env python3
"""Build FAISS/ScaNN index from knowledge base."""

import argparse
from pathlib import Path


def main():
    """Build retrieval index from KB."""
    parser = argparse.ArgumentParser(description="Build retrieval index from knowledge base")
    parser.add_argument("--kb-path", type=Path, required=True, help="Path to KB data")
    parser.add_argument("--output", type=Path, required=True, help="Output index path")
    parser.add_argument("--config", type=Path, default="config/kb.yaml", help="KB config")
    args = parser.parse_args()

    # TODO: Implement index building
    # - Load KB documents
    # - Generate embeddings
    # - Build and save index


if __name__ == "__main__":
    main()
