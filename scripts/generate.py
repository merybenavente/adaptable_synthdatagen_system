#!/usr/bin/env python3
"""CLI script for running synthetic data generation."""

import argparse
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--output", type=Path, help="Output directory")
    args = parser.parse_args()


if __name__ == "__main__":
    main()
