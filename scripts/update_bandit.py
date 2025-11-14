#!/usr/bin/env python3
"""Update contextual bandit from batch feedback."""

import argparse
from pathlib import Path


def main():
    """Update bandit model from logged feedback."""
    parser = argparse.ArgumentParser(description="Update bandit from feedback logs")
    parser.add_argument(
        "--feedback-dir",
        type=Path,
        default="artifacts/bandit/feedback/",
        help="Feedback logs directory"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default="artifacts/bandit/model.pkl",
        help="Output bandit model"
    )
    _args = parser.parse_args()

    # TODO: Implement bandit update
    # - Load feedback logs
    # - Aggregate rewards by arm/context
    # - Update bandit model
    # - Save updated model


if __name__ == "__main__":
    main()
