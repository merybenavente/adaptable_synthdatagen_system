#!/usr/bin/env python3
"""CLI script for running synthetic data generation.

Usage:
    # Basic usage with config file
    ./scripts/generate.py --config config/example.yaml

    # Save output to file
    ./scripts/generate.py --config config/example.yaml --output outputs/

    # Custom temperature
    ./scripts/generate.py --config config/example.yaml --temperature 0.9 --output outputs/

Arguments:
    --config: Path to YAML config file (required)
    --output: Output directory for generated samples (optional)
    --temperature: Generation temperature, 0.0-1.0 (default: 0.8)
"""

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from src.core.config_loader import ConfigLoader
from src.generators.naive_generator import NaiveGenerator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--output", type=Path, help="Output directory for generated samples")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Load configuration
    print(f"Loading config from {args.config}...")
    config_loader = ConfigLoader(args.config)
    spec = config_loader.load()

    print(f"Domain: {spec.domain.value}")
    print(f"Task: {spec.task_input}")
    print(f"Samples to generate: {spec.num_samples}")
    print(f"Constraints: {spec.constraints}\n")

    # Initialize generator
    print("Initializing generator...")
    generator = NaiveGenerator(spec, temperature=args.temperature)

    # Generate samples
    print(f"\n{'='*60}")
    print(f"Original Task Input: {spec.task_input}")
    print('='*60)
    print("\nGenerating samples...")
    samples = generator.generate()

    # Display results
    print(f"\n{'='*60}")
    print(f"Generated {len(samples)} samples:")
    print('='*60)

    for i, sample in enumerate(samples, 1):
        print(f"\nSample {i}:")
        print(f"  Content: {sample.content}")
        print(f"  ID: {sample.id}")
        print(f"  Evolution Number: {sample.lineage.num_of_evolutions}")
        print(f"  Generator: {sample.lineage.generator}")
        print(f"  Model: {sample.lineage.generator_parameters['model']}")
        print(f"  Temperature: {sample.lineage.generator_parameters['temperature']}")
        print(f"  Timestamp: {sample.metadata['timestamp']}")

    # Save to output directory if specified
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        output_file = args.output / "samples.jsonl"

        print(f"\nSaving samples to {output_file}...")
        with open(output_file, "w") as f:
            for sample in samples:
                f.write(json.dumps({
                    "id": sample.id,
                    "content": sample.content,
                    "lineage": {
                        "num_of_evolutions": sample.lineage.num_of_evolutions,
                        "generator": sample.lineage.generator,
                        "generator_parameters": sample.lineage.generator_parameters
                    },
                    "metadata": sample.metadata
                }) + "\n")

        print(f"Successfully saved {len(samples)} samples to {output_file}")


if __name__ == "__main__":
    main()
