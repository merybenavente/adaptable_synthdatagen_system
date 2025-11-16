#!/usr/bin/env python3
"""CLI script for running synthetic data generation.

Usage:
    # Basic usage with config file
    ./scripts/generate.py --config config/example.yaml

    # Save output to file
    ./scripts/generate.py --config config/example.yaml --output outputs/

    # Custom temperature
    ./scripts/generate.py --config config/example.yaml --temperature 0.9 --output outputs/

    # Filter out low-quality samples
    ./scripts/generate.py --config config/example.yaml --filter --output outputs/

    # Adaptive mode with feedback loop
    ./scripts/generate.py --config config/example.yaml --adaptive --batch-size 5 --output outputs/

Arguments:
    --config: Path to YAML config file (required)
    --output: Output directory for generated samples (optional)
    --temperature: Generation temperature, 0.0-1.0 (default: 0.8)
    --filter: Filter out samples that fail quality checks (optional)
    --adaptive: Enable adaptive feedback loop with iterative batches (optional)
    --batch-size: Samples per batch for adaptive mode (default: 5)
    --save-state: Save LocalFeedbackState to output directory (adaptive mode only)
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.core.config_loader import ConfigLoader
from src.core.feedback import FeedbackEngine
from src.core.pipeline import Pipeline
from src.core.spec import LocalFeedbackState
from src.generators.naive_generator import NaiveGenerator
from src.quality.orchestrator import QualityAssessmentOrchestrator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic data")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--output", type=Path, help="Output directory for generated samples")
    parser.add_argument("--temperature", type=float, default=0.8, help="Generation temperature")
    parser.add_argument(
        "--filter", action="store_true", help="Filter out samples that fail quality checks"
    )
    parser.add_argument(
        "--adaptive", action="store_true", help="Enable adaptive feedback loop"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Samples per batch (adaptive mode)"
    )
    parser.add_argument(
        "--save-state", action="store_true", help="Save feedback state (adaptive mode)"
    )
    args = parser.parse_args()

    # Load environment variables
    load_dotenv()

    # Setup logging for adaptive mode
    if args.adaptive:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )

    # Load configuration
    print(f"Loading config from {args.config}...")
    spec = ConfigLoader.load_spec(args.config)

    print(f"Domain: {spec.domain.value}")
    print(f"Task: {spec.task_input}")
    print(f"Samples to generate: {spec.num_samples}")
    print(f"Constraints: {spec.constraints}\n")

    if args.adaptive:
        # Run adaptive mode with feedback loop
        print("Running in ADAPTIVE mode with feedback loop\n")
        samples, final_state = run_adaptive(spec, args)
    else:
        # Run simple mode (original behavior)
        print("Running in SIMPLE mode\n")
        samples = run_simple(spec, args)

    # Display results
    print(f"\n{'='*60}")
    print(f"Generated {len(samples)} samples:")
    print('='*60)

    for i, sample in enumerate(samples[:3], 1):  # Show first 3
        print(f"\nSample {i}:")
        print(f"  Content: {sample.content}")
        print(f"  ID: {sample.id}")
        print(f"  Evolution Number: {sample.lineage.num_of_evolutions}")
        print(f"  Generator: {sample.lineage.generator}")
        print(f"  Model: {sample.lineage.generator_parameters['model']}")
        print(f"  Temperature: {sample.lineage.generator_parameters['temperature']}")
        print(f"  Timestamp: {sample.metadata['timestamp']}")
        print(f"  Quality Scores: {sample.quality_scores}")

    if len(samples) > 3:
        print(f"\n... and {len(samples) - 3} more samples")

    # Save to output directory if specified
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        output_file = args.output / "samples.jsonl"

        print(f"\nSaving samples to {output_file}...")
        with open(output_file, "w") as f:
            for sample in samples:
                f.write(json.dumps({
                    "id": str(sample.id),
                    "content": sample.content,
                    "lineage": {
                        "num_of_evolutions": sample.lineage.num_of_evolutions,
                        "generator": str(sample.lineage.generator),
                        "generator_parameters": sample.lineage.generator_parameters
                    },
                    "metadata": sample.metadata,
                    "quality_scores": sample.quality_scores
                }) + "\n")

        print(f"Successfully saved {len(samples)} samples to {output_file}")

        # Save feedback state if in adaptive mode
        if args.adaptive and args.save_state:
            state_file = args.output / "feedback_state.json"
            print(f"Saving feedback state to {state_file}...")
            with open(state_file, "w") as f:
                state_dict = final_state.model_dump()
                state_dict["recent_metrics"] = [
                    m.model_dump() for m in final_state.recent_metrics
                ]
                json.dump(state_dict, f, indent=2)


def run_simple(spec, args):
    """Run simple one-shot generation (original behavior)."""
    # Initialize generator
    print("Initializing generator...")
    generator = NaiveGenerator(spec, temperature=args.temperature)

    # Generate samples
    print(f"\n{'='*60}")
    print(f"Original Task Input: {spec.task_input}")
    print('='*60)
    print("\nGenerating samples...")
    samples = generator.generate()

    # Quality assessment
    print("\nRunning quality assessment...")
    orchestrator = QualityAssessmentOrchestrator()
    samples = orchestrator.assess(samples, spec)

    # Filter if requested
    if args.filter:
        original_count = len(samples)
        samples = orchestrator.filter_failing_samples(samples)
        filtered_count = original_count - len(samples)
        print(f"Filtered out {filtered_count} samples that failed quality checks")

    return samples


def run_adaptive(spec, args):
    """Run adaptive mode with feedback loop."""
    # Create initial feedback state
    initial_state = LocalFeedbackState(
        current_temperature=args.temperature,
    )

    # Create feedback engine
    feedback_engine = FeedbackEngine(
        temperature_adaptation=True,
        exploration_decay=True,
        max_history_length=10,
    )

    # Create quality orchestrator
    quality_orchestrator = QualityAssessmentOrchestrator() if args.filter else None

    # Create pipeline with feedback
    pipeline = Pipeline(
        feedback_engine=feedback_engine,
        quality_orchestrator=quality_orchestrator,
    )

    # Update router batch size if needed
    pipeline.router.default_batch_size = args.batch_size

    print(f"Starting adaptive pipeline:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Initial temperature: {args.temperature}")
    print(f"  Quality filtering: {'enabled' if args.filter else 'disabled'}\n")

    # Run pipeline with feedback loop
    samples, final_state = pipeline.run(
        spec=spec,
        initial_state=initial_state,
        max_iterations=100,
    )

    # Print final statistics
    print(f"\nFinal statistics:")
    print(f"  Total iterations: {final_state.iteration}")
    print(f"  Final temperature: {final_state.current_temperature:.3f}")
    print(f"  Final exploration rate: {final_state.exploration_rate:.3f}")

    arm_stats = feedback_engine.get_arm_statistics(final_state)
    if arm_stats:
        print(f"\nArm performance:")
        for arm_name, stats in arm_stats.items():
            print(f"  {arm_name}:")
            print(f"    Mean reward: {stats['mean_reward']:.3f}")
            print(f"    Count: {stats['count']}")

    return samples, final_state


if __name__ == "__main__":
    main()
