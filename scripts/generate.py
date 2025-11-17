#!/usr/bin/env python3
"""CLI script for synthetic data generation with adaptive feedback loop.

Usage:
    # Basic usage
    ./scripts/generate.py --config config/recipes/task_rewrite_example.yaml

    # With custom settings
    ./scripts/generate.py --config config/recipes/task_rewrite_example.yaml \
        --batch-size 5 --output data/generated/

Arguments:
    --config: Path to YAML config file (required)
    --output: Output directory for generated samples (optional)
    --batch-size: Samples per batch (default: 5)
    --save-state: Save LocalFeedbackState to output directory (optional)
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.core.config_loader import ConfigLoader
from src.core.feedback import FeedbackEngine
from src.core.models import LocalFeedbackState
from src.core.pipeline import Pipeline
from src.quality.orchestrator import QualityAssessmentOrchestrator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic data with adaptive feedback")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--batch-size", type=int, default=5, help="Samples per batch")
    parser.add_argument("--save-state", action="store_true", help="Save feedback state")
    args = parser.parse_args()

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

    # Load spec
    print(f"Loading config from {args.config}...")
    spec = ConfigLoader.load_spec(args.config)

    print(f"Domain: {spec.domain.value}")
    print(f"Task: {spec.task_input}")
    print(f"Samples to generate: {spec.num_samples}")
    print(f"Constraints: {spec.constraints}\n")

    # Create feedback engine
    feedback_engine = FeedbackEngine()

    # Create quality orchestrator (always enabled for realistic feedback loop)
    quality_orchestrator = QualityAssessmentOrchestrator()

    # Create pipeline
    pipeline = Pipeline(
        feedback_engine=feedback_engine,
        quality_orchestrator=quality_orchestrator,
    )
    pipeline.router.default_batch_size = args.batch_size

    # Create initial state
    initial_state = LocalFeedbackState()

    print("Starting adaptive pipeline:")
    print(f"  Batch size: {args.batch_size}")
    print("  Quality filtering: enabled\n")

    # Run pipeline
    accepted_samples, rejected_samples, final_state = pipeline.run(
        spec=spec,
        initial_state=initial_state,
        max_iterations=100,
    )

    # Print final statistics
    print("\nFinal statistics:")
    print(f"  Accepted samples: {len(accepted_samples)}")
    print(f"  Rejected samples: {len(rejected_samples)}")
    print(f"  Total iterations: {final_state.iteration}")
    print(f"  Final exploration rate: {final_state.exploration_rate:.3f}")

    arm_stats = feedback_engine.get_arm_statistics(final_state)
    if arm_stats:
        print("\nArm performance:")
        for arm_name, stats in arm_stats.items():
            print(f"  {arm_name}:")
            print(f"    Mean reward: {stats['mean_reward']:.3f}")
            print(f"    Count: {stats['count']}")

    # Display sample examples
    print(f"\n{'='*60}")
    print(f"Sample examples (showing first 3 of {len(accepted_samples)}):")
    print('='*60)

    for i, sample in enumerate(accepted_samples[:3], 1):
        print(f"\nSample {i}:")
        print(f"  Content: {sample.content}")
        print(f"  Generator: {sample.lineage.generator}")
        print(f"  Temperature: {sample.lineage.generator_parameters.get('temperature', 'N/A')}")
        print(f"  Quality Scores: {sample.quality_scores}")

    if len(accepted_samples) > 3:
        print(f"\n... and {len(accepted_samples) - 3} more samples")

    # Save outputs
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

        # Save accepted samples
        output_file = args.output / "accepted_samples.jsonl"
        print(f"\nSaving accepted samples to {output_file}...")
        with open(output_file, "w") as f:
            for sample in accepted_samples:
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
        print(f"Successfully saved {len(accepted_samples)} accepted samples")

        # Save rejected samples
        rejected_file = args.output / "rejected_samples.jsonl"
        print(f"Saving rejected samples to {rejected_file}...")
        with open(rejected_file, "w") as f:
            for sample in rejected_samples:
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
        print(f"Successfully saved {len(rejected_samples)} rejected samples")

        if args.save_state:
            state_file = args.output / "feedback_state.json"
            print(f"Saving feedback state to {state_file}...")
            with open(state_file, "w") as f:
                json.dump(final_state.model_dump(), f, indent=2)


if __name__ == "__main__":
    main()
