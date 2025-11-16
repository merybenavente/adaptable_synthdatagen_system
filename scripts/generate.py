#!/usr/bin/env python3
"""CLI script for synthetic data generation with adaptive feedback loop.

Usage:
    # Basic usage
    ./scripts/generate.py --config config/recipes/task_rewrite_example.yaml

    # With custom settings
    ./scripts/generate.py --config config/recipes/task_rewrite_example.yaml \
        --batch-size 5 --temperature 0.8 --filter --output data/generated/

Arguments:
    --config: Path to YAML config file (required)
    --output: Output directory for generated samples (optional)
    --temperature: Initial generation temperature, 0.0-2.0 (default: 0.7)
    --batch-size: Samples per batch (default: 5)
    --filter: Filter out samples that fail quality checks (optional)
    --save-state: Save LocalFeedbackState to output directory (optional)
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
from src.quality.orchestrator import QualityAssessmentOrchestrator


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Generate synthetic data with adaptive feedback")
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--temperature", type=float, default=0.7, help="Initial temperature")
    parser.add_argument("--batch-size", type=int, default=5, help="Samples per batch")
    parser.add_argument("--filter", action="store_true", help="Filter low-quality samples")
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
    feedback_engine = FeedbackEngine(max_history_length=10)

    # Create quality orchestrator
    quality_orchestrator = QualityAssessmentOrchestrator() if args.filter else None

    # Create pipeline
    pipeline = Pipeline(
        feedback_engine=feedback_engine,
        quality_orchestrator=quality_orchestrator,
    )
    pipeline.router.default_batch_size = args.batch_size

    # Create initial state
    initial_state = LocalFeedbackState(current_temperature=args.temperature)

    print(f"Starting adaptive pipeline:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Initial temperature: {args.temperature}")
    print(f"  Quality filtering: {'enabled' if args.filter else 'disabled'}\n")

    # Run pipeline
    samples, final_state = pipeline.run(
        spec=spec,
        initial_state=initial_state,
        max_iterations=100,
    )

    # Print final statistics
    print(f"\nFinal statistics:")
    print(f"  Total samples: {len(samples)}")
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

    # Display sample examples
    print(f"\n{'='*60}")
    print(f"Sample examples (showing first 3 of {len(samples)}):")
    print('='*60)

    for i, sample in enumerate(samples[:3], 1):
        print(f"\nSample {i}:")
        print(f"  Content: {sample.content}")
        print(f"  Generator: {sample.lineage.generator}")
        print(f"  Temperature: {sample.lineage.generator_parameters.get('temperature', 'N/A')}")
        print(f"  Quality Scores: {sample.quality_scores}")

    if len(samples) > 3:
        print(f"\n... and {len(samples) - 3} more samples")

    # Save outputs
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

        print(f"Successfully saved {len(samples)} samples")

        if args.save_state:
            state_file = args.output / "feedback_state.json"
            print(f"Saving feedback state to {state_file}...")
            with open(state_file, "w") as f:
                state_dict = final_state.model_dump()
                state_dict["recent_metrics"] = [
                    m.model_dump() for m in final_state.recent_metrics
                ]
                json.dump(state_dict, f, indent=2)


if __name__ == "__main__":
    main()
