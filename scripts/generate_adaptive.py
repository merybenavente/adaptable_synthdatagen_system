#!/usr/bin/env python3
"""CLI script for running adaptive synthetic data generation with feedback loop.

This script demonstrates the full adaptive pipeline:
- Iterative batch generation
- Router selects generator arms based on feedback
- Quality scoring and metrics computation
- Adaptive parameter adjustment (temperature, exploration rate)

Usage:
    # Basic usage with config file
    ./scripts/generate_adaptive.py --config config/recipes/task_rewrite_example.yaml

    # Customize batch size and strategy
    ./scripts/generate_adaptive.py --config config/recipes/qa_pairs_example.yaml \
        --batch-size 5 --strategy epsilon_greedy

    # Save output and feedback state
    ./scripts/generate_adaptive.py --config config/recipes/task_rewrite_example.yaml \
        --output data/generated/ --save-state

    # Disable quality filtering to see all generated samples
    ./scripts/generate_adaptive.py --config config/recipes/task_rewrite_example.yaml \
        --no-filter

Arguments:
    --config: Path to YAML config file (required)
    --output: Output directory for generated samples (optional)
    --batch-size: Number of samples per batch (default: 5)
    --strategy: Routing strategy: epsilon_greedy, thompson_sampling, ucb (default: epsilon_greedy)
    --initial-temp: Initial temperature (default: 0.7)
    --initial-exploration: Initial exploration rate (default: 0.1)
    --no-filter: Disable quality filtering (keep all samples)
    --save-state: Save final LocalFeedbackState to output directory
    --verbose: Enable verbose logging
"""

import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv

from src.core.adaptive_pipeline import AdaptivePipeline
from src.core.config_loader import ConfigLoader
from src.core.feedback import FeedbackEngine
from src.core.spec import LocalFeedbackState
from src.quality.orchestrator import QualityAssessmentOrchestrator
from src.router.router import Router


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Generate synthetic data with adaptive feedback loop"
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument("--output", type=Path, help="Output directory for generated samples")
    parser.add_argument("--batch-size", type=int, default=5, help="Samples per batch")
    parser.add_argument(
        "--strategy",
        type=str,
        default="epsilon_greedy",
        choices=["epsilon_greedy", "thompson_sampling", "ucb"],
        help="Routing strategy"
    )
    parser.add_argument("--initial-temp", type=float, default=0.7, help="Initial temperature")
    parser.add_argument(
        "--initial-exploration", type=float, default=0.1, help="Initial exploration rate"
    )
    parser.add_argument(
        "--no-filter", action="store_true", help="Disable quality filtering"
    )
    parser.add_argument(
        "--save-state", action="store_true", help="Save LocalFeedbackState to output"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)

    # Load environment variables
    load_dotenv()

    # Load configuration
    logger.info(f"Loading config from {args.config}...")
    spec = ConfigLoader.load_spec(args.config)

    print("\n" + "="*70)
    print("ADAPTIVE SYNTHETIC DATA GENERATION PIPELINE")
    print("="*70)
    print(f"Domain:          {spec.domain.value}")
    print(f"Task Input:      {spec.task_input}")
    print(f"Total Samples:   {spec.num_samples}")
    print(f"Batch Size:      {args.batch_size}")
    print(f"Strategy:        {args.strategy}")
    print(f"Initial Temp:    {args.initial_temp}")
    print(f"Exploration:     {args.initial_exploration}")
    print(f"Constraints:     {spec.constraints}")
    print("="*70 + "\n")

    # Initialize components
    logger.info("Initializing adaptive pipeline components...")

    # Create initial feedback state
    initial_state = LocalFeedbackState(
        current_temperature=args.initial_temp,
        exploration_rate=args.initial_exploration,
    )

    # Create router
    router = Router(
        default_batch_size=args.batch_size,
    )

    # Create feedback engine
    feedback_engine = FeedbackEngine(
        temperature_adaptation=True,
        exploration_decay=True,
        max_history_length=10,
    )

    # Create quality orchestrator (optional, but recommended)
    quality_orchestrator = QualityAssessmentOrchestrator()

    # Create adaptive pipeline
    pipeline = AdaptivePipeline(
        router=router,
        feedback_engine=feedback_engine,
        quality_orchestrator=quality_orchestrator if not args.no_filter else None,
    )

    # Run adaptive pipeline
    logger.info("Starting adaptive generation pipeline...\n")
    samples, final_state = pipeline.run(
        spec=spec,
        initial_state=initial_state,
        max_iterations=100,
    )

    # Display results
    print("\n" + "="*70)
    print("GENERATION COMPLETE")
    print("="*70)
    print(f"Total Samples Generated:  {len(samples)}")
    print(f"Total Iterations:         {final_state.iteration}")
    print(f"Final Temperature:        {final_state.current_temperature:.3f}")
    print(f"Final Exploration Rate:   {final_state.exploration_rate:.3f}")
    print("="*70)

    # Display arm statistics
    print("\nARM PERFORMANCE STATISTICS:")
    print("-" * 70)
    arm_stats = feedback_engine.get_arm_statistics(final_state)
    for arm_name, stats in arm_stats.items():
        print(f"  {arm_name}:")
        print(f"    Mean Reward:  {stats['mean_reward']:.3f}")
        print(f"    Std Reward:   {stats['std_reward']:.3f}")
        print(f"    Count:        {stats['count']}")

    # Display recent batch metrics
    if final_state.recent_metrics:
        print("\nRECENT BATCH METRICS:")
        print("-" * 70)
        for i, metrics in enumerate(final_state.recent_metrics[-5:], 1):  # Last 5 batches
            print(f"  Batch {i}:")
            print(f"    Samples:      {metrics.num_samples}")
            print(f"    Pass Rate:    {metrics.pass_rate:.2f}")
            if metrics.mean_quality:
                print(f"    Mean Quality: {metrics.mean_quality:.3f}")
            if metrics.diversity_score:
                print(f"    Diversity:    {metrics.diversity_score:.3f}")

    # Display sample examples
    print("\n" + "="*70)
    print(f"SAMPLE EXAMPLES (showing first 3 of {len(samples)}):")
    print("="*70)

    for i, sample in enumerate(samples[:3], 1):
        print(f"\nSample {i}:")
        print(f"  Content:      {sample.content}")
        print(f"  Generator:    {sample.lineage.generator}")
        print(f"  Temperature:  {sample.lineage.generator_parameters.get('temperature', 'N/A')}")
        print(f"  Quality:      {sample.quality_scores}")

    # Save to output directory if specified
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

        # Save samples
        output_file = args.output / "samples.jsonl"
        logger.info(f"\nSaving {len(samples)} samples to {output_file}...")
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

        # Save feedback state if requested
        if args.save_state:
            state_file = args.output / "feedback_state.json"
            logger.info(f"Saving final feedback state to {state_file}...")
            with open(state_file, "w") as f:
                # Convert state to dict (Pydantic model)
                state_dict = final_state.model_dump()
                # Convert recent_metrics to dicts
                state_dict["recent_metrics"] = [
                    m.model_dump() for m in final_state.recent_metrics
                ]
                json.dump(state_dict, f, indent=2)

        print(f"\n✓ Successfully saved outputs to {args.output}")

    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
