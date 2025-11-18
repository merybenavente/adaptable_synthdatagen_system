#!/usr/bin/env python3
"""End-to-end demo script showcasing adaptive data generation.

This script demonstrates:
1. The complete pipeline from spec to generated samples
2. Bandit learning and adaptation over iterations
3. Arm selection reasoning and performance tracking
4. Visual output showing the learning process

Usage:
    ./scripts/demo.py --config config/recipes/task_rewrite_example.yaml --output data/generated
"""

import argparse
import json
import logging
import re
import warnings
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from src.core.config_loader import ConfigLoader
from src.core.feedback import FeedbackEngine
from src.core.models import LocalFeedbackState, Sample, Spec
from src.core.pipeline import Pipeline
from src.quality.orchestrator import QualityAssessmentOrchestrator
from src.utils.logger import Colors

# Suppress transformers sentencepiece warning
warnings.filterwarnings(
    "ignore",
    message=".*sentencepiece tokenizer.*byte fallback.*",
    category=UserWarning,
    module="transformers",
)


class DemoLogger:
    """Enhanced logger for demo output with visual formatting."""

    def __init__(self):
        self.iteration_logs = []
        self.arm_selections = []
        self.rewards_history = defaultdict(list)

    def log_iteration(
        self,
        iteration: int,
        plan_reasoning: str | None,
        batch_metrics,
        arm_name: str,
        exploration_rate: float,
    ):
        """Log iteration details for visualization."""
        log_entry = {
            "iteration": iteration,
            "arm": arm_name,
            "reasoning": plan_reasoning or "No reasoning provided",
            "exploration_rate": exploration_rate,
            "pass_rate": batch_metrics.pass_rate,
            "mean_quality": batch_metrics.mean_quality,
            "num_accepted": batch_metrics.num_samples,
        }
        self.iteration_logs.append(log_entry)
        self.arm_selections.append(arm_name)
        if batch_metrics.mean_quality:
            self.rewards_history[arm_name].append(batch_metrics.mean_quality)

    def print_header(self):
        """Print demo header."""
        print(f"\n{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(
            f"{Colors.BOLD}"
            f"{' ' * 25}ADAPTIVE DATA GENERATION DEMO{Colors.RESET}"
        )
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print("\nThis demo showcases:")
        print(f"  {Colors.GREEN}•{Colors.RESET} Multi-armed bandit learning")
        print(
            f"  {Colors.GREEN}•{Colors.RESET} "
            f"Adaptive arm selection (exploration vs exploitation)"
        )
        print(f"  {Colors.GREEN}•{Colors.RESET} Quality-based feedback loop")
        print(f"  {Colors.GREEN}•{Colors.RESET} Real-time adaptation")
        print(f"\n{Colors.DIM}{'-' * 80}{Colors.RESET}\n")

    def print_iteration_summary(self, iteration: int, log_entry: dict):
        """Print formatted iteration summary."""
        print(f"\n{Colors.BRIGHT_BLUE}{Colors.BOLD}{'─' * 80}{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}ITERATION {iteration}{Colors.RESET}")
        print(f"{Colors.BRIGHT_BLUE}{Colors.BOLD}{'─' * 80}{Colors.RESET}")
        arm_color = f"{Colors.BRIGHT_YELLOW}{Colors.BOLD}"
        print(
            f"{Colors.YELLOW}Arm Selected:{Colors.RESET} "
            f"{arm_color}{log_entry['arm']}{Colors.RESET}"
        )
        print(
            f"{Colors.YELLOW}Reasoning:{Colors.RESET} "
            f"{Colors.CYAN}{log_entry['reasoning']}{Colors.RESET}"
        )
        print(
            f"{Colors.YELLOW}Exploration Rate:{Colors.RESET} "
            f"{Colors.MAGENTA}{log_entry['exploration_rate']:.3f}{Colors.RESET}"
        )
        print(f"{Colors.YELLOW}Results:{Colors.RESET}")
        num_accepted = log_entry['num_accepted']
        print(
            f"  {Colors.GREEN}•{Colors.RESET} Accepted: "
            f"{Colors.BRIGHT_GREEN}{num_accepted} samples{Colors.RESET}"
        )
        pass_rate = log_entry['pass_rate']
        print(
            f"  {Colors.GREEN}•{Colors.RESET} Pass Rate: "
            f"{Colors.BRIGHT_GREEN}{pass_rate:.2%}{Colors.RESET}"
        )
        if log_entry['mean_quality']:
            mean_quality = log_entry['mean_quality']
            print(
                f"  {Colors.GREEN}•{Colors.RESET} Mean Quality: "
                f"{Colors.BRIGHT_GREEN}{mean_quality:.3f}{Colors.RESET}"
            )

    def print_quality_evaluation(
        self,
        batch: list[Sample],
        accepted: list[Sample],
        quality_orchestrator: QualityAssessmentOrchestrator,
    ):
        """Print quality evaluation details for the batch."""
        if not batch:
            return

        print(f"\n{Colors.CYAN}{Colors.BOLD}Quality Evaluation:{Colors.RESET}")

        # Get validator names
        validator_names = list(quality_orchestrator.validators.keys())
        if not validator_names:
            print(f"  {Colors.DIM}No validators configured{Colors.RESET}")
            return

        # Count passed/failed per validator
        total_samples = len(batch)
        accepted_ids = {id(s) for s in accepted}

        for validator_name in validator_names:
            validator = quality_orchestrator.validators[validator_name]

            # For sample-level validators, count per sample
            if validator.is_sample_level():
                passed_count = 0
                failed_count = 0
                scores = []

                for sample in batch:
                    validation_results = sample.metadata.get(
                        "validation_results", {}
                    )
                    validation_result = validation_results.get(validator_name)
                    if validation_result:
                        if validation_result.get("passed", False):
                            passed_count += 1
                        else:
                            failed_count += 1
                        scores.append(validation_result.get("score", 0.0))

                avg_score = sum(scores) / len(scores) if scores else 0.0
                pass_rate = passed_count / total_samples if total_samples > 0 else 0.0

                # Display with color coding
                if pass_rate >= 0.8:
                    status_color = Colors.BRIGHT_GREEN
                elif pass_rate >= 0.5:
                    status_color = Colors.BRIGHT_YELLOW
                else:
                    status_color = Colors.BRIGHT_RED
                print(
                    f"  {Colors.YELLOW}•{Colors.RESET} "
                    f"{Colors.CYAN}{validator_name}:{Colors.RESET}"
                )
                print(
                    f"    {Colors.DIM}Passed:{Colors.RESET} "
                    f"{status_color}{passed_count}/{total_samples}{Colors.RESET} "
                    f"({pass_rate:.1%})"
                )
                print(
                    f"    {Colors.DIM}Avg Score:{Colors.RESET} "
                    f"{Colors.WHITE}{avg_score:.3f}{Colors.RESET}"
                )

                # Show metadata if available (e.g., similarity_score, entailment_score)
                if scores:
                    validation_results = batch[0].metadata.get(
                        "validation_results", {}
                    )
                    first_result = validation_results.get(validator_name, {})
                    metadata = first_result.get("metadata", {})
                    if metadata:
                        for key, value in metadata.items():
                            if isinstance(value, (int, float)):
                                print(
                                    f"    {Colors.DIM}{key}:{Colors.RESET} "
                                    f"{Colors.WHITE}{value:.3f}{Colors.RESET}"
                                )

            # For batch-level validators, show batch result
            elif validator.is_batch_level():
                validation_results = batch[0].metadata.get(
                    "validation_results", {}
                )
                batch_result = validation_results.get(validator_name)
                if batch_result:
                    score = batch_result.get("score", 0.0)
                    passed = batch_result.get("passed", False)
                    status_color = (
                        Colors.BRIGHT_GREEN if passed else Colors.BRIGHT_RED
                    )
                    status_text = "✓ Passed" if passed else "✗ Failed"

                    print(
                        f"  {Colors.YELLOW}•{Colors.RESET} "
                        f"{Colors.CYAN}{validator_name}:{Colors.RESET}"
                    )
                    print(
                        f"    {Colors.DIM}Status:{Colors.RESET} "
                        f"{status_color}{status_text}{Colors.RESET}"
                    )
                    print(
                        f"    {Colors.DIM}Score:{Colors.RESET} "
                        f"{Colors.WHITE}{score:.3f}{Colors.RESET}"
                    )

        # Show rejection reasons for failed samples
        rejected = [s for s in batch if id(s) not in accepted_ids]
        if rejected:
            print(
                f"\n  {Colors.YELLOW}Rejected Samples "
                f"({len(rejected)}):{Colors.RESET}"
            )
            for i, sample in enumerate(rejected[:3], 1):  # Show first 3
                failed_validators = []
                for validator_name in validator_names:
                    validation_results = sample.metadata.get(
                        "validation_results", {}
                    )
                    result = validation_results.get(validator_name)
                    if result and not result.get("passed", True):
                        failed_validators.append(validator_name)

                if failed_validators:
                    if len(sample.content) > 50:
                        content_preview = sample.content[:50] + "..."
                    else:
                        content_preview = sample.content
                    print(
                        f"    {Colors.RED}•{Colors.RESET} "
                        f"{Colors.DIM}{content_preview}{Colors.RESET}"
                    )
                    failed_list = ', '.join(failed_validators)
                    print(
                        f"      {Colors.RED}Failed:{Colors.RESET} "
                        f"{Colors.WHITE}{failed_list}{Colors.RESET}"
                    )

            if len(rejected) > 3:
                remaining = len(rejected) - 3
                print(
                    f"    {Colors.DIM}... and {remaining} more "
                    f"rejected samples{Colors.RESET}"
                )

    def print_generated_batch(self, batch: list[Sample]):
        """Print raw generated samples before quality evaluation."""
        if not batch:
            return

        print(
            f"\n{Colors.CYAN}{Colors.BOLD}"
            f"Generated Batch ({len(batch)} samples):{Colors.RESET}"
        )
        for i, sample in enumerate(batch, 1):
            print(
                f"  {Colors.BRIGHT_WHITE}{i}.{Colors.RESET} "
                f"{Colors.WHITE}{sample.content}{Colors.RESET}"
            )

    def print_accepted_samples_with_stats(
        self,
        all_samples: list[Sample],
        quality_orchestrator: QualityAssessmentOrchestrator,
        original_task_input: str | None = None,
    ):
        """Print all samples with validator stats, one validator per line."""
        if not all_samples:
            return

        # Print original task input if provided
        if original_task_input:
            if isinstance(original_task_input, str):
                print(
                    f"\n\033[36mOriginal Sample:\033[0m "
                    f"\"{original_task_input}\""
                )
            else:
                print(
                    f"\n\033[36mOriginal Sample:\033[0m "
                    f"{original_task_input}"
                )

        print(f"\n\033[36mAll Samples\033[0m ({len(all_samples)}):")

        # Get validator names and their thresholds
        validator_names = list(quality_orchestrator.validators.keys())
        sample_level_validators = [
            name for name in validator_names
            if quality_orchestrator.validators[name].is_sample_level()
        ]
        batch_level_validators = [
            name for name in validator_names
            if quality_orchestrator.validators[name].is_batch_level()
        ]

        # Print each sample with its sample-level validator stats (full content, no truncation)
        for i, sample in enumerate(all_samples, 1):
            print(f"{i}. {sample.content}")

            # Print sample-level validators for this sample
            for validator_name in sample_level_validators:
                validation_results = sample.metadata.get(
                    "validation_results", {}
                )
                result = validation_results.get(validator_name)
                if result:
                    metadata = result.get("metadata", {})
                    skipped = metadata.get("skipped", False)

                    if skipped:
                        # Show skipped status instead of score
                        reason = metadata.get('reason', 'not applicable')
                        print(
                            f"   {Colors.DIM}⊘\033[0m {validator_name}: "
                            f"{Colors.DIM}skipped ({reason}){Colors.RESET}"
                        )
                    else:
                        score = result.get("score", 0.0)
                        passed = result.get("passed", False)
                        threshold = quality_orchestrator.validators[validator_name].threshold
                        status = "✓" if passed else "✗"
                        # Use light green/red for the entire line
                        line_color = Colors.BRIGHT_GREEN if passed else Colors.BRIGHT_RED

                        # Print validator result with entire line colored
                        score_text = f"score={score:.3f} (threshold={threshold:.3f})"
                        print(
                            f"   {line_color}{status} {validator_name}: "
                            f"{score_text}{Colors.RESET}"
                        )

        # Print batch-level validators below all samples
        if batch_level_validators:
            print("\n\033[36mBatch Validators:\033[0m")
            # Find a sample that passed sample-level validation to get batch results
            passed_samples = [
                s for s in all_samples
                if quality_orchestrator._sample_passed_sample_level_validators(s)
            ]

            for validator_name in batch_level_validators:
                # Look for batch results in samples that passed sample-level validation
                result = None
                if passed_samples:
                    validation_results = passed_samples[0].metadata.get(
                        "validation_results", {}
                    )
                    result = validation_results.get(validator_name)

                if result:
                    score = result.get("score", 0.0)
                    passed = result.get("passed", False)
                    threshold = quality_orchestrator.validators[validator_name].threshold
                    status = "✓" if passed else "✗"
                    # Use light green/red for the entire line
                    line_color = Colors.BRIGHT_GREEN if passed else Colors.BRIGHT_RED
                    score_text = f"score={score:.3f} (threshold={threshold:.3f})"
                    print(
                        f"  {line_color}{status} {validator_name}: "
                        f"{score_text}{Colors.RESET}"
                    )

                    # Print LLM judge reasoning if available
                    metadata = result.get("metadata", {})
                    reasoning = metadata.get("reasoning", "")
                    if reasoning and validator_name == "llm_judge":
                        print(
                            f"    {Colors.CYAN}Reasoning:{Colors.RESET} "
                            f"{Colors.WHITE}{reasoning}{Colors.RESET}"
                        )

                        # Also show per-sample evaluations if available
                        per_sample = metadata.get("per_sample_evaluations", [])
                        if per_sample:
                            print(f"    {Colors.CYAN}Per-sample quality levels:{Colors.RESET}")
                            for eval_item in per_sample:
                                idx = eval_item.get("sample_index", -1)
                                quality = eval_item.get("quality_level", 0)
                                justification = eval_item.get("justification", "")
                                # Color code based on quality level
                                if quality >= 4:
                                    quality_color = Colors.BRIGHT_GREEN
                                elif quality == 3:
                                    quality_color = Colors.BRIGHT_YELLOW
                                else:
                                    quality_color = Colors.BRIGHT_RED
                                print(
                                    f"      {Colors.DIM}Sample {idx+1}:{Colors.RESET} "
                                    f"{quality_color}Level {quality}{Colors.RESET} - "
                                    f"{Colors.WHITE}{justification}{Colors.RESET}"
                                )
                else:
                    # No samples passed sample-level validation, so no batch results
                    msg = "not computed (no samples passed sample-level validation)"
                    print(f"  {Colors.DIM}⊘ {validator_name}: {msg}{Colors.RESET}")

    def print_progress_summary(
        self,
        collected: int,
        rejected: int,
        total_needed: int,
        batch_metrics,
        rejected_samples: list[Sample],
        accepted_samples: list[Sample],
        quality_orchestrator: QualityAssessmentOrchestrator,
        router,
        state=None,
        plan=None,
    ):
        """Print progress summary between batches."""
        remaining = max(0, total_needed - collected)
        progress_pct = (collected / total_needed * 100) if total_needed > 0 else 0

        print(f"\n{Colors.CYAN}{Colors.BOLD}{'─' * 80}{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}Progress Summary{Colors.RESET}")
        print(f"{Colors.CYAN}{Colors.BOLD}{'─' * 80}{Colors.RESET}")

        # Accepted samples first
        if accepted_samples:
            print(
                f"\n{Colors.YELLOW}Accepted Samples "
                f"({len(accepted_samples)}):{Colors.RESET}"
            )
            for i, sample in enumerate(accepted_samples, 1):
                content = sample.content
                if isinstance(content, dict):
                    content_preview = json.dumps(content, ensure_ascii=False)[:80]
                else:
                    content_preview = content[:80] + ("..." if len(content) > 80 else "")
                print(
                    f"    {Colors.BRIGHT_GREEN}{i}.{Colors.RESET} "
                    f"{Colors.WHITE}{content_preview}{Colors.RESET}"
                )
                if sample.quality_scores:
                    scores_str = ", ".join(
                        f"{k}={v:.3f}" for k, v in sample.quality_scores.items()
                    )
                    print(f"       {Colors.DIM}Quality: {scores_str}{Colors.RESET}")
        else:
            print(
                f"\n{Colors.GREEN}Accepted Samples:{Colors.RESET} "
                f"{Colors.DIM}None this round{Colors.RESET}"
            )

        # Progress section
        bar_width = 40
        filled = int((collected / total_needed) * bar_width) if total_needed > 0 else 0
        bar = (
            f"{Colors.BRIGHT_GREEN}{'█' * filled}"
            f"{Colors.DIM}{'░' * (bar_width - filled)}{Colors.RESET}"
        )
        print(
            f"\n{Colors.YELLOW}Progress:{Colors.RESET} {bar} "
            f"{Colors.BRIGHT_WHITE}{collected}/{total_needed}{Colors.RESET} "
            f"({progress_pct:.1f}%)"
        )
        print(
            f"  {Colors.GREEN}✅ Collected:{Colors.RESET} "
            f"{Colors.GREEN}{collected}{Colors.RESET}"
        )
        print(
            f"  {Colors.RED}❌ Rejected:{Colors.RESET} "
            f"{Colors.RED}{rejected}{Colors.RESET}"
        )
        print(f"  {Colors.BLUE}⏳ Remaining:{Colors.RESET} {Colors.BLUE}{remaining}{Colors.RESET}")

        # Batch metrics section
        if batch_metrics:
            print(f"\n{Colors.YELLOW}Batch Metrics:{Colors.RESET}")
            print(
                f"  {Colors.GREEN}•{Colors.RESET} Pass Rate: "
                f"{Colors.BRIGHT_GREEN}{batch_metrics.pass_rate:.2%}{Colors.RESET}"
            )
            if batch_metrics.mean_quality:
                print(
                    f"  {Colors.GREEN}•{Colors.RESET} Mean Quality: "
                    f"{Colors.BRIGHT_GREEN}{batch_metrics.mean_quality:.3f}{Colors.RESET}"
                )
            if batch_metrics.mean_similarity:
                print(
                    f"  {Colors.GREEN}•{Colors.RESET} Mean Similarity: "
                    f"{Colors.BRIGHT_GREEN}{batch_metrics.mean_similarity:.3f}{Colors.RESET}"
                )
            if batch_metrics.diversity_score:
                print(
                    f"  {Colors.GREEN}•{Colors.RESET} Diversity Score: "
                    f"{Colors.BRIGHT_GREEN}{batch_metrics.diversity_score:.3f}{Colors.RESET}"
                )
        else:
            print(
                f"\n{Colors.YELLOW}Batch Metrics:{Colors.RESET} "
                f"{Colors.DIM}Pending{Colors.RESET}"
            )

        # Feedback / Reward section
        print(f"\n{Colors.MAGENTA}Feedback / Reward:{Colors.RESET}")

        # Show reward calculation for this batch
        if batch_metrics and plan:
            pass_rate = batch_metrics.pass_rate if batch_metrics.pass_rate is not None else 0.0
            quality = batch_metrics.mean_quality if batch_metrics.mean_quality is not None else 0.0
            reward = pass_rate * quality
            print(
                f"  {Colors.WHITE}- this batch reward:{Colors.RESET} "
                f"{Colors.BRIGHT_WHITE}{reward:.3f}{Colors.RESET} "
                f"{Colors.DIM}(pass_rate={pass_rate:.2%} × quality={quality:.3f}){Colors.RESET}"
            )

        # Show current arm statistics after update (show ALL available arms)
        if state:
            print(f"  {Colors.WHITE}- arm rewards (mean):{Colors.RESET}")
            # Get all available arms from router
            all_arms = list(router.arms.keys())

            # Build stats for all arms (used and unused)
            arm_stats = []
            for arm_name in all_arms:
                rewards = state.arm_rewards.get(arm_name, [])
                if rewards:
                    mean_reward = sum(rewards) / len(rewards)
                    arm_stats.append((arm_name, mean_reward, len(rewards), True))
                else:
                    # Arm not yet used
                    arm_stats.append((arm_name, 0.0, 0, False))

            # Sort: used arms by reward (desc), then unused arms
            arm_stats.sort(key=lambda x: (x[3], x[1]), reverse=True)

            # Find best arm (highest mean reward among used arms)
            used_arms = [a for a in arm_stats if a[3]]
            best_arm = used_arms[0][0] if used_arms else None

            for arm_name, mean_reward, count, is_used in arm_stats:
                # Mark current batch arm in cyan, best arm in yellow
                if plan and arm_name == plan.generator_arm:
                    arm_display = f"  {Colors.CYAN}{arm_name}{Colors.RESET}"
                elif arm_name == best_arm and mean_reward > 0:
                    arm_display = f"  {Colors.YELLOW}★{Colors.RESET} {arm_name}"
                else:
                    arm_display = f"  {arm_name}"

                # Show N/A for unused arms
                if is_used:
                    reward_display = f"{Colors.BRIGHT_WHITE}{mean_reward:.3f}{Colors.RESET}"
                else:
                    reward_display = f"{Colors.DIM}N/A{Colors.RESET}"

                print(
                    f"    {arm_display}: "
                    f"{reward_display} "
                    f"{Colors.DIM}(n={count}){Colors.RESET}"
                )
        else:
            print(f"  {Colors.DIM}- arm rewards: not available yet{Colors.RESET}")

        # Note: Detailed sample validation stats are printed in print_accepted_samples_with_stats
        # No need to duplicate rejection reasons here - they're already shown with all samples

        print(f"{Colors.CYAN}{Colors.BOLD}{'─' * 80}{Colors.RESET}\n")

    def print_bandit_reasoning_summary(self):
        """Print summary of bandit reasoning decisions over time."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}BANDIT REASONING - Decision History{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print()

        if not self.iteration_logs:
            print(f"{Colors.YELLOW}No iteration logs available.{Colors.RESET}")
            return

        # Group consecutive same arms to reduce verbosity
        current_arm = None
        start_iter = None
        for log in self.iteration_logs:
            iteration = log['iteration']
            arm = log['arm']
            reasoning = log['reasoning']
            exploration_rate = log['exploration_rate']

            # Check if arm changed
            if arm != current_arm:
                # Print reasoning for arm change
                print(f"{Colors.BOLD}{Colors.YELLOW}Iteration {iteration}:{Colors.RESET}")
                print(f"  {Colors.CYAN}Arm:{Colors.RESET} {Colors.BRIGHT_WHITE}{arm}{Colors.RESET}")
                exploration_display = f"{Colors.MAGENTA}{exploration_rate:.3f}{Colors.RESET}"
                print(f"  {Colors.CYAN}Exploration Rate:{Colors.RESET} {exploration_display}")
                reasoning_display = f"{Colors.WHITE}{reasoning}{Colors.RESET}"
                print(f"  {Colors.CYAN}Reasoning:{Colors.RESET} {reasoning_display}")
                print()
                current_arm = arm
                start_iter = iteration
            # For same arm selections, only print every 3rd iteration or exploration changes
            elif (
                iteration - start_iter >= 3
                or abs(
                    exploration_rate
                    - self.iteration_logs[iteration - 2]['exploration_rate']
                )
                > 0.05
            ):
                print(f"{Colors.DIM}Iteration {iteration}:{Colors.RESET}")
                print(f"  {Colors.DIM}Arm: {arm} (unchanged){Colors.RESET}")
                print(f"  {Colors.DIM}Reasoning: {reasoning}{Colors.RESET}")
                print()
                start_iter = iteration  # Reset to avoid printing every iteration

    def print_learning_curve(self, final_state):
        """Print ASCII learning curve visualization."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        header_text = "LEARNING CURVE - Arm Selection Over Time"
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{header_text}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"\n{Colors.CYAN}Iteration → Arm Selection:{Colors.RESET}")
        print()

        # Group consecutive selections
        current_arm = None
        start_iter = 0
        for i, arm in enumerate(self.arm_selections):
            if arm != current_arm:
                if current_arm is not None:
                    if i - start_iter > 1:
                        print(f"  [{start_iter}-{i-1}] {current_arm}")
                    else:
                        print(f"  [{start_iter}] {current_arm}")
                current_arm = arm
                start_iter = i

        # Print last segment
        if current_arm is not None:
            if len(self.arm_selections) - start_iter > 1:
                print(f"  [{start_iter}-{len(self.arm_selections)-1}] {current_arm}")
            else:
                print(f"  [{start_iter}] {current_arm}")

    def print_arm_performance(self, arm_stats: dict):
        """Print arm performance comparison."""
        print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}ARM PERFORMANCE SUMMARY{Colors.RESET}")
        print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
        print()

        if not arm_stats:
            print(f"{Colors.YELLOW}No arm statistics available.{Colors.RESET}")
            return

        # Sort by mean reward
        sorted_arms = sorted(
            arm_stats.items(), key=lambda x: x[1]["mean_reward"], reverse=True
        )

        header = f"{'Arm':<25} {'Mean Reward':<15} {'Std Dev':<15} {'Count':<10}"
        print(f"{Colors.BOLD}{header}{Colors.RESET}")
        print(f"{Colors.DIM}{'-' * 80}{Colors.RESET}")

        # Color mapping for arms
        arm_colors = {
            "naive_conservative": Colors.BLUE,
            "naive_balanced": Colors.CYAN,
            "naive_creative": Colors.MAGENTA,
        }

        for arm_name, stats in sorted_arms:
            mean_reward = stats["mean_reward"]
            std_reward = stats["std_reward"]
            count = stats["count"]

            # Color for this arm
            arm_color = arm_colors.get(arm_name, Colors.WHITE)

            # Visual indicator with color
            bar_length = int(mean_reward * 20)
            if mean_reward > 0.7:
                bar_color = Colors.BRIGHT_GREEN
            elif mean_reward > 0.5:
                bar_color = Colors.BRIGHT_YELLOW
            else:
                bar_color = Colors.BRIGHT_RED
            empty_part = f"{Colors.DIM}{'░' * (20 - bar_length)}{Colors.RESET}"
            bar = f"{bar_color}{'█' * bar_length}{empty_part}"

            # Highlight best performer
            is_best = arm_name == sorted_arms[0][0]
            name_style = f"{Colors.BOLD}{arm_color}" if is_best else arm_color

            print(
                f"{name_style}{arm_name:<25}{Colors.RESET} "
                f"{Colors.BRIGHT_WHITE}{mean_reward:<15.3f}{Colors.RESET} "
                f"{Colors.WHITE}{std_reward:<15.3f}{Colors.RESET} "
                f"{Colors.WHITE}{count:<10}{Colors.RESET} {bar}"
            )

    def save_demo_data(self, output_path: Path, final_state: LocalFeedbackState):
        """Save demo data for further analysis."""
        demo_data = {
            "iterations": self.iteration_logs,
            "arm_selections": self.arm_selections,
            "final_state": final_state.model_dump(),
            "rewards_history": {
                arm: rewards for arm, rewards in self.rewards_history.items()
            },
        }

        demo_file = output_path / "demo_data.json"
        with open(demo_file, "w") as f:
            json.dump(demo_data, f, indent=2)
        print(f"\nDemo data saved to: {demo_file}")


def main():
    """Main demo entry point."""
    parser = argparse.ArgumentParser(
        description="Run end-to-end adaptive data generation demo"
    )
    parser.add_argument("--config", type=Path, required=True, help="Path to config file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/generated"),
        help="Output directory (default: data/generated)",
    )
    parser.add_argument("--batch-size", type=int, default=5, help="Samples per batch")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed logging for each iteration",
    )
    args = parser.parse_args()

    load_dotenv()

    # Setup logging with colored output
    log_level = logging.DEBUG if args.verbose else logging.INFO

    class ColoredFormatter(logging.Formatter):
        """Custom formatter that adds colors to log messages."""

        def format(self, record):
            # Create a copy to avoid modifying the original
            record_copy = logging.makeLogRecord(record.__dict__)

            # Check if this is an httpx logger
            is_httpx = record.name.startswith("httpx")

            # For httpx logs, don't add individual colors
            if not is_httpx:
                # Color levelname and name based on log level
                colors_map = {
                    logging.INFO: f"{Colors.DIM}{Colors.BRIGHT_BLACK}",
                    logging.WARNING: Colors.YELLOW,
                    logging.ERROR: Colors.RED,
                    logging.DEBUG: f"{Colors.DIM}{Colors.BLUE}",
                }
                color = colors_map.get(record.levelno, "")
                if color:
                    record_copy.levelname = f"{color}{record.levelname}{Colors.RESET}"
                    if record.levelno == logging.INFO:
                        record_copy.name = f"{color}{record.name}{Colors.RESET}"

            formatted = super().format(record_copy)

            # For non-httpx logs, color timestamp and dashes
            if not is_httpx:
                # Color timestamp
                grey = f"{Colors.DIM}{Colors.BRIGHT_BLACK}"
                formatted = re.sub(
                    r'^(\d{2}:\d{2}:\d{2})',
                    lambda m: f"{grey}{m.group(1)}{Colors.RESET}",
                    formatted,
                    count=1,
                )
                # Color dashes
                parts = formatted.split(' - ', 3)
                if len(parts) >= 2:
                    dash = f"{grey} - {Colors.RESET}"
                    if len(parts) == 4:
                        formatted = f"{parts[0]}{dash}{parts[1]}{dash}{parts[2]}{dash}{parts[3]}"
                    elif len(parts) == 3:
                        formatted = f"{parts[0]}{dash}{parts[1]}{dash}{parts[2]}"
                    else:
                        formatted = f"{parts[0]}{dash}{parts[1]}"

            # For httpx logs, wrap everything in grey
            if is_httpx:
                grey = f"{Colors.DIM}{Colors.BRIGHT_BLACK}"
                formatted = f"{grey}{formatted}{Colors.RESET}"

            return formatted

    # Clear any existing handlers
    logger = logging.getLogger()
    logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(ColoredFormatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S"
    ))

    logger.setLevel(log_level)
    logger.addHandler(handler)

    # Initialize demo logger
    demo_logger = DemoLogger()
    demo_logger.print_header()

    # Load spec
    print(
        f"{Colors.BOLD}📋 Loading configuration from "
        f"{Colors.YELLOW}{args.config}...{Colors.RESET}"
    )
    spec = ConfigLoader.load_spec(args.config)

    # print(f"\n{Colors.BOLD}Configuration:{Colors.RESET}")
    domain_display = f"{Colors.BRIGHT_WHITE}{spec.domain}{Colors.RESET}"
    print(f"  {Colors.YELLOW}Domain:{Colors.RESET} {domain_display}")
    if isinstance(spec.task_input, str):
        task_display = spec.task_input
    else:
        task_display = 'Complex input (see config)'
    print(f"  {Colors.YELLOW}Task:{Colors.RESET} {Colors.WHITE}{task_display}{Colors.RESET}")
    samples_display = f"{Colors.BRIGHT_WHITE}{spec.num_samples}{Colors.RESET}"
    print(f"  {Colors.YELLOW}Samples to generate:{Colors.RESET} {samples_display}")

    # Pretty print constraints if they exist
    if spec.constraints:
        print(f"  {Colors.YELLOW}Constraints:{Colors.RESET}")
        for key, value in spec.constraints.items():
            # Format boolean values nicely
            if isinstance(value, bool):
                value_str = "✓" if value else "✗"
                value_display = f"{Colors.BRIGHT_WHITE}{value_str}{Colors.RESET}"
                key_display = f"{Colors.CYAN}{key}:{Colors.RESET}"
                print(f"    {Colors.GREEN}•{Colors.RESET} {key_display} {value_display}")
            else:
                value_display = f"{Colors.BRIGHT_WHITE}{value}{Colors.RESET}"
                key_display = f"{Colors.CYAN}{key}:{Colors.RESET}"
                print(f"    {Colors.GREEN}•{Colors.RESET} {key_display} {value_display}")
    else:
        print(f"  {Colors.YELLOW}Constraints:{Colors.RESET} {Colors.DIM}None{Colors.RESET}")

    # Create components
    feedback_engine = FeedbackEngine()
    quality_orchestrator = QualityAssessmentOrchestrator()

    # Create initial state
    initial_state = LocalFeedbackState()

    print(f"\n{Colors.BOLD}🚀 Starting adaptive pipeline...{Colors.RESET}")
    batch_display = f"{Colors.BRIGHT_WHITE}{args.batch_size}{Colors.RESET}"
    print(f"  {Colors.YELLOW}Batch size:{Colors.RESET} {batch_display}")
    exploration_display = f"{Colors.BRIGHT_CYAN}{initial_state.exploration_rate:.2f}{Colors.RESET}"
    print(f"  {Colors.YELLOW}Initial exploration rate:{Colors.RESET} {exploration_display}")

    # Create a custom pipeline class that logs iteration details
    class DemoPipeline(Pipeline):
        """Pipeline subclass that logs iteration details for demo."""

        def _filter_and_score(self, context, samples):
            """Override to add quality evaluation logging."""
            # Create a minimal Spec from context for validator compatibility
            spec = Spec(
                domain=context.domain,
                task_input=context.task_input,
                num_samples=context.num_samples,
                constraints=context.constraints,
            )

            # Run all validators and populate quality_scores
            assessed_samples = self.quality_orchestrator.assess(samples, spec)

            # Filter to get accepted samples
            accepted = self.quality_orchestrator.filter_failing_samples(assessed_samples)

            # Print all assessed samples with their validator stats (both accepted and rejected)
            # Get original task input from context
            original_input = context.task_input
            if isinstance(original_input, str):
                original_input_str = original_input
            else:
                # For dict inputs, try to extract a meaningful string representation
                original_input_str = str(original_input)
            # Print all assessed samples, not just accepted ones
            demo_logger.print_accepted_samples_with_stats(
                assessed_samples, self.quality_orchestrator, original_input_str
            )

            # Log quality evaluation details
            if args.verbose:
                demo_logger.print_quality_evaluation(
                    assessed_samples, accepted, self.quality_orchestrator
                )

            # Filter out samples that failed validation
            return accepted

        def _run_from_single_input(
            self, spec, initial_state, max_iterations
        ):
            """Override to capture iteration details."""
            context = self.context_extractor.extract(spec)
            state = initial_state
            collected = []
            rejected = []

            iteration = 0
            while len(collected) < spec.num_samples and iteration < max_iterations:
                iteration += 1

                context = context.update_progress(
                    collected=len(collected),
                    rejected=len(rejected),
                    iteration=iteration,
                )

                plan = self.router.route(context, state)

                # Log iteration start
                if args.verbose:
                    demo_logger.print_iteration_summary(
                        iteration,
                        {
                            "arm": plan.generator_arm,
                            "reasoning": plan.reasoning,
                            "exploration_rate": state.exploration_rate,
                            "pass_rate": 0.0,
                            "mean_quality": None,
                            "num_accepted": 0,
                        },
                    )

                # Generate batch
                try:
                    batch = self._generate_batch(context, plan)
                except Exception as e:
                    logging.error(f"Generation failed: {e}")
                    continue

                # Print generated batch before quality evaluation
                demo_logger.print_generated_batch(batch)

                # Filter and score (includes quality evaluation logging)
                accepted = self._filter_and_score(context, batch)
                accepted_ids = {id(s) for s in accepted}
                batch_rejected = [s for s in batch if id(s) not in accepted_ids]
                rejected.extend(batch_rejected)

                # Compute metrics
                batch_metrics = self.feedback_engine.compute_batch_metrics(
                    samples=accepted,
                    total_generated=len(batch),
                )

                # Log iteration results
                demo_logger.log_iteration(
                    iteration=iteration,
                    plan_reasoning=plan.reasoning,
                    batch_metrics=batch_metrics,
                    arm_name=str(plan.generator_arm),
                    exploration_rate=state.exploration_rate,
                )

                # Update state
                state = self.feedback_engine.update_feedback_state(
                    state=state,
                    plan=plan,
                    batch_metrics=batch_metrics,
                    samples=accepted,
                )
                state = self.router.adapt(state=state, metrics=batch_metrics)

                collected.extend(accepted)

                # Print progress summary between batches
                demo_logger.print_progress_summary(
                    collected=len(collected),
                    rejected=len(rejected),
                    total_needed=spec.num_samples,
                    batch_metrics=batch_metrics,
                    rejected_samples=batch_rejected,
                    accepted_samples=accepted,
                    quality_orchestrator=self.quality_orchestrator,
                    router=self.router,
                    state=state,
                    plan=plan,
                )

                if iteration >= max_iterations:
                    logging.warning(f"Reached max iterations ({max_iterations})")
                    break

            return collected, rejected, state

    # Use demo pipeline
    demo_pipeline = DemoPipeline(
        feedback_engine=feedback_engine,
        quality_orchestrator=quality_orchestrator,
    )
    demo_pipeline.router.default_batch_size = args.batch_size

    # Extract context to build available arms
    from src.router.context_extractor import ContextExtractor
    extractor = ContextExtractor()
    context = extractor.extract(spec)
    available_arms = demo_pipeline.router._build_available_arms(context)

    arms_list = ', '.join(available_arms.keys())
    arms_display = f"{Colors.BRIGHT_CYAN}{arms_list}{Colors.RESET}"
    print(f"  {Colors.YELLOW}Available arms:{Colors.RESET} {arms_display}\n")

    # Run pipeline
    accepted_samples, rejected_samples, final_state = demo_pipeline.run(
        spec=spec,
        initial_state=initial_state,
        max_iterations=100,
    )

    # Print results
    print(f"\n{Colors.BRIGHT_GREEN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}DEMO COMPLETE{Colors.RESET}")
    print(f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")

    print(f"\n{Colors.BOLD}{Colors.CYAN}📊 Final Statistics:{Colors.RESET}")
    accepted_display = f"{Colors.BRIGHT_GREEN}{Colors.BOLD}{len(accepted_samples)}{Colors.RESET}"
    print(f"  {Colors.GREEN}✅ Accepted samples:{Colors.RESET} {accepted_display}")
    rejected_display = f"{Colors.BRIGHT_RED}{len(rejected_samples)}{Colors.RESET}"
    print(f"  {Colors.RED}❌ Rejected samples:{Colors.RESET} {rejected_display}")
    iterations_display = f"{Colors.BRIGHT_BLUE}{final_state.iteration}{Colors.RESET}"
    print(f"  {Colors.BLUE}🔄 Total iterations:{Colors.RESET} {iterations_display}")
    exploration_rate_display = (
        f"{Colors.BRIGHT_MAGENTA}{final_state.exploration_rate:.3f}{Colors.RESET}"
    )
    print(f"  {Colors.MAGENTA}📉 Final exploration rate:{Colors.RESET} {exploration_rate_display}")

    # Get arm statistics
    arm_stats = feedback_engine.get_arm_statistics(final_state)
    demo_logger.print_arm_performance(arm_stats)
    demo_logger.print_learning_curve(final_state)
    demo_logger.print_bandit_reasoning_summary()

    # Display sample examples
    print(f"\n{Colors.BRIGHT_CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}SAMPLE EXAMPLES{Colors.RESET}")
    print(f"{Colors.BRIGHT_CYAN}{Colors.BOLD}{'=' * 80}{Colors.RESET}")
    for i, sample in enumerate(accepted_samples[:3], 1):
        print(f"\n{Colors.BOLD}{Colors.YELLOW}Sample {i}:{Colors.RESET}")
        content = sample.content[:100] + "..." if len(sample.content) > 100 else sample.content
        print(f"  {Colors.CYAN}Content:{Colors.RESET} {Colors.WHITE}{content}{Colors.RESET}")
        generator_display = f"{Colors.BRIGHT_WHITE}{sample.lineage.generator}{Colors.RESET}"
        print(f"  {Colors.CYAN}Generator:{Colors.RESET} {generator_display}")
        temperature = sample.lineage.generator_parameters.get('temperature', 'N/A')
        temp_display = f"{Colors.BRIGHT_WHITE}{temperature}{Colors.RESET}"
        print(f"  {Colors.CYAN}Temperature:{Colors.RESET} {temp_display}")
        scores_display = f"{Colors.WHITE}{sample.quality_scores}{Colors.RESET}"
        print(f"  {Colors.CYAN}Quality Scores:{Colors.RESET} {scores_display}")

    if len(accepted_samples) > 3:
        print(f"\n{Colors.DIM}... and {len(accepted_samples) - 3} more samples{Colors.RESET}")

    # Save outputs
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)

        # Save samples
        output_file = args.output / "accepted_samples.jsonl"
        save_msg = (
            f"\n{Colors.BRIGHT_CYAN}💾 Saving accepted samples to "
            f"{output_file}...{Colors.RESET}"
        )
        print(save_msg)
        with open(output_file, "w") as f:
            for sample in accepted_samples:
                f.write(
                    json.dumps(
                        {
                            "id": str(sample.id),
                            "content": sample.content,
                            "lineage": {
                                "num_of_evolutions": sample.lineage.num_of_evolutions,
                                "generator": str(sample.lineage.generator),
                                "generator_parameters": sample.lineage.generator_parameters,
                            },
                            "metadata": sample.metadata,
                            "quality_scores": sample.quality_scores,
                        }
                    )
                    + "\n"
                )
        success_msg = (
            f"{Colors.BRIGHT_GREEN}✅ Successfully saved "
            f"{len(accepted_samples)} accepted samples{Colors.RESET}"
        )
        print(success_msg)

        # Save rejected samples
        rejected_file = args.output / "rejected_samples.jsonl"
        rejected_msg = (
            f"{Colors.BRIGHT_CYAN}💾 Saving rejected samples to {rejected_file}...{Colors.RESET}"
        )
        print(rejected_msg)
        with open(rejected_file, "w") as f:
            for sample in rejected_samples:
                f.write(
                    json.dumps(
                        {
                            "id": str(sample.id),
                            "content": sample.content,
                            "lineage": {
                                "num_of_evolutions": sample.lineage.num_of_evolutions,
                                "generator": str(sample.lineage.generator),
                                "generator_parameters": sample.lineage.generator_parameters,
                            },
                            "metadata": sample.metadata,
                            "quality_scores": sample.quality_scores,
                        }
                    )
                    + "\n"
                )
        success_rejected_msg = (
            f"{Colors.BRIGHT_GREEN}✅ Successfully saved "
            f"{len(rejected_samples)} rejected samples{Colors.RESET}"
        )
        print(success_rejected_msg)

        # Save state
        state_file = args.output / "feedback_state.json"
        with open(state_file, "w") as f:
            json.dump(final_state.model_dump(), f, indent=2)

        final_msg = (
            f"{Colors.BRIGHT_GREEN}✅ All outputs saved to: "
            f"{Colors.BOLD}{args.output}{Colors.RESET}"
        )
        print(final_msg)


if __name__ == "__main__":
    main()

