#!/usr/bin/env python3
"""
Simple test to validate the adaptive pipeline logic without requiring API dependencies.
"""

import sys
sys.path.insert(0, '/home/user/adaptable_synthdatagen_system')

# Test data models
print("Testing data models...")
from src.core.spec import GenerationPlan, LocalFeedbackState, BatchMetrics, Sample, Lineage, Spec, Domain

# Test 1: Create LocalFeedbackState
print("✓ Test 1: LocalFeedbackState")
state = LocalFeedbackState()
assert state.generated_so_far == 0
assert state.iteration == 0
assert state.current_temperature == 0.7
assert state.exploration_rate == 0.1
print(f"  Initial state: temp={state.current_temperature}, exploration={state.exploration_rate}")

# Test 2: Create BatchMetrics
print("✓ Test 2: BatchMetrics")
metrics = BatchMetrics(
    mean_similarity=0.85,
    diversity_score=0.72,
    mean_quality=0.80,
    pass_rate=0.9,
    num_samples=5
)
assert metrics.mean_quality == 0.80
assert metrics.pass_rate == 0.9
print(f"  Metrics: quality={metrics.mean_quality}, pass_rate={metrics.pass_rate}")

# Test 3: Create GenerationPlan
print("✓ Test 3: GenerationPlan")
from src.core.generator_types import GeneratorType
plan = GenerationPlan(
    batch_size=5,
    generator_arm=GeneratorType.NAIVE,
    parameters={"temperature": 0.7},
    iteration=0,
    reasoning="Initial batch"
)
assert plan.batch_size == 5
assert plan.generator_arm == GeneratorType.NAIVE
print(f"  Plan: batch_size={plan.batch_size}, arm={plan.generator_arm}")

# Test 4: FeedbackEngine logic
print("✓ Test 4: FeedbackEngine")
from src.core.feedback import FeedbackEngine

engine = FeedbackEngine()

# Create sample with quality scores
sample = Sample(
    content="Test sample",
    lineage=Lineage(generator=GeneratorType.NAIVE),
    quality_scores={"similarity": 0.85, "diversity": 0.72}
)

# Compute metrics
batch_metrics = engine.compute_batch_metrics(
    samples=[sample],
    total_generated=1
)
assert batch_metrics.num_samples == 1
assert batch_metrics.pass_rate == 1.0
assert batch_metrics.mean_similarity == 0.85
print(f"  Computed metrics: similarity={batch_metrics.mean_similarity}, pass_rate={batch_metrics.pass_rate}")

# Update state
state = engine.update_feedback_state(
    state=state,
    plan=plan,
    batch_metrics=batch_metrics,
    samples=[sample]
)
assert state.iteration == 1
assert state.generated_so_far == 1
print(f"  Arm counts: {state.arm_counts}")
assert "GeneratorType.NAIVE" in state.arm_counts or "naive" in state.arm_counts
print(f"  Updated state: iteration={state.iteration}, generated={state.generated_so_far}")

# Test 5: Temperature adaptation
print("✓ Test 5: Temperature adaptation")
# Low quality should decrease temperature
low_quality_metrics = BatchMetrics(
    mean_quality=0.5,
    pass_rate=0.6,
    num_samples=5
)
plan2 = GenerationPlan(
    batch_size=5,
    generator_arm=GeneratorType.NAIVE,
    parameters={"temperature": 0.7},
    iteration=1
)
state = engine.update_feedback_state(
    state=state,
    plan=plan2,
    batch_metrics=low_quality_metrics,
    samples=[]
)
print(f"  Low quality → temp adjusted to {state.current_temperature}")
assert state.current_temperature < 0.7  # Should have decreased

# Test 6: AdaptiveRouter epsilon-greedy logic
print("✓ Test 6: AdaptiveRouter")
from src.router.adaptive_router import AdaptiveRouter

router = AdaptiveRouter(default_batch_size=5)
spec = Spec(
    domain=Domain.TASK_REWRITE,
    task_input="Test input",
    num_samples=10,
)

# Initial state - should explore or randomly select
state_initial = LocalFeedbackState()
plan = router.plan_next_batch(spec, state_initial)
assert plan.batch_size > 0
assert plan.batch_size <= 5
assert plan.iteration == 0
print(f"  Initial plan: batch_size={plan.batch_size}, reasoning={plan.reasoning[:50]}...")

# State with history - test exploitation
state_with_history = LocalFeedbackState(
    exploration_rate=0.0,  # No exploration
    arm_rewards={
        GeneratorType.NAIVE.value: [0.8, 0.85, 0.9]
    },
    arm_counts={
        GeneratorType.NAIVE.value: 3
    }
)
plan = router.plan_next_batch(spec, state_with_history)
# With epsilon=0, should always exploit (select NAIVE)
assert plan.generator_arm == GeneratorType.NAIVE
print(f"  Exploitation plan: arm={plan.generator_arm}, reasoning={plan.reasoning[:50]}...")

# Test 7: Arm statistics
print("✓ Test 7: Arm statistics")
stats = engine.get_arm_statistics(state_with_history)
assert GeneratorType.NAIVE.value in stats
assert stats[GeneratorType.NAIVE.value]["mean_reward"] == 0.85
assert stats[GeneratorType.NAIVE.value]["count"] == 3
print(f"  Arm stats: {stats}")

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nCore adaptive pipeline logic is working correctly:")
print("  • Data models: GenerationPlan, LocalFeedbackState, BatchMetrics")
print("  • FeedbackEngine: metrics computation, state updates, temperature adaptation")
print("  • AdaptiveRouter: epsilon-greedy strategy, arm selection")
print("  • Integration: feedback loop state transitions")
print("\nThe pipeline is ready to run with API dependencies installed.")
