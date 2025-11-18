import logging
import random

from src.core.feedback import FeedbackEngine
from src.core.generator_types import GeneratorType
from src.core.models import GenerationContext, GenerationPlan, LocalFeedbackState, Sample, Spec
from src.core.type_guards import (
    is_batch_input_dict,
    is_valid_generator_arm_string,
    is_valid_generator_type_string,
)
from src.generators.naive_generator import NaiveGenerator
from src.quality.orchestrator import QualityAssessmentOrchestrator
from src.router import Router
from src.router.context_extractor import ContextExtractor

logger = logging.getLogger(__name__)


class Pipeline:
    """Main orchestration pipeline for synthetic data generation."""

    def __init__(
        self,
        feedback_engine: FeedbackEngine,
        quality_orchestrator: QualityAssessmentOrchestrator,
    ):
        self.router = Router()
        self.context_extractor = ContextExtractor()
        self.feedback_engine = feedback_engine
        self.quality_orchestrator = quality_orchestrator

        # TODO: Replace hardcoded mapping with centralized registry for dynamic component lookup - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/15
        self.generators = {
            GeneratorType.NAIVE: NaiveGenerator,
            # GeneratorType.WIZARDLM: WizardLMGenerator,
            # GeneratorType.TEMPLATER: TemplaterGenerator,
        }

    def run(
        self,
        spec: Spec,
        initial_state: LocalFeedbackState,
        max_iterations: int = 100,
    ) -> tuple[list[Sample], list[Sample], LocalFeedbackState]:
        """Execute adaptive pipeline with iterative feedback loop."""
        # Detect batch input mode (CSV) using type guard
        if is_batch_input_dict(spec.task_input):
            return self._run_from_batch_input(spec, initial_state, max_iterations)
        else:
            return self._run_from_single_input(spec, initial_state, max_iterations)

    def _run_from_single_input(
        self,
        spec: Spec,
        initial_state: LocalFeedbackState,
        max_iterations: int = 100,
    ) -> tuple[list[Sample], list[Sample], LocalFeedbackState]:
        """Execute pipeline from a single task input (direct from spec)."""
        # Build intelligent context from spec
        context = self.context_extractor.extract(spec)

        # Initialize local feedback state for this spec/job
        state = initial_state
        collected = []
        rejected = []

        logger.info(f"Starting adaptive pipeline for {spec.num_samples} samples")

        iteration = 0
        while len(collected) < spec.num_samples and iteration < max_iterations:
            iteration += 1

            # Update context progress
            context = context.update_progress(
                collected=len(collected),
                rejected=len(rejected),
                iteration=iteration,
            )

            # Get GenerationPlan from Router
            plan = self.router.route(context, state)

            logger.info(
                f"Iteration {iteration}: Batch size: {plan.batch_size} | "
                f"Temperature: {plan.parameters.get('temperature', 'N/A')}"
            )

            # Generate batch
            try:
                batch = self._generate_batch(context, plan)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                continue

            # Filter and score batch to get accepted samples
            accepted = self._filter_and_score(context, batch)

            # Identify rejected samples
            accepted_ids = {id(s) for s in accepted}
            batch_rejected = [s for s in batch if id(s) not in accepted_ids]
            rejected.extend(batch_rejected)

            # Compute batch metrics
            batch_metrics = self.feedback_engine.compute_batch_metrics(
                samples=accepted,
                total_generated=len(batch),
            )
            reward = self.feedback_engine.compute_reward(batch_metrics)
            quality_str = (
                f"{batch_metrics.mean_quality:.2f}"
                if batch_metrics.mean_quality is not None
                else "None"
            )

            logger.info(
                f"Batch metrics: {batch_metrics.num_samples} accepted, "
                f"pass_rate={batch_metrics.pass_rate:.2f}, "
                f"quality_score={quality_str}, "
                f"reward={reward:.2f}"
            )

            # Update local feedback state
            state = self.feedback_engine.update_feedback_state(
                state=state,
                plan=plan,
                batch_metrics=batch_metrics,
                samples=accepted,
            )

            # Adapt generation parameters
            state = self.router.adapt(state=state, metrics=batch_metrics)

            # Collect accepted samples (randomly select if we have more than needed)
            remaining_needed = spec.num_samples - len(collected)
            if len(accepted) > remaining_needed:
                selected = random.sample(accepted, remaining_needed)
                logger.info(
                    f"Randomly selected {remaining_needed} samples from "
                    f"{len(accepted)} valid samples"
                )
                collected.extend(selected)
            else:
                collected.extend(accepted)

            # Safety check
            if iteration >= max_iterations:
                logger.warning(f"Reached max iterations ({max_iterations})")
                break

        # Log final statistics
        arm_stats = self.feedback_engine.get_arm_statistics(state)
        logger.info(f"Pipeline complete: {len(collected)} accepted, {len(rejected)} rejected")
        logger.info(f"Arm statistics: {arm_stats}")

        return collected, rejected, state

    def _generate_batch(self, context: GenerationContext, plan: GenerationPlan) -> list[Sample]:
        """Generate a batch of samples according to context and plan."""
        # Get generator type from arm or directly
        generator_arm = plan.generator_arm

        # Resolve generator type using type guards
        if is_valid_generator_arm_string(generator_arm):
            # Check if arm is a configured arm name (e.g., "naive_conservative")
            if generator_arm in self.router.arms:
                generator_type = self.router.arms[generator_arm]["generator"]
            elif is_valid_generator_type_string(generator_arm):
                generator_type = GeneratorType(generator_arm)
            else:
                raise ValueError(
                    f"Invalid generator arm '{generator_arm}'. "
                    f"Must be a configured arm name or valid GeneratorType. "
                    f"Available arms: {list(self.router.arms.keys())}, "
                    f"Valid types: {[gt.value for gt in GeneratorType]}"
                )
        else:
            # Already a GeneratorType enum
            generator_type = generator_arm

        generator_class = self.generators.get(generator_type)
        if not generator_class:
            raise ValueError(
                f"Unknown generator: {generator_type}. "
                f"Available generators: {list(self.generators.keys())}"
            )

        # Instantiate and run generator with context and plan
        generator = generator_class(context, plan)
        samples = generator.generate()

        return samples

    def _filter_and_score(self, context: GenerationContext, samples: list[Sample]) -> list[Sample]:
        """Filter and score batch using quality orchestrator."""
        # Create a minimal Spec from context for validator compatibility
        spec = Spec(
            domain=context.domain,
            task_input=context.task_input,
            num_samples=context.num_samples,
            constraints=context.constraints,
        )

        # Run all validators and populate quality_scores
        samples = self.quality_orchestrator.assess(samples, spec)

        # Filter out samples that failed validation
        accepted = self.quality_orchestrator.filter_failing_samples(samples)

        return accepted

    # TODO: Extract dataset-level context once before row loop - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/13
    def _run_from_batch_input(
        self,
        spec: Spec,
        initial_state: LocalFeedbackState,
        max_iterations: int = 100,
    ) -> tuple[list[Sample], list[Sample], LocalFeedbackState]:
        """Execute pipeline from batch input (CSV file with multiple task inputs)."""
        from src.utils.batch_processor import BatchProcessor

        batch_info, row_specs_iterator = BatchProcessor.read_row_specs(spec)
        logger.info(f"Batch input mode detected ({batch_info.format.upper()})")

        # Process each row
        all_accepted = []
        all_rejected = []
        final_state = initial_state

        for idx, row_spec, row_samples, row in row_specs_iterator:
            logger.info(f"Row {idx + 1}: Generating {row_samples} variants")

            # Run generation for this row
            accepted, rejected, final_state = self._run_from_single_input(
                row_spec, final_state, max_iterations
            )

            # Enrich samples with batch row data
            for sample in accepted:
                sample.metadata["batch_row"] = row
                sample.metadata["original_input"] = row_spec.task_input["original_input"]
                sample.metadata["expected_output"] = row_spec.task_input["expected_output"]

            all_accepted.extend(accepted)
            all_rejected.extend(rejected)

        # Write CSV output
        if spec.output_path:
            BatchProcessor.write_results(batch_info, all_accepted, spec.output_path)

        logger.info(
            f"{batch_info.format.upper()} batch complete: "
            f"{len(all_accepted)} accepted, {len(all_rejected)} rejected"
        )

        return all_accepted, all_rejected, final_state
