import logging

from src.core.feedback import FeedbackEngine
from src.core.generator_types import GeneratorType
from src.core.models import GenerationContext, GenerationPlan, LocalFeedbackState, Sample, Spec
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
        # Detect batch input mode (CSV)
        is_batch_input = isinstance(spec.task_input, dict) and "input_file" in spec.task_input

        if is_batch_input:
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

            logger.info(
                f"Batch metrics: {batch_metrics.num_samples} accepted, "
                f"pass_rate={batch_metrics.pass_rate:.2f}, "
                f"mean_quality={batch_metrics.mean_quality or 'N/A'}"
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

            # Collect accepted samples
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

        # Check if arm is a configured arm name (e.g., "naive_conservative")
        if isinstance(generator_arm, str) and generator_arm in self.router.arms:
            generator_type = self.router.arms[generator_arm]["generator"]
        elif isinstance(generator_arm, str):
            generator_type = GeneratorType(generator_arm)
        else:
            generator_type = generator_arm

        generator_class = self.generators.get(generator_type)
        if not generator_class:
            raise ValueError(f"Unknown generator: {generator_type}")

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
        from src.utils.csv_batch_processor import CSVBatchProcessor

        logger.info("Batch input mode detected (CSV)")

        # Read CSV and get row specs
        original_df, row_specs_iterator = CSVBatchProcessor.read_row_specs(spec)
        task_input = spec.task_input
        input_column = task_input["input_column"]

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

            # Enrich samples with CSV row data
            for sample in accepted:
                sample.metadata["csv_row"] = row.to_dict()
                sample.metadata["original_input"] = row_spec.task_input["original_input"]
                sample.metadata["expected_output"] = row_spec.task_input["expected_output"]

            all_accepted.extend(accepted)
            all_rejected.extend(rejected)

        # Write CSV output
        if spec.output_path:
            CSVBatchProcessor.write_results(
                original_df, all_accepted, spec.output_path, input_column
            )

        logger.info(
            f"CSV batch complete: {len(all_accepted)} accepted, {len(all_rejected)} rejected"
        )

        return all_accepted, all_rejected, final_state
