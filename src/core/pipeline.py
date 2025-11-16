import logging

from src.core.feedback import FeedbackEngine
from src.core.generator_types import GeneratorType
from src.core.spec import LocalFeedbackState, Sample, Spec
from src.generators.naive_generator import NaiveGenerator
from src.quality.orchestrator import QualityOrchestrator
from src.router import Router
from src.router.context_extractor import ContextExtractor

logger = logging.getLogger(__name__)


class Pipeline:
    """Main orchestration pipeline for synthetic data generation."""

    def __init__(
        self,
        routing_config_path: str | None = None,
        feedback_engine: FeedbackEngine | None = None,
        quality_orchestrator: QualityOrchestrator | None = None,
    ):
        self.router = Router()
        self.context_extractor = ContextExtractor()
        self.feedback_engine = feedback_engine
        self.quality_orchestrator = quality_orchestrator

        # Map generator types to classes
        self.generators = {
            GeneratorType.NAIVE: NaiveGenerator,
            # GeneratorType.WIZARDLM: WizardLMGenerator,
            # GeneratorType.TEMPLATER: TemplaterGenerator,
        }
        # TODO: Use routing_config_path when Router supports configuration

    def run(
        self,
        spec: Spec,
        initial_state: LocalFeedbackState | None = None,
        max_iterations: int = 100,
    ) -> list[Sample] | tuple[list[Sample], LocalFeedbackState]:
        """Execute generation pipeline with optional adaptive feedback loop."""
        if initial_state is not None and self.feedback_engine is not None:
            return self._run_adaptive(spec, initial_state, max_iterations)
        else:
            return self._run_simple(spec)

    def _run_simple(self, spec: Spec) -> list[Sample]:
        """Execute simple generation pipeline: route → generate → return."""
        # 1. Route to appropriate generator
        generator_type = self.router.route(spec)

        # 2. Instantiate selected generator
        generator_class = self.generators.get(generator_type)
        if not generator_class:
            raise ValueError(f"Unknown generator: {generator_type}")

        generator = generator_class(spec)

        # 3. Generate samples
        samples = generator.generate()

        return samples

    def _run_adaptive(
        self,
        spec: Spec,
        initial_state: LocalFeedbackState,
        max_iterations: int,
    ) -> tuple[list[Sample], LocalFeedbackState]:
        """Execute adaptive pipeline with iterative feedback loop."""
        # 1) Build context from spec
        context = self.context_extractor.extract(spec)

        # 2) Initialize local feedback state for this spec/job
        state = initial_state
        collected = []

        logger.info(f"Starting adaptive pipeline for {spec.num_samples} samples")

        iteration = 0
        while len(collected) < spec.num_samples and iteration < max_iterations:
            iteration += 1

            # 3) Get GenerationPlan from Router (Router gets context + state)
            plan = self.router.plan_next_batch(context, state)

            # Cap batch size based on remaining samples needed
            remaining = spec.num_samples - len(collected)
            actual_batch_size = min(plan.batch_size, remaining)
            plan.batch_size = actual_batch_size

            logger.info(
                f"Iteration {iteration}: Batch size: {plan.batch_size} | "
                f"Temperature: {plan.parameters.get('temperature', 'N/A')}"
            )

            # 4) Generate batch
            try:
                batch = self._generate_batch(spec, plan)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                continue

            # 5) Filter and score batch to get accepted samples
            accepted = self._filter_and_score(batch, spec)

            # 6) Compute batch metrics
            batch_metrics = self.feedback_engine.compute_batch_metrics(
                samples=accepted,
                total_generated=len(batch),
            )

            logger.info(
                f"Batch metrics: {batch_metrics.num_samples} accepted, "
                f"pass_rate={batch_metrics.pass_rate:.2f}, "
                f"mean_quality={batch_metrics.mean_quality or 'N/A'}"
            )

            # 7) Update local feedback state
            state = self.feedback_engine.update_feedback_state(
                state=state,
                plan=plan,
                batch_metrics=batch_metrics,
                samples=accepted,
            )

            # 8) Collect accepted samples
            collected.extend(accepted)

            # Safety check
            if iteration >= max_iterations:
                logger.warning(f"Reached max iterations ({max_iterations})")
                break

        # Log final statistics
        arm_stats = self.feedback_engine.get_arm_statistics(state)
        logger.info(f"Pipeline complete: {len(collected)} total samples collected")
        logger.info(f"Arm statistics: {arm_stats}")

        return collected, state

    def _generate_batch(self, spec: Spec, plan) -> list[Sample]:
        """Generate a batch of samples according to the GenerationPlan."""
        # Get generator class
        generator_type = plan.generator_arm
        if isinstance(generator_type, str):
            generator_type = GeneratorType(generator_type)

        generator_class = self.generators.get(generator_type)
        if not generator_class:
            raise ValueError(f"Unknown generator: {generator_type}")

        # Create temporary spec for this batch
        batch_spec = Spec(
            domain=spec.domain,
            task_input=spec.task_input,
            num_samples=plan.batch_size,
            constraints=plan.parameters,
            output_format=spec.output_format,
        )

        # Instantiate and run generator
        generator = generator_class(batch_spec)
        samples = generator.generate()

        return samples

    def _filter_and_score(self, samples: list[Sample], spec: Spec) -> list[Sample]:
        """Filter and score batch using quality orchestrator."""
        if not self.quality_orchestrator:
            # No filtering, accept all samples
            return samples

        # Run sample-level validation
        for sample in samples:
            validation_results = self.quality_orchestrator.validate_sample(sample)

            # Aggregate validation results into quality scores
            for validator_name, result in validation_results.items():
                if result.passed:
                    score = result.metadata.get("score", 1.0) if result.metadata else 1.0
                    sample.quality_scores[validator_name] = score
                else:
                    sample.quality_scores[validator_name] = 0.0

        # Run batch-level validation (e.g., diversity)
        batch_results = self.quality_orchestrator.validate_batch(samples)
        for validator_name, result in batch_results.items():
            if result.metadata:
                for sample in samples:
                    if "batch_metrics" not in sample.metadata:
                        sample.metadata["batch_metrics"] = {}
                    sample.metadata["batch_metrics"][validator_name] = result.metadata

        # Filter out samples that failed validation
        if hasattr(self.quality_orchestrator, 'filter_failing_samples'):
            accepted = self.quality_orchestrator.filter_failing_samples(samples)
        else:
            # If no filter method, accept all (fallback)
            accepted = samples

        return accepted
