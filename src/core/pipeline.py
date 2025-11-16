import logging

from src.core.feedback import FeedbackEngine
from src.core.generator_types import GeneratorType
from src.core.spec import GenerationPlan, LocalFeedbackState, Sample, Spec
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

        # Map generator types to classes
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
        # Store spec for batch generation
        self.spec = spec

        # Build static context from spec
        context = self.context_extractor.extract(spec)

        # Initialize local feedback state for this spec/job
        state = initial_state
        collected = []
        rejected = []

        logger.info(f"Starting adaptive pipeline for {spec.num_samples} samples")

        iteration = 0
        while len(collected) < spec.num_samples and iteration < max_iterations:
            iteration += 1

            # 3) Build dynamic progress info
            progress = {
                "remaining_samples": spec.num_samples - len(collected),
                "collected_samples": len(collected),
                "iteration": iteration,
            }

            # 4) Get GenerationPlan from Router
            plan = self.router.route(context, state, progress)

            logger.info(
                f"Iteration {iteration}: Batch size: {plan.batch_size} | "
                f"Temperature: {plan.parameters.get('temperature', 'N/A')}"
            )

            # 5) Generate batch
            try:
                batch = self._generate_batch(plan)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                continue

            # 6) Filter and score batch to get accepted samples
            accepted = self._filter_and_score(batch)

            # Identify rejected samples
            accepted_ids = {id(s) for s in accepted}
            batch_rejected = [s for s in batch if id(s) not in accepted_ids]
            rejected.extend(batch_rejected)

            # 7) Compute batch metrics
            batch_metrics = self.feedback_engine.compute_batch_metrics(
                samples=accepted,
                total_generated=len(batch),
            )

            logger.info(
                f"Batch metrics: {batch_metrics.num_samples} accepted, "
                f"pass_rate={batch_metrics.pass_rate:.2f}, "
                f"mean_quality={batch_metrics.mean_quality or 'N/A'}"
            )

            # 8) Update local feedback state
            state = self.feedback_engine.update_feedback_state(
                state=state,
                plan=plan,
                batch_metrics=batch_metrics,
                samples=accepted,
            )

            # 9) Adapt generation parameters
            state = self.router.adapt(state=state, metrics=batch_metrics)

            # 10) Collect accepted samples
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

    def _generate_batch(self, plan: GenerationPlan) -> list[Sample]:
        """Generate a batch of samples according to the GenerationPlan."""
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

        # Create temporary spec for this batch
        batch_spec = Spec(
            domain=self.spec.domain,
            task_input=self.spec.task_input,
            num_samples=plan.batch_size,
            constraints=plan.parameters,
            output_format=self.spec.output_format,
        )

        # Instantiate and run generator
        generator = generator_class(batch_spec)
        samples = generator.generate()

        return samples

    def _filter_and_score(self, samples: list[Sample]) -> list[Sample]:
        """Filter and score batch using quality orchestrator."""
        # Run all validators and populate quality_scores
        samples = self.quality_orchestrator.assess(samples, self.spec)

        # Filter out samples that failed validation
        accepted = self.quality_orchestrator.filter_failing_samples(samples)

        return accepted
