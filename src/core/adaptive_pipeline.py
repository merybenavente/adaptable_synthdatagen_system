"""Adaptive Pipeline for iterative synthetic data generation with feedback."""

import logging

from src.core.feedback import FeedbackEngine
from src.core.generator_types import GeneratorType
from src.core.spec import GenerationPlan, LocalFeedbackState, Sample, Spec
from src.generators.naive_generator import NaiveGenerator
from src.quality.orchestrator import QualityOrchestrator
from src.router.router import Router

logger = logging.getLogger(__name__)


class AdaptivePipeline:
    """Adaptive pipeline for iterative batch generation with feedback loop."""

    def __init__(
        self,
        router: Router | None = None,
        feedback_engine: FeedbackEngine | None = None,
        quality_orchestrator: QualityOrchestrator | None = None,
        available_generators: dict[GeneratorType, type] | None = None,
    ):
        """Initialize AdaptivePipeline with router, feedback engine, and orchestrator."""
        self.router = router or Router()
        self.feedback_engine = feedback_engine or FeedbackEngine()
        self.quality_orchestrator = quality_orchestrator

        # Map generator types to classes
        self.generators = available_generators or {
            GeneratorType.NAIVE: NaiveGenerator,
            # GeneratorType.TEMPLATER: TemplaterGenerator,  # TODO: Implement
            # GeneratorType.RAG_LLM: RAGLLMGenerator,  # TODO: Implement
        }

    def run(
        self,
        spec: Spec,
        initial_state: LocalFeedbackState | None = None,
        max_iterations: int = 100,
    ) -> tuple[list[Sample], LocalFeedbackState]:
        """Execute adaptive generation pipeline with feedback loop."""
        # Initialize feedback state
        state = initial_state or LocalFeedbackState()
        all_samples: list[Sample] = []

        logger.info(f"Starting adaptive pipeline for {spec.num_samples} samples")

        iteration = 0
        while state.generated_so_far < spec.num_samples and iteration < max_iterations:
            iteration += 1

            # Step 1: Get GenerationPlan from Router
            plan = self.router.plan_next_batch(spec, state)

            logger.info(
                f"Iteration {iteration}: {plan.reasoning} | "
                f"Batch size: {plan.batch_size} | "
                f"Temperature: {plan.parameters.get('temperature', 'N/A')}"
            )

            # Step 2: Generate batch
            try:
                batch_samples = self._generate_batch(spec, plan)
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                # Continue with next iteration
                continue

            # Step 3: Run quality scoring (if orchestrator provided)
            if self.quality_orchestrator:
                try:
                    batch_samples = self._score_batch(batch_samples, spec)
                except Exception as e:
                    logger.warning(f"Quality scoring failed: {e}")
                    # Continue without scores

            # Step 4: Compute batch metrics
            batch_metrics = self.feedback_engine.compute_batch_metrics(
                samples=batch_samples,
                total_generated=plan.batch_size,
            )

            logger.info(
                f"Batch metrics: {batch_metrics.num_samples} samples, "
                f"pass_rate={batch_metrics.pass_rate:.2f}, "
                f"mean_quality={batch_metrics.mean_quality or 'N/A'}"
            )

            # Step 5: Update feedback state
            state = self.feedback_engine.update_feedback_state(
                state=state,
                plan=plan,
                batch_metrics=batch_metrics,
                samples=batch_samples,
            )

            # Add samples to collection
            all_samples.extend(batch_samples)

            # Safety check
            if iteration >= max_iterations:
                logger.warning(f"Reached max iterations ({max_iterations})")
                break

        # Log final statistics
        arm_stats = self.feedback_engine.get_arm_statistics(state)
        logger.info(f"Pipeline complete: {len(all_samples)} total samples generated")
        logger.info(f"Arm statistics: {arm_stats}")

        return all_samples, state

    def _generate_batch(self, spec: Spec, plan: GenerationPlan) -> list[Sample]:
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
            constraints=plan.parameters,  # Pass parameters as constraints
            output_format=spec.output_format,
        )

        # Instantiate and run generator
        generator = generator_class(batch_spec)
        samples = generator.generate()

        return samples

    def _score_batch(self, samples: list[Sample], spec: Spec) -> list[Sample]:
        """Score a batch of samples using QualityOrchestrator."""
        if not self.quality_orchestrator:
            return samples

        # Run sample-level validation
        for sample in samples:
            validation_results = self.quality_orchestrator.validate_sample(sample)

            # Aggregate validation results into quality scores
            for validator_name, result in validation_results.items():
                if result.passed:
                    # Use the score if available, otherwise binary 1.0
                    score = result.metadata.get("score", 1.0) if result.metadata else 1.0
                    sample.quality_scores[validator_name] = score
                else:
                    # Failed validation gets 0.0
                    sample.quality_scores[validator_name] = 0.0

        # Run batch-level validation (e.g., diversity)
        batch_results = self.quality_orchestrator.validate_batch(samples)
        for validator_name, result in batch_results.items():
            # Store batch-level scores in metadata of each sample
            if result.metadata:
                for sample in samples:
                    # Add batch metrics to sample metadata
                    if "batch_metrics" not in sample.metadata:
                        sample.metadata["batch_metrics"] = {}
                    sample.metadata["batch_metrics"][validator_name] = result.metadata

        return samples


# Legacy Pipeline for backwards compatibility
class Pipeline:
    """Simple pipeline without feedback (legacy)."""

    def __init__(self, routing_config_path: str | None = None):
        from src.router import Router
        self.router = Router()
        self.generators = {
            GeneratorType.NAIVE: NaiveGenerator,
        }

    def run(self, spec: Spec) -> list[Sample]:
        """Execute simple generation pipeline: route → generate → return."""
        generator_type = self.router.route(spec)
        generator_class = self.generators.get(generator_type)
        if not generator_class:
            raise ValueError(f"Unknown generator: {generator_type}")
        generator = generator_class(spec)
        samples = generator.generate()
        return samples
