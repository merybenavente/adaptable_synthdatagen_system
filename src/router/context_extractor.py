from src.core.spec import GenerationContext, ProgressState, Spec


# TODO: Enhanced context intelligence - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/14
# TODO: Two-level context extraction for batch generation - https://github.com/merybenavente/adaptable_synthdatagen_system/issues/13
class ContextExtractor:
    """Extract intelligent routing context from generation requests."""

    def extract(self, spec: Spec) -> GenerationContext:
        """Extract intelligent context from Spec with feature inference."""
        # Analyze constraints for intelligent features
        constraints = spec.constraints or {}
        complexity = self._infer_complexity(spec)

        return GenerationContext(
            domain=spec.domain,
            task_input=spec.task_input,
            num_samples=spec.num_samples,
            constraints=constraints,
            complexity_level=complexity,
            constraint_count=len(constraints),
            has_examples="examples" in constraints,
            has_knowledge_base="knowledge_base" in constraints,
            progress=ProgressState(
                remaining_samples=spec.num_samples,
                collected_samples=0,
                rejected_samples=0,
                iteration=0,
            ),
        )

    @staticmethod
    def _infer_complexity(spec: Spec) -> str:
        """Infer complexity level from spec characteristics."""
        constraints = spec.constraints or {}

        # Simple heuristics for now
        if len(constraints) == 0:
            return "simple"
        elif len(constraints) > 3 or "knowledge_base" in constraints:
            return "complex"
        else:
            return "medium"
