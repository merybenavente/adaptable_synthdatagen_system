# Architecture Documentation

## System Overview

This system implements an **adaptive synthetic data generation pipeline** that produces text data through iterative batches with feedback-driven optimization. The pipeline adapts its generation strategy after each batch based on quality metrics, implementing a contextual multi-armed bandit approach.

## Core Concept

Each batch is a small experiment:
1. **Router** picks a configuration (generator arm + parameters)
2. **Generator** produces samples
3. **Quality assessment** scores the output
4. **Feedback Engine** updates state with metrics
5. **Router** adapts next configuration based on results

This creates a self-improving loop that balances exploration (trying different strategies) with exploitation (using what works).

---

## System Architecture

### 1. Data Models (`src/core/spec.py`)

#### Spec
Input specification for a generation job:
- `domain`: Type of data (TASK_REWRITE, QA_PAIRS, CODE_SNIPPETS)
- `task_input`: The input content or prompt
- `num_samples`: Total number of samples to generate
- `constraints`: Domain-specific requirements
- `output_format`: Desired output format

#### GenerationPlan
Output of Router describing batch configuration:
- `batch_size`: Number of samples in this batch
- `generator_arm`: Which generator to use (NAIVE, TEMPLATER, RAG_LLM)
- `parameters`: Generation parameters (temperature, etc.)
- `iteration`: Batch iteration number
- `reasoning`: Why this plan was chosen

#### LocalFeedbackState
State container for the feedback loop:
- `generated_so_far`: Total samples generated
- `iteration`: Current iteration number
- `current_temperature`: Adaptive temperature parameter
- `arm_counts`: How many times each generator was used
- `arm_rewards`: Reward history for each generator
- `recent_metrics`: Recent batch metrics for trend analysis
- `exploration_rate`: Current exploration vs exploitation balance

#### BatchMetrics
Computed metrics for a batch:
- `mean_similarity`: Average semantic similarity score
- `diversity_score`: Batch diversity metric
- `mean_quality`: Overall quality score
- `pass_rate`: Fraction of samples passing validation
- `num_samples`: Samples in batch

#### Sample
Individual generated sample:
- `id`: Unique identifier
- `content`: Generated content
- `metadata`: Timestamps and operational data
- `lineage`: Provenance tracking (generator, parameters, parent)
- `quality_scores`: Validation scores

---

### 2. Pipeline (`src/core/adaptive_pipeline.py`)

The `AdaptivePipeline` orchestrates the iterative generation loop:

```python
while generated_so_far < num_samples:
    1. plan = router.plan_next_batch(spec, state)
    2. batch = generator.generate(plan)
    3. scored_batch = quality.assess(batch)
    4. metrics = feedback.compute_metrics(scored_batch)
    5. state = feedback.update_state(state, metrics)
```

**Key characteristics:**
- **Stateless iterations**: Each batch is independent
- **No intelligence**: Pipeline just coordinates components
- **Graceful degradation**: Continues even if quality assessment fails
- **Safety limits**: Maximum iteration count prevents infinite loops

---

### 3. Router (`src/router/adaptive_router.py`)

The `AdaptiveRouter` selects generator arms using bandit strategies:

#### Epsilon-Greedy Strategy (Currently Implemented)
- **Exploration** (probability ε): Select random arm
- **Exploitation** (probability 1-ε): Select best-performing arm based on mean reward

The router:
1. Extracts context from Spec
2. Reads LocalFeedbackState for arm performance
3. Selects arm based on strategy
4. Determines batch size (min, max, remaining)
5. Builds parameters dict (temperature, constraints)
6. Returns GenerationPlan with reasoning

**Future strategies** (documented, not yet implemented):
- Thompson Sampling: Beta-Bernoulli sampling
- UCB (Upper Confidence Bound): Balance mean + uncertainty

---

### 4. Feedback Engine (`src/core/feedback.py`)

The `FeedbackEngine` computes metrics and updates state:

#### compute_batch_metrics()
Extracts quality scores from samples and computes:
- Mean similarity (semantic similarity to original)
- Diversity score (batch-level diversity)
- Mean quality (average of all quality scores)
- Pass rate (fraction passing validation)

#### update_feedback_state()
Updates LocalFeedbackState with:
1. **Arm statistics**: Increment counts, append rewards
2. **Temperature adaptation**:
   - Low quality or pass rate → decrease temperature (more conservative)
   - High quality and pass rate → increase temperature (more diversity)
   - Clamped to [0.3, 1.2]
3. **Exploration decay**: Reduce exploration rate by 5% each iteration
4. **Metrics history**: Keep last N batch metrics

**Adaptive temperature logic:**
```python
if pass_rate < 0.5 or quality < 0.6:
    temperature -= 0.05  # Be more conservative
elif quality > 0.8 and pass_rate > 0.8:
    temperature += 0.025  # Try more diversity
```

---

### 5. Generators (`src/generators/`)

#### NaiveGenerator (Implemented)
Direct LLM generation with domain-specific prompts:
- Task rewrite: Generates paraphrases maintaining semantic meaning
- QA pairs: Generates question-answer pairs from topics
- Two-step prompting: Constraints → natural language → generation

**Lineage tracking**: Records generator type, model, temperature, parent ID

#### TemplaterGenerator (Documented, Not Implemented)
Grammar-based generation using PEG/PCFG:
- Structured data with syntactic constraints
- Deterministic variation

#### RAG-LLMGenerator (Documented, Not Implemented)
Retrieval-augmented generation:
- Retrieves relevant context from knowledge base
- Augments prompts with retrieved information
- Constrained decoding for structured output

---

### 6. Quality Assessment (`src/quality/`)

#### QualityOrchestrator
Coordinates multiple validators:

**Sample-level validators:**
- `SemanticValidator`: Cosine similarity + entailment checking
- `RuleValidator`: Length, format, schema (documented, not implemented)
- `ModelValidator`: PII, toxicity, perplexity (documented, not implemented)

**Batch-level validators:**
- `DiversityValidator`: Pairwise similarity, cluster analysis

**Filtering:**
- `filter_failing_samples()`: Removes samples below threshold
- Configurable thresholds per validator

---

## Configuration

### Spec Configuration (YAML)
```yaml
domain: task_rewrite
task_input: "Explain recursion to a beginner."
num_samples: 20
constraints:
  maintain_semantic_meaning: true
  min_length: 15
  max_length: 150
output_format: text
```

### Pipeline Configuration
Set via CLI arguments:
- `--batch-size`: Samples per batch (default: 5)
- `--strategy`: epsilon_greedy | thompson_sampling | ucb
- `--initial-temp`: Starting temperature (default: 0.7)
- `--initial-exploration`: Starting exploration rate (default: 0.1)

---

## Data Flow

### End-to-End Example

**Input:**
```yaml
domain: task_rewrite
task_input: "Summarize this article."
num_samples: 10
```

**Iteration 1:**
- Router: No history → random arm (NAIVE), temp=0.7, batch_size=5
- Generator: Produces 5 samples
- Quality: Scores samples (similarity, diversity)
- Metrics: mean_quality=0.65, pass_rate=0.8
- Feedback: NAIVE reward=0.65, temp→0.65 (quality low)

**Iteration 2:**
- Router: Exploit NAIVE (best arm), temp=0.65, batch_size=5
- Generator: Produces 5 samples (more conservative)
- Quality: Scores samples
- Metrics: mean_quality=0.82, pass_rate=1.0
- Feedback: NAIVE reward=0.82, temp→0.70 (quality high)

**Output:**
- 10 samples total
- Final state: 2 iterations, exploration_rate=0.09, temperature=0.70
- Arm stats: NAIVE (mean_reward=0.735, count=2)

---

## Extensibility

### Adding a New Generator

1. Create class in `src/generators/`:
```python
class MyGenerator(BaseGenerator):
    def generate(self) -> list[Sample]:
        # Implementation
        pass
```

2. Add to GeneratorType enum:
```python
class GeneratorType(str, Enum):
    MY_GENERATOR = "my_generator"
```

3. Register in pipeline:
```python
available_generators = {
    GeneratorType.MY_GENERATOR: MyGenerator,
}
```

### Adding a New Validator

1. Implement validator in `src/quality/`:
```python
class MyValidator(BaseValidator):
    def validate(self, sample: Sample) -> ValidationResult:
        # Implementation
        pass
```

2. Add to orchestrator configuration

### Adding a New Routing Strategy

1. Implement method in `AdaptiveRouter`:
```python
def _my_strategy(self, state: LocalFeedbackState) -> tuple[GeneratorType, str]:
    # Implementation
    return selected_arm, reasoning
```

2. Add to strategy selection in `plan_next_batch()`

---

## Key Design Decisions

### Why LocalFeedbackState?
- **Isolation**: Each generation job has independent state
- **Reproducibility**: State can be saved/loaded for debugging
- **Flexibility**: Different jobs can use different strategies simultaneously

### Why Batch-Based Generation?
- **Efficiency**: Amortize API calls and validation overhead
- **Stability**: Metrics more reliable with multiple samples
- **Adaptability**: Fast feedback loop (adjust every N samples vs all at once)

### Why Quality as Reward?
- **Direct signal**: Quality scores directly measure output usefulness
- **Multi-faceted**: Can combine similarity, diversity, downstream metrics
- **Interpretable**: Easy to debug why router made decisions

### Why Epsilon-Greedy First?
- **Simplicity**: Easy to implement and understand
- **Effectiveness**: Works well in practice for many problems
- **Baseline**: Provides comparison point for advanced strategies

---

## Performance Considerations

### API Costs
- Batch size controls API call frequency
- Smaller batches = more iterations = more API calls
- Larger batches = fewer iterations = less adaptation

### Quality Assessment Overhead
- Semantic similarity requires embeddings (1 API call per sample)
- Entailment checking requires DeBERTa inference (local)
- Diversity validation is O(n²) in batch size

### Memory Usage
- LocalFeedbackState stores all rewards (grows with iterations)
- Consider truncating arm_rewards if running very long jobs
- Batch metrics limited to max_history_length (default: 10)

---

## Future Enhancements

### Planned (Documented, Not Implemented)
1. **Thompson Sampling**: Bayesian approach to exploration/exploitation
2. **UCB**: Upper confidence bound for optimistic exploration
3. **Additional Generators**: Templater, RAG-LLM, WizardLM evolution
4. **Additional Validators**: PII, toxicity, perplexity, format
5. **Evaluation Harness**: Train models on synthetic data, measure utility
6. **Knowledge Base**: Retrieval for RAG generation

### Research Directions
1. **Contextual bandits**: Use context features to personalize arm selection
2. **Meta-learning**: Learn initialization strategies across jobs
3. **Multi-objective optimization**: Balance quality, diversity, cost
4. **Active learning**: Request human feedback on uncertain samples

---

## Troubleshooting

### Router always selects the same arm
- Check exploration_rate (may have decayed too low)
- Verify arm_rewards are being updated
- Enable verbose logging to see reasoning

### Quality scores all zero
- Check API keys (OPENAI_API_KEY for embeddings)
- Verify validators are configured correctly
- Check sample content format matches validator expectations

### Temperature not adapting
- Ensure temperature_adaptation=True in FeedbackEngine
- Check that quality scores are being computed
- Verify samples have quality_scores populated

### Pipeline runs too many iterations
- Check that generated_so_far is incrementing
- Verify batch_size > 0
- Ensure max_iterations is set appropriately

---

## Testing

### Unit Tests
- Test each component in isolation
- Mock dependencies (LLM API, embeddings)
- Verify state transitions in FeedbackEngine

### Integration Tests
- Run full pipeline with small num_samples
- Verify all components interact correctly
- Check output format and metadata

### End-to-End Tests
- Use example configs with actual APIs
- Verify quality of generated data
- Measure adaptation behavior

---

## References

### Multi-Armed Bandits
- Sutton & Barto, "Reinforcement Learning: An Introduction" (Chapter 2)
- Lattimore & Szepesvári, "Bandit Algorithms"

### Synthetic Data Generation
- WizardLM: Empowering Large Language Models to Follow Complex Instructions
- Self-Instruct: Aligning Language Model with Self Generated Instructions

### Quality Assessment
- Semantic similarity: Sentence-BERT, OpenAI embeddings
- Entailment: DeBERTa NLI models
- Diversity: Determinantal Point Processes, cluster-based metrics
