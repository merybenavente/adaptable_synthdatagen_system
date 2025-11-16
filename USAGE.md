# Usage Guide

## Quick Start

### Prerequisites

1. **Install dependencies:**
```bash
pip install -e .
```

2. **Set up environment variables:**
```bash
# Create .env file
echo "OPENAI_API_KEY=your_key_here" > .env
```

3. **Verify installation:**
```bash
python scripts/generate_adaptive.py --help
```

---

## Running the Adaptive Pipeline

### Basic Usage

```bash
python scripts/generate_adaptive.py \
    --config config/recipes/task_rewrite_example.yaml
```

This will:
- Generate 5 samples (as specified in the config)
- Use epsilon-greedy strategy
- Adapt temperature based on quality
- Display results to console

### With Output Saving

```bash
python scripts/generate_adaptive.py \
    --config config/recipes/adaptive_demo_example.yaml \
    --output data/generated/ \
    --save-state
```

This saves:
- `data/generated/samples.jsonl`: All generated samples
- `data/generated/feedback_state.json`: Final LocalFeedbackState

### Custom Configuration

```bash
python scripts/generate_adaptive.py \
    --config config/recipes/qa_pairs_example.yaml \
    --batch-size 3 \
    --initial-temp 0.8 \
    --initial-exploration 0.2 \
    --strategy epsilon_greedy \
    --verbose
```

**Parameters:**
- `--batch-size`: Samples per iteration (default: 5)
- `--initial-temp`: Starting temperature (default: 0.7)
- `--initial-exploration`: Starting exploration rate (default: 0.1)
- `--strategy`: epsilon_greedy | thompson_sampling | ucb
- `--verbose`: Enable detailed logging
- `--no-filter`: Disable quality filtering (keep all samples)
- `--save-state`: Save LocalFeedbackState to output directory

---

## Creating Custom Specifications

### Specification Format (YAML)

```yaml
# Domain type (required)
domain: task_rewrite  # or qa_pairs, code_snippets

# Input content (required)
task_input: "Your input text or prompt here"

# Total samples to generate (required)
num_samples: 20

# Domain-specific constraints (optional)
constraints:
  maintain_semantic_meaning: true
  vary_formality: true
  min_length: 15
  max_length: 150

# Output format (optional, default: text)
output_format: text  # or json
```

### Domain-Specific Examples

#### Task Rewrite
Generate paraphrases of instructions/prompts:
```yaml
domain: task_rewrite
task_input: "Explain machine learning to a 5-year-old."
num_samples: 10
constraints:
  maintain_semantic_meaning: true
  vary_formality: true
  min_length: 20
```

#### Q&A Pairs
Generate question-answer pairs:
```yaml
domain: qa_pairs
task_input:
  topic: "Neural networks"
  context: "Neural networks are computing systems inspired by biological neural networks."
  difficulty: "beginner"
num_samples: 15
constraints:
  question_types: ["what", "how", "why"]
  answer_length_range: [50, 200]
  include_code_examples: false
output_format: json
```

---

## Understanding Output

### Console Output

```
======================================================================
ADAPTIVE SYNTHETIC DATA GENERATION PIPELINE
======================================================================
Domain:          task_rewrite
Task Input:      Explain recursion to a beginner.
Total Samples:   20
Batch Size:      5
Strategy:        epsilon_greedy
Initial Temp:    0.7
Exploration:     0.1
======================================================================

Iteration 1: Exploring: randomly selected naive (ε=0.100) |
             Batch size: 5 | Temperature: 0.7
Batch metrics: 5 samples, pass_rate=0.80, mean_quality=0.65

Iteration 2: Exploiting: selected naive with mean reward 0.650 (count=1) |
             Batch size: 5 | Temperature: 0.65
Batch metrics: 5 samples, pass_rate=1.00, mean_quality=0.82

...

======================================================================
GENERATION COMPLETE
======================================================================
Total Samples Generated:  20
Total Iterations:         4
Final Temperature:        0.70
Final Exploration Rate:   0.08

ARM PERFORMANCE STATISTICS:
----------------------------------------------------------------------
  naive:
    Mean Reward:  0.735
    Std Reward:   0.082
    Count:        4
```

### Output Files

#### samples.jsonl
```jsonl
{
  "id": "uuid-here",
  "content": "Generated sample text...",
  "lineage": {
    "num_of_evolutions": 0,
    "generator": "naive",
    "generator_parameters": {
      "model": "gpt-4",
      "temperature": 0.7
    }
  },
  "metadata": {
    "timestamp": "2025-01-15T10:30:00",
    "batch_metrics": {...}
  },
  "quality_scores": {
    "similarity": 0.85,
    "diversity": 0.72
  }
}
```

#### feedback_state.json
```json
{
  "generated_so_far": 20,
  "iteration": 4,
  "current_temperature": 0.70,
  "arm_counts": {
    "naive": 4
  },
  "arm_rewards": {
    "naive": [0.65, 0.82, 0.75, 0.70]
  },
  "recent_metrics": [
    {
      "mean_similarity": 0.83,
      "diversity_score": 0.75,
      "mean_quality": 0.70,
      "pass_rate": 1.0,
      "num_samples": 5
    }
  ],
  "exploration_rate": 0.08
}
```

---

## Advanced Usage

### Running Without Quality Filtering

Useful for debugging or when you want to see all generated samples:

```bash
python scripts/generate_adaptive.py \
    --config config/recipes/task_rewrite_example.yaml \
    --no-filter
```

### Resuming from Previous State

Currently not supported in CLI, but can be done programmatically:

```python
from src.core.adaptive_pipeline import AdaptivePipeline
from src.core.spec import LocalFeedbackState
import json

# Load previous state
with open("data/generated/feedback_state.json") as f:
    state_dict = json.load(f)
    state = LocalFeedbackState(**state_dict)

# Resume pipeline
pipeline = AdaptivePipeline()
samples, final_state = pipeline.run(spec, initial_state=state)
```

### Custom Generators

To use a custom generator:

```python
from src.core.adaptive_pipeline import AdaptivePipeline
from src.core.generator_types import GeneratorType
from my_generators import MyCustomGenerator

# Register custom generator
available_generators = {
    GeneratorType.NAIVE: NaiveGenerator,
    GeneratorType.CUSTOM: MyCustomGenerator,
}

# Create pipeline with custom generators
pipeline = AdaptivePipeline(available_generators=available_generators)
```

---

## Monitoring and Debugging

### Enable Verbose Logging

```bash
python scripts/generate_adaptive.py \
    --config config/recipes/task_rewrite_example.yaml \
    --verbose
```

This shows:
- Detailed router reasoning
- Quality assessment results
- Feedback state updates
- Error stack traces

### Interpreting Metrics

**Pass Rate:**
- 1.0 = All samples passed quality checks
- 0.5 = Half the samples were filtered out
- Low pass rate → Pipeline decreases temperature

**Mean Quality:**
- Range: 0.0 to 1.0
- Aggregate of all quality scores
- Low quality → Pipeline decreases temperature
- High quality → Pipeline may increase temperature for diversity

**Diversity Score:**
- Measures uniqueness within batch
- Low diversity → Samples are too similar
- High diversity → Good variation

**Exploration Rate:**
- Starts at initial_exploration (default: 0.1)
- Decays by 5% each iteration
- Controls random vs best-arm selection

---

## Troubleshooting

### Error: "OPENAI_API_KEY not found"

```bash
# Set in .env file
echo "OPENAI_API_KEY=sk-..." > .env

# Or export directly
export OPENAI_API_KEY=sk-...
```

### Error: "No samples generated"

Possible causes:
1. All samples filtered out (try --no-filter)
2. API rate limits (reduce batch_size)
3. Invalid spec (check YAML syntax)

### Warning: "Reached max iterations"

The pipeline hit the safety limit (default: 100 iterations).

Possible causes:
1. Batch size too small for num_samples
2. Pass rate too low (most samples filtered)
3. Bug in state updates

Solutions:
- Increase batch_size
- Disable filtering (--no-filter)
- Check logs for errors

### Low Quality Scores

If quality scores are consistently low:
1. Check task_input clarity
2. Adjust constraints (may be too strict)
3. Try higher temperature for more creativity
4. Review validator thresholds

---

## Best Practices

### Batch Size Selection
- **Small batches (3-5)**: Fast adaptation, more API calls
- **Large batches (10-20)**: Slower adaptation, fewer API calls
- **Rule of thumb**: num_samples / 5 iterations

### Temperature Settings
- **Low (0.3-0.5)**: Conservative, consistent output
- **Medium (0.6-0.8)**: Balanced creativity and consistency
- **High (0.9-1.2)**: Creative, diverse, potentially inconsistent

### Exploration Settings
- **Low (0.05)**: Mostly exploit best arm
- **Medium (0.1-0.2)**: Balanced exploration
- **High (0.3+)**: Frequently try new arms

### Domain-Specific Tips

**Task Rewrite:**
- Use semantic similarity validation
- Set min/max length constraints
- Enable formality variation

**Q&A Pairs:**
- Use JSON output format
- Specify question types
- Set answer length ranges
- Enable code examples if relevant

**Code Snippets:**
- Use syntax validators (when implemented)
- Set complexity constraints
- Enable format validation

---

## Performance Optimization

### Reducing Costs
1. Use smaller batch_size (fewer samples per API call)
2. Disable quality assessment for prototyping (--no-filter)
3. Use lower num_samples during development
4. Cache embeddings for similarity validation

### Reducing Latency
1. Increase batch_size (fewer iterations)
2. Disable slow validators
3. Use async generation (future enhancement)

### Improving Quality
1. Use higher initial_temp for creativity
2. Enable all validators
3. Increase num_samples for more selection
4. Tune validator thresholds

---

## Example Workflows

### Workflow 1: Rapid Prototyping

```bash
# Quick test with minimal samples
python scripts/generate_adaptive.py \
    --config config/recipes/task_rewrite_example.yaml \
    --no-filter \
    --batch-size 10
```

### Workflow 2: High-Quality Generation

```bash
# Generate many samples with quality filtering
python scripts/generate_adaptive.py \
    --config config/recipes/adaptive_demo_example.yaml \
    --batch-size 5 \
    --initial-temp 0.7 \
    --output data/generated/ \
    --save-state \
    --verbose
```

### Workflow 3: Exploration Focus

```bash
# Try different arms more frequently
python scripts/generate_adaptive.py \
    --config config/recipes/qa_pairs_example.yaml \
    --initial-exploration 0.3 \
    --batch-size 3
```

---

## Next Steps

1. **Experiment with different domains**: Try task_rewrite, qa_pairs
2. **Adjust hyperparameters**: Explore temperature, batch size, exploration
3. **Create custom specs**: Define your own generation tasks
4. **Analyze feedback states**: Study how the pipeline adapts
5. **Extend the system**: Add custom generators, validators, or strategies

For more details, see:
- `ARCHITECTURE.md`: System design and components
- `README.md`: Project overview and philosophy
- `config/recipes/`: Example specifications
