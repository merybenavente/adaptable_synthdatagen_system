# An Adaptable Data Generation System

## The Philosophy of Adaptation

Generally speaking, effective adaptation operates on three principles, each with its own complexity:

- **Read the room**: Assess context and gather all relevant signals to correctly understand the situation you're in. The challenge is gathering sufficient context to accurately grasp what's actually happening.
- **Dual-objective recognition**: Distinguish stated expectations from actual needs. The challenge is separating surface-level expectations from underlying needs that may not be explicitly stated.
- **Calibrated response**: Balance both based on role and influence capacity. The challenge is determining the right tradeoffs dynamically while recognizing when adaptation would compromise the goal.

This is an iterative process. As you act and receive feedback, you refine your understanding and adjust.

---

## The System

Building an adaptable data generation system requires operationalizing these principles: gathering sufficient context and constraints, translating high-level requirements into concrete generation parameters, and knowing when to make informed assumptions versus requesting clarification, all while maintaining data quality and integrity.

This pet project explores building such a system through a **multi-armed bandit architecture** that learns from experience.

### Core Pipeline

The generation pipeline flows through five interconnected modules:

1. **Requirements specification**
   Input specifications define the domain, constraints, format requirements, and desired sample count.

2. **Generators**
   Multiple generation strategies—from simple direct LLM calls (naive) to sophisticated approaches like RAG-augmented generation, evolution-based instruction enhancement (WizardLM), and grammar-based templating—each representing an "arm" the system can select.

3. **Router**
   A contextual bandit that intelligently selects which generator to use based on context features (domain type, complexity level, data format, knowledge base availability).

4. **Quality assessment**
   Multi-layered validation including rule-based checks (length, format, schema), model-based checks (PII detection, toxicity, entailment), and diversity scoring.

5. **Feedback loop**
   Quality outcomes are fed back as rewards to the router, which learns which strategies work best for different contexts and continuously adapts its routing decisions.

### Evaluation

Outside this generation loop, the system includes an **evaluation** module that tests the generated datasets on downstream tasks—training small models on the synthetic data to measure their real-world utility and feeding those results back as additional signals for the bandit.

---

**The result**: A system that doesn't just generate data, but adapts its approach based on what actually works, balancing exploration of new strategies with exploitation of proven ones.

---

## Quick Start

### Installation

```bash
# Install dependencies
pip install -e .

# Set up environment
echo "OPENAI_API_KEY=your_key_here" > .env
```

### Run the Adaptive Pipeline

```bash
# Basic usage
python scripts/generate_adaptive.py \
    --config config/recipes/adaptive_demo_example.yaml

# With custom settings
python scripts/generate_adaptive.py \
    --config config/recipes/task_rewrite_example.yaml \
    --batch-size 5 \
    --initial-temp 0.7 \
    --output data/generated/ \
    --save-state \
    --verbose
```

### What You'll See

The pipeline runs in iterative batches:
1. **Router** selects a generator arm and parameters based on feedback
2. **Generator** produces samples for the batch
3. **Quality assessment** scores each sample
4. **Feedback engine** computes metrics and updates state
5. Repeat until target samples reached

The system adapts:
- **Temperature** adjusts based on quality (lower if quality drops, higher if quality is good)
- **Exploration rate** decays over time (more exploitation of best arms)
- **Arm selection** balances trying new generators vs using proven ones

---

## Documentation

- **[USAGE.md](USAGE.md)**: Complete usage guide with examples and workflows
- **[ARCHITECTURE.md](ARCHITECTURE.md)**: Detailed technical documentation
- **[config/recipes/](config/recipes/)**: Example specifications

---

## System Status

### ✅ Fully Implemented
- **Adaptive Pipeline**: Iterative batch generation with feedback loop
- **Epsilon-Greedy Router**: Contextual bandit arm selection
- **Feedback Engine**: Metrics computation and state updates
- **NaiveGenerator**: Direct LLM generation with domain-specific prompts
- **Quality Assessment**: Semantic similarity, diversity, entailment validation
- **CLI**: Full-featured command-line interface
- **Data Models**: GenerationPlan, LocalFeedbackState, BatchMetrics

### 📋 Documented (Not Yet Implemented)
- **Thompson Sampling Router**: Bayesian approach to exploration/exploitation
- **UCB Router**: Upper confidence bound strategy
- **TemplaterGenerator**: Grammar-based structured data generation
- **RAG-LLMGenerator**: Retrieval-augmented generation with knowledge base
- **Additional Validators**: PII detection, toxicity, format, perplexity
- **Evaluation Harness**: Downstream task evaluation for measuring utility
- **Knowledge Base System**: Indexing and retrieval for RAG

---

## Project Structure

```
adaptable_synthdatagen_system/
├── config/
│   └── recipes/               # Example specifications
├── scripts/
│   ├── generate_adaptive.py   # ✅ Main CLI for adaptive pipeline
│   └── generate.py            # ✅ Legacy simple pipeline
├── src/
│   ├── core/
│   │   ├── adaptive_pipeline.py  # ✅ Iterative feedback loop
│   │   ├── feedback.py           # ✅ Metrics and state updates
│   │   ├── spec.py               # ✅ Data models
│   │   └── pipeline.py           # ✅ Legacy pipeline
│   ├── router/
│   │   └── adaptive_router.py    # ✅ Epsilon-greedy bandit
│   ├── generators/
│   │   └── naive_generator.py    # ✅ Direct LLM generation
│   ├── quality/
│   │   ├── orchestrator.py       # ✅ Validation coordinator
│   │   ├── semantic_validator.py # ✅ Similarity + entailment
│   │   └── diversity_validator.py # ✅ Batch diversity
│   └── utils/                     # ✅ LLM, embedding clients
├── ARCHITECTURE.md            # ✅ Technical documentation
├── USAGE.md                   # ✅ Usage guide
└── README.md                  # This file
```

---

## Example Output

```
======================================================================
ADAPTIVE SYNTHETIC DATA GENERATION PIPELINE
======================================================================
Domain:          task_rewrite
Total Samples:   20
Batch Size:      5
Strategy:        epsilon_greedy
======================================================================

Iteration 1: Exploring: randomly selected naive (ε=0.100)
             Batch size: 5 | Temperature: 0.7
Batch metrics: 5 samples, pass_rate=0.80, mean_quality=0.65

Iteration 2: Exploiting: selected naive with mean reward 0.650
             Batch size: 5 | Temperature: 0.65
Batch metrics: 5 samples, pass_rate=1.00, mean_quality=0.82

...

GENERATION COMPLETE
Total Samples Generated:  20
Final Temperature:        0.70
Exploration Rate:         0.08

ARM PERFORMANCE STATISTICS:
  naive:
    Mean Reward:  0.735
    Count:        4
```

---

## Contributing

This is a demonstration project showcasing ML engineering skills. Key areas for extension:

1. **New generators**: Implement TemplaterGenerator or RAG-LLMGenerator
2. **New strategies**: Implement Thompson Sampling or UCB
3. **New validators**: Add PII detection, toxicity filtering
4. **Evaluation**: Build downstream task evaluation harness
5. **Optimization**: Async generation, caching, batched APIs

---

## License

MIT