# Core Module Overview

## `Models`, frame our solution
The core models walk us through the system's logic:

`Spec` (input config) → `GenerationContext` (extracted features) → Router reads `LocalFeedbackState` (arm rewards) → `GenerationPlan` (arm selection) → `Samples` (generated batch) → `BatchMetrics` (quality scores) → `LocalFeedbackState` (updated rewards) → loop

On each iteration the Router reads context and feedback state, selects an arm via epsilon-greedy, generator produces samples, quality orchestrator validates and scores them, feedback updates rewards for next iteration.

## `Spec / Context`, from human config to routing features
- Loads YAML recipes and extracts routing features (grammar, constraints).
- Auto-detects the grammar sections and triggers schema auto-derivation for TEMPLATER.
- Builds `GenerationContext` to inform dynamic arm selection.


## `Router`, selects arms via epsilon-greedy bandit
- Builds available arms based on context: NAIVE arms (no grammar) or TEMPLATER arms (grammar present).
- Uses epsilon-greedy policy to balance exploration vs exploitation.
- Exploration rate decays over iterations via `AdaptationPolicy`.

## `Generators`, two different strategies with arms each
- Both extend the minimal interface from `base_generator.py`.
- `NAIVE`: Auto-plans prompts via LLM, then generates samples via LLM.
- `TEMPLATER`: Uses PCFG grammar templates with LLM slot filling, auto-derives schemas from grammar.

## `Pipeline`, the (dumb) orchestrator
For each iteration:
  1. Asks the `Router` for a `GenerationPlan`.
  2. Instantiates the requested generator and creates a batch of `Sample`s.
  3. Hands the batch to the quality orchestrator for validation and scoring.
  4. Computes batch metrics and updates the feedback state.
  5. Lets the router adapt exploration parameters for the next round.

## `Quality`, validates and scores samples
- Configured per-recipe via `validators:` block in YAML configs.
- Sample-level validators: deduplication, schema validation, similarity, entailment.
- Batch-level validators: diversity, LLM judge.

## `Feedback`, computes (modestly smart) rewards and updates state
- Computes reward as `pass_rate * quality_score`. Example: 8/10 pass (`0.8`) × judge score `0.8` = reward `0.64`.
- Rewards go into `LocalFeedbackState.arm_rewards`, which the router reads to decide which arm to exploit next.

## `Storage`, saves data locally
- Outputs generated samples to local files (JSONL, CSV, text).
- Preserves lineage, metadata, and quality scores for each sample.


## Future Steps

| Component          | Current Scope                                           | Future Steps                                                            |
|--------------------|---------------------------------------------------------|-------------------------------------------------------------------------|
| Spec / Context     | Parse config, extract features, build context           | Chat + data driven experience; reason and ask about missing information |
| Router             | Simple bandit routing + exploration decay               | Contextual bandits or RL model, include costs in routing decision       |
| Generators         | `NAIVE` (auto-prompt) + `TEMPLATER` (grammar)           | More types (RAG-LLM, auto-evolvers), hybrid plans                       |
| Pipeline           | Iterates batches, updates state                         | Async execution, multi-job orchestration, checkpointing/resume          |
| Quality            | Sample-level filters and batch scoring                  | Smart constraint interpretation and validator selection                 |
| Feedback State     | Local rewards from pass rate × judge score              | Persistent memory across jobs, per-domain memory, confidence modeling   |
| Storage            | Save data file locally                                  | Vector DB integration, versioning, dataset lineage tracking             |
| Evaluation         | Toy implementation with Tinker                          | Downstream task training, feedback loop from benchmarks to router       |
