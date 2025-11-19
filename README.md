# An Adaptable Data Generation System

## The Philosophy behind

Generally speaking, effective adaptation operates on three principles, each with its own complexity:

- **Read the room**: Assess context, gather relevant signals, and distinguish what actually matters from what's explicitly stated. The challenge is accurately grasping both what's being openly stated and which underlying needs will drive success.

- **Take action**: The chosen action should optimize the intersection of what you can do and what's actually needed. The challenge is leveraging your strengths effectively given the situation's constraints, while finding the right balance between exploiting what works and exploring new approaches.

- **Learn from the experience**: Extract what worked, what didn't, and why to refine future responses. The challenge is distilling specific, actionable insights that you can integrate into your understanding of the environment or your action plan.

This is an iterative process. As you act and receive feedback, you refine your understanding and adjust.

---

## An adaptable data generation system

Building an adaptable system requires operationalizing these principles into the data generation process:
- gathering sufficient context and constraints,
- translating high-level requirements into concrete generation actions,
- and learning from the generation outcomes to continuously improve the system

This pet project explores building such a system through a **multi-armed bandit architecture** that learns from experience.

### Pipeline Architecture

The pipeline runs in iterative batches, continuously adapting its generation strategy based on quality feedback:

```text
                               ╭──────────────────────────────╮
Specs →   Context   →  Init →  │  Loop: collected <= target?  │ → Dataset → Evaluate
         Extraction   State    ╰────────┬─────────────────────╯
                                ↑       ↓
                                │     Router → Generate → Quality ╮
                                │       ↑                         │
                                │   [ Adapt Feedback ] ←─ ─ ─ ─ ──╯
                                │                                 │
                                ╰─ ─ ─ Collect Samples ←─ ─ ─ ─ -─╯
```

**Core Modules:**

1. **Requirements specification**
   Input specifications define the domain, constraints, format requirements, and desired sample count.

2. **Generators**
   Two generation strategies: direct LLM calls (naive) and grammar-based PCFG templates (templater). Each strategy has multiple configurations (arms) with different parameters (conservative, balanced, creative). The router conditionally activates generator types based on spec features (e.g., grammar presence → templater arms).

3. **Router**
   An epsilon-greedy multi-armed bandit that selects generation strategies (arms) based on quality feedback. Arms are dynamically built from context: naive arms or templater arms when grammar is specified. Balances exploration of new strategies with exploitation of proven ones.

4. **Quality assessment**
   Multi-layered validation including rule-based checks (length, format, schema), model-based checks (similarity, semantic validation, entailment), diversity scoring, and LLM judge. Validators are configured per recipe via a top-level `validators:` block in each YAML under `config/recipes/`. JSON schemas auto-derive from grammar templates when not manually specified.

5. **Feedback loop**
   Quality outcomes are fed back as rewards to the router, which learns which strategies (arms) work best and continuously adapts its routing decisions through the epsilon-greedy bandit algorithm.


**Adaptive Mechanisms:**

- **Arm selection** uses epsilon-greedy strategy balancing exploration vs exploitation
- **Exploration rate** decays over iterations to shift from exploration to exploitation
- **LocalFeedbackState** tracks per-job iteration state, arm rewards, and performance metrics


**Evaluation**

Outside this generation loop, the system includes an **evaluation** module that tests the generated datasets on downstream tasks—training small models on the synthetic data to measure their real-world utility and feeding those results back as additional signals for the bandit.

**The result**

A system that doesn't just generate data, but adapts its approach based on what actually works, balancing exploration of new strategies with exploitation of proven ones.

---

## Try it!

```bash
# NAIVE: Adaptive temperature learning (see the system learn!)
python scripts/demo.py config/recipes/desert_plant_adaptability_example.yaml

# TEMPLATER: Grammar-based structured generation (high success rate)
python scripts/demo.py config/recipes/freediving_gear_reviews.yaml
```

---

## License

MIT


