# An Adaptable Data Generation System

## The Philosophy of Adaptation

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

### Core Pipeline

The generation pipeline flows through five interconnected modules:

1. **Requirements specification**
   Input specifications define the domain, constraints, format requirements, and desired sample count.

2. **Generators**
   Multiple generation strategies—currently implementing direct LLM calls (naive) with different parameter configurations. The architecture is designed to support additional strategies in the future—each representing an "arm" the system can select.

3. **Router**
   An epsilon-greedy multi-armed bandit that selects generation strategies (arms) based on quality feedback. Currently implements three arms using the same generator with different temperature/top_p parameters (conservative, balanced, creative), balancing exploration of new strategies with exploitation of proven ones.

4. **Quality assessment**
   Multi-layered validation including rule-based checks (length, format, schema), model-based checks (similarity, semantic validation, entailment), and diversity scoring.

5. **Feedback loop**
   Quality outcomes are fed back as rewards to the router, which learns which strategies (arms) work best and continuously adapts its routing decisions through the epsilon-greedy bandit algorithm.

### Evaluation

Outside this generation loop, the system includes an **evaluation** module that tests the generated datasets on downstream tasks—training small models on the synthetic data to measure their real-world utility and feeding those results back as additional signals for the bandit.

---

**The result**: A system that doesn't just generate data, but adapts its approach based on what actually works, balancing exploration of new strategies with exploitation of proven ones.

---

## Architecture

The pipeline runs in iterative batches:
1. **Context Extraction**: Build context from Spec (domain, constraints, sample count)
2. **Router**: Select generator arm and parameters based on context + feedback state
3. **Generator**: Produce batch of samples
4. **Quality Assessment**: Score and filter samples
5. **Feedback Engine**: Compute metrics and update state
6. Repeat until target samples collected

The system adapts:
- **Arm selection** uses epsilon-greedy strategy balancing exploration vs exploitation
- **Exploration rate** decays over iterations to shift from exploration to exploitation
- **LocalFeedbackState** tracks per-job iteration state, arm rewards, and performance metrics

---

## License

MIT
