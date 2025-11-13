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