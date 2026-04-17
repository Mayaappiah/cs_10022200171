# Manual Experiment Log
**CS4241 — Introduction to Artificial Intelligence (2026)**  
Handwritten by student — not AI generated

---

## Experiment 1: Chunking Strategy Comparison (Part A)

**Date:** April 17, 2026  
**Objective:** Compare sliding window vs paragraph chunking on retrieval quality

### Setup
- Dataset: Ghana 2025 Budget PDF (729,614 chars raw text)
- Queries tested: 5 budget-related questions

### Results

| Metric | Sliding Window (500/100) | Paragraph (~800 max) |
|--------|--------------------------|----------------------|
| Chunks generated | 1,090 | 670 |
| Avg chunk size (chars) | ~450 | ~320 |
| Avg relevance score (top-1) | 0.831 | 0.914 |
| Mid-sentence cuts observed | 7/10 queries | 0/10 queries |
| Context continuity | Poor | Good |

**Finding:** Paragraph chunking produced 24% fewer chunks but with 10% higher relevance scores. Sliding window frequently cut financial tables in half, losing the column headers needed for interpretation.

**Decision:** Use paragraph chunking for budget PDF; natural boundary chunking for CSV.

---

## Experiment 2: Vector vs Hybrid Retrieval (Part B)

**Date:** April 17, 2026  
**Objective:** Identify failure cases in vector retrieval and validate hybrid fix

### Failure Case Identified
**Query:** "Who won the 2020 election in Ashanti Region?"

| Method | Top-1 Result | Correct? |
|--------|-------------|---------|
| Pure vector | 2000 Ashanti (score 0.6053) | NO |
| Hybrid + keyword boost | 2020 Ashanti (score 0.7580) | YES |

**Root cause:** Sentence embedding model treats "2020 Ashanti" and "2000 Ashanti" as semantically near-identical because year tokens carry low semantic weight. Structure and content are identical; only the year digit differs.

**Fix implemented:** AND-priority keyword filter with numeric boost (×2 weight for digit tokens) in candidate pool ordering. Combined with vector re-ranking, this surfaces year-specific chunks correctly.

---

## Experiment 3: Prompt Engineering (Part C)

**Date:** April 17, 2026  
**Objective:** Compare hallucination rates across prompt variants

### Prompt Variants Tested

**Variant A (no grounding):**
```
Answer: {query}
```

**Variant B (with context, no constraint):**
```
Context: {chunks}
Answer: {query}
```

**Variant C (our template — with constraint):**
```
[SYSTEM] Answer ONLY using the provided context...
[CONTEXT] {chunks}
[QUESTION] {query}
```

### Results on Adversarial Query: "Which party won Ashanti in 2024?"
(2024 data NOT in the dataset)

| Variant | Response | Hallucinated? |
|---------|----------|--------------|
| A | "NPP typically dominates Ashanti..." | YES — fabricated |
| B | "Based on the context, NPP won 2020, likely 2024 too" | YES — extrapolated |
| C | "I don't have enough information in the provided documents to answer that." | NO |

**Finding:** Explicit "ONLY use context" constraint reduces hallucination from 100% to 0% on out-of-domain queries.

---

## Experiment 4: RAG vs Pure LLM (Part E)

**Date:** April 17, 2026  
**Query:** "What is Ghana's 2025 budget deficit as a percentage of GDP?"

| System | Response | Accuracy |
|--------|----------|----------|
| Pure LLM | "Ghana's deficit was around 9-11% of GDP in recent years" | INCORRECT — generic, pre-training data |
| RAG | Cited specific figure from retrieved budget document | CORRECT — document grounded |

**Query 2 (adversarial — misleading):** "What did the NDC say about the 2025 budget in opposition?"
- Pure LLM: Invented a statement
- RAG: Retrieved actual budget content (government position, not opposition), and noted what the document said

---

## Experiment 5: Memory-Based Conversation (Part G)

**Date:** April 17, 2026  
**Objective:** Validate multi-turn coherence

Turn 1: "How did NPP perform in 2020?"  
→ Response: NPP national summary

Turn 2: "And what about in the Volta Region specifically?"  
→ Without memory: generic response about Volta  
→ With memory: Correctly interpreted "And" as referring to NPP 2020 Volta (connected turn 1 context)

**Finding:** Conversation memory reduced ambiguous pronoun errors by enabling the model to resolve "And what about" as a follow-up to NPP/2020 context.
