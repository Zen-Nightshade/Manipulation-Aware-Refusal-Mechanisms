# From Static Averages to Conversation Dynamics: Approaches to Manipulation Detection

## 1. Motivation

Psychological manipulation in dialogue is rarely confined to isolated words; it often manifests through context, emotional shifts, and relational escalation over time. The central research question driving this module is: *Does explicitly modeling conversational order, interlocutor context, and turn-to-turn changes provide a stronger signal for manipulation detection than analyzing text in isolation?*

To answer this systematically, we conducted an incremental study across two experimental phases. Phase I explored five recurrent neural architectures, progressing from order-agnostic pooling to explicit transition tracking. Phase II then built on Phase I's key insight by using frozen semantic representations to rigorously isolate which conversational features carry the manipulation signal — stripping away the confound of representation learning entirely.

---

## 2. Phase I: Initial Sequence Modeling

### Methodology

In Phase I, we trained BiLSTM-based models directly on the dialogue snippets. Each model was trained end-to-end: the BiLSTM learned both how to represent utterances and how to combine them. The five architectures were organized into two groups based on whether they respect conversational order.

**Order Agnostic (Bag-of-Turns Models)**

These models treat the conversation as an unordered set of utterances. They capture the aggregate distribution of features but cannot model escalation or sequential reaction.

| Model | Architecture | Design Rationale |
| :--- | :--- | :--- |
| **Model 1 — Mean Pooling** | BiLSTM; averages all turn encodings | Manipulation should alter the overall tone, but averaging may dilute sharp spikes |
| **Model 2 — Max + Mean Pooling** | BiLSTM; concatenates mean and max pool | The "worst moment" of a conversation may be more diagnostic than its average |
| **Model 3 — Attention Pooling** | BiLSTM with learned attention over turns | Let the network learn which turns carry manipulative weight, rather than assuming all turns contribute equally |

**Sequential History (Dynamic Models)**

These models attempt to track the temporal flow of the conversation.

| Model | Architecture | Design Rationale |
| :--- | :--- | :--- |
| **Model 4 — Hierarchical LSTM** | Turn-level BiLSTM feeds a dialogue-level LSTM | Implicitly learn conversation flow and the pattern of escalation over ordered turns |
| **Model 5 — Explicit Transitions** | Computes $v - u$ and $v \times u$ between consecutive turns | Explicitly encode behavioral shifts between each adjacent pair of utterances |

### Results

| Model | Macro F1 | Micro F1 |
| :--- | :---: | :---: |
| Model 1 — Mean Pooling | 0.215 | 0.336 |
| **Model 2 — Max + Mean Pooling** | **0.250** | **0.357** |
| Model 3 — Attention Pooling | 0.237 | 0.349 |
| Model 4 — Hierarchical LSTM | 0.161 | 0.284 |
| Model 5 — Explicit Transitions | 0.213 | 0.352 |

### Analysis

The clearest finding from Phase I is that **Max + Mean Pooling outperforms all other configurations**, including the more complex sequential models. This points to an important property of short dialogue snippets: manipulation is often concentrated in a single extreme moment — an accusation, an intimidation, a sudden guilt-trip — rather than distributed evenly across turns. Mean pooling obscures this; Max pooling preserves it.

The sequential models (Models 4 and 5) consistently underperform their simpler bag-of-turns counterparts, despite being theoretically richer. This is most likely a capacity-vs-data tradeoff: these models have substantially more parameters, and the dataset of short snippets does not provide sufficient signal to prevent overfitting. The performance drop of the Hierarchical LSTM is particularly sharp, suggesting that stacking two LSTMs amplifies this problem.

The key insight carried forward into Phase II: **the manipulative signal resides in spikes, not in sequential transition patterns.**

---

## 3. Phase II: Controlled Representation Study

### Methodology

Phase I's end-to-end training meant that any difference in model performance could be attributed to the classifier architecture, the representation quality, or their interaction. Phase II eliminates this confound by freezing the text representations entirely.

All utterances are encoded using `sentence-transformers/all-MiniLM-L6-v2`, a strong pretrained model producing 384-dimensional embeddings. These embeddings are cached and fixed throughout all experiments. Each model then uses a simple two-layer linear head, so observed performance differences are driven purely by **which features are constructed from those embeddings** — not by the classifier's capacity to learn representations.

Five models were evaluated across three tiers of feature complexity:

**Tier 1 — Baseline**

| Model | Features | Description |
| :--- | :--- | :--- |
| **Model 1 — Content Baseline** | Mean(P1) | Mean representation of the manipulator's turns only. Establishes the floor. |
| **Model 2 — Context Baseline** | Mean(P1) ⊕ Mean(P2) | Adds the victim's mean representation. Tests whether knowing the victim's overall conversational stance helps. |

**Tier 2 — Peak Dynamics (Informed by Phase I)**

| Model | Features | Description |
| :--- | :--- | :--- |
| **Model 3 — Peak Dynamics** | Context + Peak Trajectory + Peak Interaction | Applies Phase I's Max Pooling insight: `Max(P1) − Mean(P1)` captures the manipulator's internal spike intensity; `Max(P1) − Mean(P2)` captures that spike relative to the victim's tone. |

**Tier 3 — Relational Dynamics (Novel)**

| Model | Features | Description |
| :--- | :--- | :--- |
| **Model 4 — Asymmetry + Convergence** | Context + Asymmetry + Convergence | `Mean(P1) − Mean(P2)` captures the persistent tonal dominance gap. `Mean(P1) ⊙ Mean(P2)` (element-wise product) captures shared semantic alignment — topics the manipulator steers the victim toward. |
| **Model 5 — Victim Volatility + Peak Dominance** | Context + P2 Volatility + Dominance | `Max(P2) − Min(P2)` captures how much the victim's emotional state was destabilized across their turns. `Max(P1) − Max(P2)` captures the peak-to-peak power gap between both speakers. |

### Results

| Model | Macro F1 | Micro F1 |
| :--- | :---: | :---: |
| Model 1 — Content Baseline | 0.1712 | 0.3012 |
| Model 2 — Context Baseline | 0.2237 | 0.3086 |
| Model 3 — Peak Dynamics | 0.2224 | 0.3377 |
| Model 4 — Asymmetry + Convergence | 0.2105 | 0.3160 |
| **Model 5 — Victim Volatility + Peak Dominance** | **0.2433** | **0.3494** |

### Analysis

**Context is essential.** Model 2's improvement over Model 1 (0.1712 → 0.2237 Macro F1) confirms that the manipulator's content alone is insufficient. Manipulation is inherently relational; the victim's responses carry information about what kind of pressure is being applied.

**Peak Dynamics validate Phase I.** Model 3 achieves the highest Micro F1 of all three baseline/peak models (0.3377), confirming the Phase I insight: maximum-intensity features outperform mean-only features. However, it does not improve Macro F1 over the Context Baseline, suggesting peak features help detect the more common manipulation classes but offer less benefit for rare ones.

**Asymmetry + Convergence falls short.** Model 4 underperforms the Context Baseline on Macro F1. While sustained tonal dominance and shared topic alignment are intuitively linked to manipulation, mean-pooled asymmetry vectors are too coarse to isolate this signal reliably. The effect may be more visible in longer conversations or with more nuanced aggregation.

**Victim Volatility is the strongest signal.** Model 5 achieves the highest Macro F1 (0.2433) and Micro F1 (0.3494) across all Phase II experiments, surpassing even the carefully designed Peak Dynamics features. The result tells a clear story: a victim whose emotional register is highly volatile — whose responses swing dramatically from turn to turn — is the most consistent observable signal that manipulation is actively working. Paired with the peak dominance gap, this configuration captures both *who is pushing* (the manipulator's extreme ceiling) and *who is being pushed* (the victim's instability).

---

## 4. Cross-Phase Comparison

The two phases are not directly comparable — Phase I uses end-to-end BiLSTMs with a learned vocabulary, while Phase II uses fixed semantic embeddings — but placing them side-by-side reveals a consistent pattern.

| Approach | Best Model | Macro F1 | Micro F1 |
| :--- | :--- | :---: | :---: |
| Phase I (BiLSTM) | Max + Mean Pooling | 0.250 | 0.357 |
| Phase II (SentenceTransformer) | Victim Volatility + Peak Dominance | 0.243 | 0.349 |

Phase I's best model slightly edges Phase II on raw numbers, but this reflects the stronger representation backbone (a trained BiLSTM vs fixed embeddings from a general-purpose encoder), not a better feature design. The Phase II experiments isolate the contribution of feature engineering independently of representation quality. In a full pipeline where both components are optimised together, the victim-centric dynamic features identified in Phase II would be the natural starting point.

---

## 5. Conclusion

Both experiments converge on the same finding: **the victim's destabilisation is the most reliable observable signal of manipulation in short dialogue snippets.**

Manipulation cannot be detected reliably from the manipulator's tone alone. It requires observing the relational context — specifically, how much the victim's emotional state fluctuates and how large the peak intensity gap between the two speakers is. Step-by-step sequential tracking (via Hierarchical LSTMs or explicit turn-difference features) does not provide additional value over these simpler, victim-centric statistics on this dataset.

The practical implication is that manipulation detection systems should invest in modeling the *response pattern of the target*, not only the content of the manipulator's utterances.
