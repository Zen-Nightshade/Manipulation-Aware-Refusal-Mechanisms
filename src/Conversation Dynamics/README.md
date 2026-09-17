# Conversation Dynamics

This module investigates whether explicitly modeling conversational context and turn-to-turn changes provides additional signal for psychological manipulation detection beyond analyzing text in isolation.

The research spans two experimental phases:

1. **Phase I — Initial Sequence Modeling:** An ablation over five BiLSTM architectures, progressing from order-agnostic bag-of-turns pooling to hierarchical and explicit transition models.
2. **Phase II — Controlled Representation Study:** A rigorous follow-up using frozen `all-MiniLM-L6-v2` embeddings and linear classifiers across five feature configurations to isolate the predictive value of specific conversational dynamics.

---

## Experimental Progression & Results

### Phase I: BiLSTM Sequence Modeling

| Model | Architecture | Macro F1 | Micro F1 |
| :--- | :--- | :---: | :---: |
| **Model 1 — Mean Pooling** | BiLSTM over turns; mean-pooled | 0.215 | 0.336 |
| **Model 2 — Max + Mean Pooling** | BiLSTM; concatenates mean and max pool | **0.250** | **0.357** |
| **Model 3 — Attention Pooling** | BiLSTM with learned attention over turns | 0.237 | 0.349 |
| **Model 4 — Hierarchical LSTM** | Turn BiLSTM feeds dialogue LSTM | 0.161 | 0.284 |
| **Model 5 — Explicit Transitions** | Difference ($v - u$) and similarity ($v \times u$) | 0.213 | 0.352 |

### Phase II: Controlled Representation Study (SentenceTransformer)

| Model | Feature Configuration | Macro F1 | Micro F1 |
| :--- | :--- | :---: | :---: |
| **Model 1 — Content Baseline** | `Mean(P1)` | 0.1712 | 0.3012 |
| **Model 2 — Context Baseline** | `Mean(P1)` + `Mean(P2)` | 0.2237 | 0.3086 |
| **Model 3 — Peak Dynamics** | Context + Peak Trajectory + Peak Interaction | 0.2224 | 0.3377 |
| **Model 4 — Asymmetry + Convergence** | Context + `Mean(P1) - Mean(P2)` + `Mean(P1) ⊙ Mean(P2)` | 0.2105 | 0.3160 |
| **Model 5 — Victim Volatility + Dominance** | Context + `Max(P2) - Min(P2)` + `Max(P1) - Max(P2)` | **0.2433** | **0.3494** |

---

## Key Findings

- **Spikes beat averages:** The manipulator's most extreme moment is more informative than their mean tone. Max Pooling (Phase I) and Peak Dynamic features (Phase II) consistently outperform mean-only baselines.
- **Context is essential:** Including the victim's (Person2's) conversational stance significantly improves detection over content alone (Macro F1: 0.1712 → 0.2237).
- **The victim's destabilisation is the strongest dynamic signal:** Model 5 in Phase II captures victim emotional volatility (`Max(P2) - Min(P2)`) and peak dominance gap (`Max(P1) - Max(P2)`), achieving the best performance in Phase II (Macro F1: 0.2433, Micro F1: 0.3494).
- **Step-by-step sequential tracking underperforms:** Hierarchical LSTMs and explicit turn-difference features consistently fall behind simpler aggregation strategies on short dialogue snippets due to overfitting.

📖 **[Read the Full Research Report →](Report.md)**

---

## Notebooks

| Notebook | Contents |
| :--- | :--- |
| [`01_initial_bilstm.ipynb`](01_initial_bilstm.ipynb) | Early BiLSTM iterations and dataset inspection |
| [`02_advanced_bilstm.ipynb`](02_advanced_bilstm.ipynb) | Full Phase I: 5-model ablation study with F1 evaluation plots |
| [`03_sentence_transformer_experiment.ipynb`](03_sentence_transformer_experiment.ipynb) | Full Phase II: 5 controlled feature configurations with frozen embeddings & F1 plots |

---

## Module Structure

```
Conversation Dynamics/
├── config/
│   └── paths.py                  # Centralised path definitions
├── embedder.py                   # Computes and caches MiniLM embeddings
├── dataset.py                    # Feature construction from cached embeddings
├── models.py                     # All 5 Phase II linear classifier definitions
├── 01_initial_bilstm.ipynb
├── 02_advanced_bilstm.ipynb
├── 03_sentence_transformer_experiment.ipynb
├── README.md
└── Report.md
```
