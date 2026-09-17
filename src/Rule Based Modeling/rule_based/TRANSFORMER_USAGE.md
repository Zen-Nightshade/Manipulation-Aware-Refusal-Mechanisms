# Transformer Embeddings Usage Guide

## Quick Start

### **1. Baseline Only (Fast - ~2-3 seconds)**
```bash
python3 train.py --baseline-only
```
Runs the original features (Empath + VADER + Interaction + n-grams) without transformers.

---

## **2. Test with Subset (Local Machine - ~30-60 seconds per transformer)**

### Try SentenceTransformers (lightest, fastest)
```bash
python3 train.py --subset 0.1 --transformer sentence-transformers
```
- `--subset 0.1`: Uses 10% of dataset (faster testing)
- `--transformer sentence-transformers`: Lightweight embeddings (384-dim)
- Caches embeddings in `./embeddings_cache/`

### Try DistilBERT (lightweight BERT)
```bash
python3 train.py --subset 0.1 --transformer distilbert
```
- Smaller, faster BERT variant
- 768-dim embeddings

### Try RoBERTa (best quality, slowest)
```bash
python3 train.py --subset 0.1 --transformer roberta
```
- Contextual embeddings, better quality
- 768-dim embeddings
- Slower on CPU (~few minutes)

---

## **3. Full Dataset (CPU or Local Machine)**

If you have time and want full results:
```bash
python3 train.py --transformer sentence-transformers
```
- Uses 100% of dataset
- First run: slow (depends on transformer size)
- Caches embeddings, so subsequent runs are instant

---

## **4. Run on Kaggle GPU (Recommended for Full Dataset)**

### Option A: Direct Notebook
```python
# In Kaggle notebook, run:
!cd /kaggle/working && python3 train.py --transformer distilbert
```

### Option B: Full Comparison Script
```python
# Run all transformers sequentially
transformers = ['sentence-transformers', 'distilbert', 'roberta']
for t in transformers:
    print(f"\n{'='*70}\nTesting {t.upper()}\n{'='*70}")
    os.system(f'python3 train.py --transformer {t}')
```

---

## **Command Line Arguments**

```bash
python3 train.py [OPTIONS]

Options:
  --subset FRACTION           Use fraction of dataset (0.0-1.0)
                             Default: 1.0 (full dataset)
                             Example: 0.1 = 10%

  --transformer NAME         Transformer to use:
                             - sentence-transformers (fastest, default)
                             - distilbert (medium)
                             - roberta (best quality, slower)

  --cache-dir PATH          Directory to cache embeddings
                             Default: ./embeddings_cache

  --baseline-only           Run baseline features only (no transformers)

Examples:
  # Test with 20% subset
  python3 train.py --subset 0.2 --transformer sentence-transformers

  # Full dataset with best transformer
  python3 train.py --transformer roberta

  # Just baseline comparison
  python3 train.py --baseline-only
```

---

## **Embedding Caching**

Embeddings are automatically cached in `./embeddings_cache/`:
```
embeddings_cache/
├── sentence-transformers_embeddings.pkl  (384-dim, small)
├── distilbert_embeddings.pkl             (768-dim, medium)
└── roberta_embeddings.pkl                (768-dim, large)
```

**Benefits:**
- First run: Compute and cache embeddings
- Subsequent runs: Load from cache (instant)
- Clear cache: `rm -rf ./embeddings_cache/`

---

## **Performance Comparison Workflow**

### Step 1: Quick Test (Local)
```bash
# Test each transformer with small subset (10%)
python3 train.py --subset 0.1 --transformer sentence-transformers
python3 train.py --subset 0.1 --transformer distilbert
python3 train.py --subset 0.1 --transformer roberta
python3 train.py --baseline-only
```
**Time:** ~5-10 minutes total
**Output:** Quick comparison, see which transformer works best

### Step 2: Full Comparison (Kaggle GPU)
```bash
# Run on full dataset with GPU
python3 train.py --transformer sentence-transformers  # ~5 min cached
python3 train.py --transformer distilbert           # ~10 min
python3 train.py --transformer roberta              # ~15 min
python3 train.py --baseline-only                    # ~2 min
```
**Time:** ~30 minutes total
**Output:** Final performance metrics for all transformers

---

## **Interpreting Results**

Look at the **SUMMARY TABLE** at the end:
```
F1-Score (Macro)                        0.1750  ← Main metric
Hamming Loss                            0.1640  ← Secondary
Subset Accuracy                         0.1098  ← Challenge metric
```

**Comparison Strategy:**
- **Higher F1-Macro is better** (main focus)
- Compare F1-Macro across transformers
- Check if adding transformer improves over baseline:
  - If Baseline F1: 0.1750 → Transformer F1: 0.25+ → **Success!**
  - If similar → Transformer doesn't help enough to justify compute cost

---

## **GPU Detection**

The code automatically detects and uses GPU:
```python
GPU_AVAILABLE = torch.cuda.is_available()
DEVICE = "cuda" if GPU_AVAILABLE else "cpu"
```

**Check if GPU available:**
```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

---

## **Troubleshooting**

### Issue: Out of Memory (OOM)
```bash
# Use smaller subset
python3 train.py --subset 0.05 --transformer distilbert

# Or use lightest transformer
python3 train.py --subset 0.1 --transformer sentence-transformers
```

### Issue: Embeddings not loading
```bash
# Clear cache and recompute
rm -rf ./embeddings_cache/
python3 train.py --subset 0.1 --transformer sentence-transformers
```

### Issue: Missing dependencies
```bash
# Install transformers
pip install transformers sentence-transformers torch

# In Kaggle: already pre-installed
```

---

## **Expected Output Examples**

### Baseline Only (No Transformers)
```
Train shape: (2335, 5412)
Val shape: (500, 5412)
Test shape: (501, 5412)

===========================
TRAINING MODEL
===========================
Mode: (BASELINE ONLY)
Total features: 5412
```

### With SentenceTransformers
```
Train shape: (2335, 5796)  ← Features increased (+384 transformer embeddings)
Val shape: (500, 5796)
Test shape: (501, 5796)

===========================
TRAINING MODEL
===========================
Mode: + SENTENCE-TRANSFORMERS
Total features: 5796
```

---

## **Notes & Tips**

1. **Subset Mode:** Great for quick testing, use 10-20% for fast iteration
2. **Caching:** Embeddings cache speeds up repeated runs 100x
3. **Kaggle:** Free GPU makes full dataset runs practical (5-15 min)
4. **Local:** If no GPU, stick with SentenceTransformers + subset
5. **Comparison:** Save output files for comparison:
   ```bash
   python3 train.py --subset 1.0 --transformer sentence-transformers > results/sentence-transformers.txt
   python3 train.py --subset 1.0 --transformer distilbert > results/distilbert.txt
   ```
