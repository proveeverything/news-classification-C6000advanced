# News Topic Classification

Classify news articles into four topics — **World, Sports, Business, Sci/Tech** —
from their title and description, using classic NLP (TF-IDF) and a small neural network.

> An end-to-end text-classification project: data cleaning → EDA →
> feature engineering → model comparison → **honest** evaluation.

---

## 1. Problem

Given a news headline and a short description, predict which of four categories it
belongs to. This is a **multi-class classification** problem on **unstructured text**,
which makes preprocessing and feature engineering the core challenge (unlike tidy
numeric tables, raw text carries noise, varying length, and no ready-made features).

## 2. Dataset

- **Source:** AG News (subset), `data/ag_news.csv`
- **Size:** 7,600 articles after cleaning
- **Balance:** 1,900 articles per class (perfectly balanced → accuracy is a fair metric)
- **Columns:** `Class Index` (1=World, 2=Sports, 3=Business, 4=Sci/Tech), `Title`, `Description`

## 3. Project structure

```
news-classification/
├── README.md            # this file
├── requirements.txt     # dependencies
├── .gitignore
├── data/
│   └── ag_news.csv      # raw dataset
├── src/                 # reusable logic (imported by the notebook)
│   ├── preprocess.py    #   text cleaning + data loading
│   ├── features.py      #   train/test split + TF-IDF
│   ├── train.py         #   three classic models + evaluation
│   └── neural_net.py    #   PyTorch MLP (vocab, model, training loop)
└── notebooks/
    └── analysis.ipynb   # the narrative: EDA, modeling, results, plots
```

The notebook only tells the story; all logic lives in `src/` so it is reusable and
testable. Cleaning is a single function called on every column — no copy-paste bugs.

## 4. How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/analysis.ipynb   # then "Run All"
```

> **Note (Windows + conda):** if importing `torch` raises `OMP: Error #15` (duplicate
> OpenMP runtime), set the environment variable `KMP_DUPLICATE_LIB_OK=TRUE` before running.
> The notebook already sets this in its first cell.

## 5. Approach

1. **Cleaning** (`preprocess.py`) — decode HTML entities (`&lt;` → `<`, so we don't leave
   junk tokens like `lt`/`quot`/`39`), lowercase, keep letters only, drop stopwords / HTML
   remnants / single-character fragments, then fuse `Title` + `Description` into one field.
2. **Features** (`features.py`) — TF-IDF, **fit on the training set only** to avoid leaking
   the test set into the IDF statistics.
3. **Classic models** (`train.py`) — Naive Bayes, Logistic Regression, Linear SVM, all scored
   on the held-out test set.
4. **Neural network** (`neural_net.py`) — a word-embedding + mean-pooling + MLP, evaluated on
   the **same** test split for a fair comparison.

## 6. Results

Test-set accuracy (same 80/20 stratified split for every model):

| Model | Test accuracy |
|---|---|
| **Naive Bayes** | **0.892** |
| Logistic Regression | 0.884 |
| Linear SVM | 0.880 |
| Neural Net (MLP) | 0.830 |

- The three classic models land within ~1% of each other — once the text is clean, **feature
  quality matters more than model choice**.
- **Naive Bayes wins narrowly** on this bag-of-words task.
- The **neural network underperforms (~0.83)**: averaging learned embeddings discards signal
  that TF-IDF keeps, and ~6k articles is too little to learn good embeddings from scratch.
  *Fancier ≠ better — always benchmark against simple baselines.*
- Main confusion is **Business ↔ Sci/Tech**, driven by shared company / finance vocabulary.

## 7. Key learnings

- **Honest evaluation changes the story.** The original coursework reported 96.76% — a
  *training* accuracy; on a held-out test set the network sits well below the simple models.
- **Cleaning order is knowledge.** Decode HTML *before* stripping symbols, or `&lt;` becomes
  the junk token `lt`.
- **No data leakage.** Fit TF-IDF / build the vocabulary on the training set only.
- **Interpretability surfaces shortcuts.** Logistic-regression weights are sensible, but reveal
  news-agency tags (`afp` / `ap`) leaking in as spurious signal.
