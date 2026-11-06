# Amazon Review Rating Prediction — Experimental Study

This folder documents a sequence of experiments to predict **1–5-star Amazon review ratings** from review text, with an optional signal from each reviewer's past ratings. The goal was to improve predictive performance while keeping training practical on Google Colab's T4 GPU.

## Experiments

| Version | Main decision | Validation Micro-F1 |
|---|---|---:|
| **V1** | Lightweight, fine-tuned Mini BERT baseline; 128-token review text | **70.54%** |
| **V2** | Increase to 256 tokens; use the review summary and head–tail truncation for long reviews | **72.19%** |
| **V3** | Test stopword removal on top of V2 | **71.81%** |
| **V4** | Return to V2 text processing and add time-valid user rating history | **72.20%** |

The experiments trained on **half of the available labeled data** to enable faster iteration. V2 delivered the main improvement; stopword removal reduced performance slightly, and user history provided only a marginal additional gain in this run.

## Folder contents

- `Quera_V1(3).ipynb` — baseline model and validation.
- `Quera_V2(3).ipynb` — longer, summary-aware input experiment.
- `Quera_V3(3).ipynb` — stopword-removal experiment.
- `test_train_user_comparison.ipynb` — analysis of train/test user overlap and usable prior rating history.
- `amazon_bert_mini_V4_history_alpha(1)(2).ipynb` — V4 training and validation with history-aware prediction.
- `V4_History_Final_TEST_No_Training(1)(1).ipynb` — evaluates the saved V4 model on the labeled internal test split, without retraining.
- `Amazon_Review_Sentiment_Experimental_Report_V1-V4.pdf` — full discussion of decisions, results, plots, and confusion matrices.

## Final result — V4 (internal test)

| Metric | Result |
|---|---:|
| **Micro-F1 / Accuracy** | **72.06%** |
| Macro-F1 | 57.07% |
| Mean Absolute Error (MAE) | 0.3680 |
| Test reviews | 62,921 |

**Note:** These final metrics are from a held-out **labeled internal test split**, not the separate unlabeled competition test file. The full PDF report provides detailed comparisons and limitations.
