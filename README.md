# AI Bootcamp - Project 2

QBC13 AI Bootcamp, group project 2.

## Structure

```
problem2/
  q2_part1_part2.ipynb        Part 1 (EDA, Q1-Q6) + Part 2 (warranty aspect analysis)
  outputs/                    files written by the notebook
    manual_check.csv          manual labels for the warranty filter check
    verb_check.csv            manual labels for reviews dropped by the "guarantee as a verb" rule
    rma_check.csv             manual labels for reviews added by the word "rma"
    warranty_by_product.csv   warranty-related ratings per product
```

## Data

The raw data (`train_data.csv`, `test_data.csv`, `title_brand.csv`) and the GloVe vectors
(`glove.6B.100d.txt`) are too large for GitHub and are not included.

## Part 3: rating prediction

```
problem2/
  q2_part3_tfidf_stacking.ipynb   data checks, TF-IDF + LogisticRegression, reviewer/product history, final stacking
  q2_part3_transformer.ipynb      DistilRoBERTa fine-tuning on a Colab T4 GPU
  q2_submission.csv               final predictions for the test set (one column: predicted)
  outputs/part3/
    q2_submission_bedoon_eslah.csv  backup: the final model without the prior correction
    q2_submission_tfidf.csv         TF-IDF + LogisticRegression only
    q2_submission_stack_hist.csv    TF-IDF + reviewer/product history
    q2_submission_blend.csv         average of transformer and TF-IDF
    results_tfidf.csv               TF-IDF experiments
    results_final.csv               every model with micro-F1, macro-F1 and mean recall
    recall_per_rating.png           recall of each rating, transformer vs final model
```

Run order: steps 0 to 24 of `q2_part3_tfidf_stacking.ipynb`, then `q2_part3_transformer.ipynb` (needs a GPU),
then steps 25 to 29 of the first notebook. Both notebooks read the data from the `project_q2` folder on
Google Drive and save their intermediate files there.

| model | prior correction | micro-F1, normal validation | mean recall (estimate for a balanced test) |
|---|---|---|---|
| TF-IDF + LogisticRegression | no | 0.7076 | 0.5246 |
| TF-IDF + LogisticRegression | yes | 0.6650 | 0.5763 |
| DistilRoBERTa | no | 0.7697 | 0.6454 |
| stacking C (TF-IDF + transformer + history) | no | 0.7769 | 0.6460 |
| **stacking C, final file** | **yes** | 0.7465 | **0.6697** |

In train, 55% of the reviews are 5 stars. The test set does not look like that: the BBSE estimate gives about
20% for each rating, and the transformer predicts "5" for 57% of validation but only 26% of test. So the final
file uses the prior correction, and we expect a test score of about 0.67. If the test set turns out to have the
train distribution, `q2_submission_bedoon_eslah.csv` is the better file (0.777 on normal validation).

Not in the repo: `train.csv`, `test.csv`, `mapping.csv`, the `.pkl` / `.npy` intermediate files and the trained
model folder `best_distilroberta_n150000_len384` (a few hundred MB). They stay on Google Drive.
