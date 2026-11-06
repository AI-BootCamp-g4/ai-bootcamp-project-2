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
Put them next to the notebook (`problem2/`) to run it.
