# Freight Rate Prediction

Solution for the Spotter Machine Learning Engineer assessment. The model predicts the posted rate for truckloads in November and December 2025, using labelled loads from January to October 2025.

The final model is an average of a ridge regression and a gradient boosting model, both trained on log price per mile. On a September–October holdout it is about 2.3% off on average, against 3.9% for a simple lookup baseline.

## Project structure

```
├── data/                     # input files (not committed, see below)
├── src/
│   ├── eda.ipynb             # data exploration and data-quality checks
│   ├── features.py           # cleaning and feature engineering
│   ├── models.py             # metrics, ridge, gradient boosting and the blend
│   ├── evaluate.py           # time split, model comparison and rolling backtest
│   ├── predict.py            # final training and prediction files
│   └── score.py              # scorer provided with the assessment
├── scorer_results/
│   └── candidate_december.png
├── validation_predictions.csv
└── requirements.txt
```

## Setup

Tested with Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux
python -m pip install -r requirements.txt
```

## Data

The `data/` folder is in `.gitignore`, so the input files need to be added before running anything. Put the four assessment files in `data/` with these names:

```
data/train_test.csv
data/validation.csv
data/validation-predictions-template.csv
data/december-chart-inputs.csv
```

## How to run

The scripts read and write files with paths relative to `src/` (for example `../data/train_test.csv`), so run them from inside the `src` folder:

```bash
cd src
python evaluate.py
python predict.py
python score.py --predictions ../validation_predictions.csv --december-predictions ../data/december-chart-inputs.csv --output-dir ../scorer_results
```

- `evaluate.py` prints the holdout results, the feature tests, the rolling backtest, the unseen-city check and feature importance. It retrains several models, so it takes a few minutes.
- `predict.py` retrains the final model on all labelled data, writes `validation_predictions.csv` to the project root and fills the `predicted_rate` column of `data/december-chart-inputs.csv`.
- `score.py` checks both files and saves the December chart to `scorer_results/candidate_december.png`.

On Windows, scikit-learn may print a warning about not finding the number of physical CPU cores. It doesn't affect the results.

## Approach

- Cleaning. Negative weights were sign errors, so they are made positive. Missing weights are filled with the median for the equipment type, and missing `market_index` with that day's median. Loads priced below $1 or above $5 per mile (1.35%) are removed from training only.
- Features. Distance, straight-line distance and route directness, coordinates and direction of travel, equipment, weight, `market_index`, and month, day of year and day of week. City names are not used because 8 validation cities never appear in training. `quote_signal` is not used because its relationship with price changes sign from month to month.
- Target. The models predict log(rate / distance), which is converted back to dollars afterwards.
- Validation. Split by time: trained on January–August and tested on September–October, plus a three-window rolling backtest. A random split would leak prices from neighbouring days.

| Model (Sep–Oct holdout) | MAE | MAPE |
|---|---|---|
| Baseline (median price per mile lookup) | $86.24 | 3.88% |
| Ridge regression | $58.15 | 2.46% |
| Gradient boosting | $58.76 | 2.37% |
| Blend (final model) | $55.29 | 2.27% |

Scores are on holdout loads with plausible prices. The full write-up, including the data split and the December chart, is in the report.
