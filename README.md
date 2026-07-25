# Walmart Store Sales Forecasting

Final project for our ML class, based on the [Walmart Recruiting - Store Sales Forecasting](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting) Kaggle competition. The task is predicting weekly sales for ~3,300 store-department combinations across 45 Walmart stores, with holiday weeks (Thanksgiving, Christmas, Super Bowl, Labor Day) weighted 5x in the scoring metric since they're the hardest to get right and matter the most for the business.

Done by me (tamari1990) and my teammate Tako (tgura23).

## What we tried

We ended up comparing a lot more models than we originally planned, partly because once you set up the data pipeline once it's not that much extra work to try another one, and partly because some of them just didn't work well and we wanted to understand why instead of giving up.

| Model | Local holdout WMAE | Notes |
|---|---|---|
| **XGBoost** | 1639.12 | best model overall, used for the Kaggle submission |
| LightGBM | 1672.26 | very close second |
| N-BEATS | 2161.25 | best config after two rounds of tuning (had a holiday-underprediction issue at first) |
| PatchTST | 2190.61 | |
| DLinear | 2532.49 | |
| ARIMA | 2579.07 | per-series classical models, huge variance across series |
| TimesFM | 2618.40 | Google's pretrained foundation model, zero-shot (no training at all) |
| TFT | 3996.37 | see note below, this number comes with an asterisk |
| Prophet | 6932.91 | |

Gradient boosted trees won pretty clearly. Not hugely surprising in hindsight — this dataset is only 143 weeks of history, so the deep learning models don't really get enough data per series to make their extra capacity worth it, and XGBoost/LightGBM handle the mix of categorical (Store, Dept, Type) and continuous features naturally.

### About that XGBoost number

We actually caught a real bug after submitting to Kaggle the first time. The local holdout WMAE (1639.12) looked great, but our first Kaggle submission scored ~8260, almost 5x worse. Turned out the issue was in how we generated predictions for the real test set: Kaggle's `test.csv` has no sales data at all (obviously, that's what we're predicting), but our feature pipeline computed lag/rolling features by featurizing the whole 39-week test set in one shot. Since there's no real data to compute a "sales 4 weeks ago" feature for week 20 of the test set, those features silently went NaN for most of the test period.

Fixed it by predicting one week at a time and feeding each week's own prediction back in as pseudo-history for the next week's features. Resubmitted and got 2961.72, much more in line with what we'd expect. (We also found a smaller, separate issue while debugging our own validation of the fix — an earlier "confirmed the fix works, 1580.80" number turned out to be leaked too, because we accidentally validated the production model, which had already been trained on the same data we were testing it on. A properly retrained model gives 2364.49 locally, which lines up with the Kaggle score a lot better.)

Not going to pretend that was fun to find, but it was a good reminder to actually think about what a model's evaluation set has and hasn't seen, not just trust a good-looking number.

### About the TFT number

Temporal Fusion Transformer was the one that gave us the most trouble. Locally on CPU it's just too slow — we measured about 16 minutes per epoch for the full grid search, and a 10 hour run didn't even finish the first hyperparameter combination. We tried running it on Colab with a GPU instead, which actually worked, but the session kept dying from Colab's idle timeout before the whole grid search could finish (twice).

Given the deadline, we ended up training a much smaller, untuned version locally just to have a real number to report (fixed hyperparameters reused from an earlier successful baseline run, no CV, 15 epochs) — that's where the 3996.37 comes from. It's a legitimate result, just not from the fully tuned pipeline the notebook is actually set up to run. If you have GPU time to spare, `model_experiment_TFT.ipynb` is Colab-ready and should give a real, tuned number in a few hours.

### Other things we tried that didn't pan out

- Recursive block-by-block forecasting for TimesFM instead of one direct 52-week forecast — made things slightly worse (2787.23 vs 2618.40), which told us the problem wasn't really the horizon length, it's that TimesFM has zero calendar/holiday information to work with in our zero-shot setup.
- Adding a feature for the actual pre-Christmas sales peak, which we noticed happens a full week before Kaggle's official "Christmas" holiday label. The data insight was real (confirmed directly against the training data — the officially labeled Christmas week is actually below Thanksgiving in sales, the real spike is the week before), but the feature itself made the real Kaggle score worse, most likely because we only had two actual historical instances of it to learn from and no way to verify the date lines up correctly for the actual test year.

## Repo layout

- `model_experiment_*.ipynb` — one notebook per model, each following roughly the same structure (setup, local train/test split, CV, tuning, MLflow logging, plots, full pipeline)
- `utils/` — shared feature engineering, model code, and metrics used across notebooks
- `model_inference.ipynb` — generates the actual Kaggle submission from the trained XGBoost pipeline
- `eda_exploration.ipynb` — initial data exploration
- `reports/` — some writeups from the N-BEATS tuning process
- `data/raw/` — the competition data (train/test/features/stores csvs)
- `models/` — saved trained pipelines (joblib/pt files)
- `plots/` — generated figures from each notebook

## Running things

Most notebooks are self-contained if you have the requirements installed (`requirements.txt` for the general stuff, `requirements-dlinear.txt`/`requirements-tft.txt` for the deep learning environments we split out separately because of dependency conflicts). We used [DagsHub](https://dagshub.com/tgela23/ml-final-project) for MLflow experiment tracking, so a lot of the CV/tuning results are logged there rather than only living in the notebook outputs.

`model_experiment_TFT.ipynb` needs a GPU to actually finish in reasonable time — it's got clone/pip-install cells at the top ready to go in Colab.
