import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from features import BASE_FEATURES, FEATURES, load_training_data
from models import blend_predict, fit_gbm, fit_ridge, metrics

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

BANDS = [0, 250, 500, 1000, 1500, 2500, 5000]


def band(d):
    return pd.cut(d["distance"], BANDS)


def evaluate(name, predict_fn, hold, hold_clean):
    row = {"model": name}
    for label, frame in [("all", hold), ("clean", hold_clean)]:
        m = metrics(frame["posted_rate"].to_numpy(), predict_fn(frame))
        row.update({f"{k} ({label})": v for k, v in m.items()})
    return row


def main():
    raw, train_all, weight_medians, is_outlier = load_training_data()
    print("Weight medians used for filling:\n", weight_medians.to_string())
    print("Remaining missing:", train_all[["weight", "market_index"]].isna().sum().to_dict())
    print(f"Outliers flagged: {is_outlier.sum()} of {len(train_all):,} ({is_outlier.mean():.2%})")

    # first train on data from Jan to August and test on september to October
    SPLIT_DATE = "2025-09-01"
    dev = train_all[train_all["date"] < SPLIT_DATE]
    hold = train_all[train_all["date"] >= SPLIT_DATE]
    dev_clean = dev[~is_outlier.loc[dev.index]]
    hold_clean = hold[~is_outlier.loc[hold.index]]
    print(f"Fit:     {dev['date'].min().date()} .. {dev['date'].max().date()}  {len(dev_clean):,} rows (outliers removed)")
    print(f"Holdout: {hold['date'].min().date()} .. {hold['date'].max().date()}  {len(hold):,} rows ({len(hold_clean):,} clean)")

    results = []

    baseline_table = dev_clean.groupby(["equipment", band(dev_clean)], observed=True)["rpm"].median()

    def baseline_predict(df):
        keys = pd.MultiIndex.from_arrays([df["equipment"], band(df)])
        return baseline_table.reindex(keys).to_numpy() * df["distance"].to_numpy()

    results.append(evaluate("Baseline (median rpm table)", baseline_predict, hold, hold_clean))

    ridge = fit_ridge(dev_clean)
    results.append(evaluate("Ridge regression",
                            lambda df: np.exp(ridge.predict(df)) * df["distance"].to_numpy(),
                            hold, hold_clean))

    gbm = fit_gbm(dev_clean)
    print("Boosting rounds used:", gbm.n_iter_)
    results.append(evaluate("Gradient boosting",
                            lambda df: np.exp(gbm.predict(df[FEATURES])) * df["distance"].to_numpy(),
                            hold, hold_clean))

    #  compare the models
    results.append(evaluate("Blend (50% GBM + 50% Ridge)",
                            lambda df: blend_predict(df, gbm, ridge),
                            hold, hold_clean))
    print(pd.DataFrame(results).set_index("model").round(2))

    # using the boosting model with quote_signal and date removed
    ablations = {
        "with quote_signal": FEATURES + ["quote_signal"],
        "without quote_signal": FEATURES,
        "without date features": BASE_FEATURES,
    }
    rows = []
    for name, feats in ablations.items():
        m = fit_gbm(dev_clean, feats)
        rows.append(evaluate(name,
                             lambda df, m=m, f=feats: np.exp(m.predict(df[f])) * df["distance"].to_numpy(),
                             hold, hold_clean))
    print(pd.DataFrame(rows).set_index("model").round(2))

    # rolling backtest
    windows = [("2025-07-01", "2025-09-01"), ("2025-08-01", "2025-10-01"), ("2025-09-01", "2025-11-01")]
    rows = []
    for start, end in windows:
        fit_rows = train_all[(train_all["date"] < start) & ~is_outlier]
        test_rows = train_all[(train_all["date"] >= start) & (train_all["date"] < end) & ~is_outlier]
        g, r = fit_gbm(fit_rows), fit_ridge(fit_rows)
        y, dist = test_rows["posted_rate"].to_numpy(), test_rows["distance"].to_numpy()
        preds = {
            "GBM + quote_signal": np.exp(fit_gbm(fit_rows, FEATURES + ["quote_signal"]).predict(test_rows[FEATURES + ["quote_signal"]])) * dist,
            "GBM": np.exp(g.predict(test_rows[FEATURES])) * dist,
            "Ridge": np.exp(r.predict(test_rows)) * dist,
            "Blend": blend_predict(test_rows, g, r),
        }
        for name, p in preds.items():
            rows.append({"test": f"{start[:7]} & next month", "model": name, **metrics(y, p)})

    backtest = pd.DataFrame(rows)
    print(backtest.pivot(index="test", columns="model", values="MAPE%").round(2))
    print("\nAverage across windows:")
    print(backtest.groupby("model")[["MAE", "RMSE", "MAPE%"]].mean().round(2).sort_values("MAPE%"))

    #  testing unseen city names by hiding the random 8 cities
    rng = np.random.default_rng(0)
    cities = np.array(sorted(set(dev_clean["pickup"]) | set(dev_clean["delivery"])))
    held_cities = set(rng.choice(cities, size=8, replace=False))

    def touches(d):
        return d["pickup"].isin(held_cities) | d["delivery"].isin(held_cities)

    fit_rows = dev_clean[~touches(dev_clean)]
    g, r = fit_gbm(fit_rows), fit_ridge(fit_rows)
    print("Cities hidden from training:", sorted(map(str, held_cities)))
    for label, frame in [("seen cities", hold_clean[~touches(hold_clean)]),
                         ("unseen cities", hold_clean[touches(hold_clean)])]:
        m = metrics(frame["posted_rate"].to_numpy(), blend_predict(frame, g, r))
        print(f"{label:14s} n={len(frame):5,}  MAE=${m['MAE']:.1f}  MAPE={m['MAPE%']:.2f}%")

    # Shuffles one feature at a time and measures the effect on error
    imp = permutation_importance(gbm, hold_clean[FEATURES], np.log(hold_clean["rpm"]),
                                 n_repeats=3, random_state=0, scoring="neg_mean_absolute_error")
    print(pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
