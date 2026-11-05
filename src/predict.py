import pandas as pd

from features import add_features, clean, load_training_data
from models import blend_predict, fit_gbm, fit_ridge


def main():
    raw, train_all, weight_medians, is_outlier = load_training_data()

    # retrain all labelled data
    train_final = train_all[~is_outlier]
    gbm_final = fit_gbm(train_final)
    ridge_final = fit_ridge(train_final)
    print(f"Final models trained on {len(train_final):,} loads "
          f"({train_final['date'].min().date()} .. {train_final['date'].max().date()}), "
          f"boosting rounds: {gbm_final.n_iter_}")

    # predict 12000 validation loads
    valid_raw = pd.read_csv("../data/validation.csv", parse_dates=["date"])
    valid, _ = clean(valid_raw, weight_medians)
    valid = add_features(valid)
    valid["predicted_rate"] = blend_predict(valid, gbm_final, ridge_final).round(2)

    print(valid["predicted_rate"].describe().round(2).to_string())
    print("Missing predictions:", valid["predicted_rate"].isna().sum(),
          "| Non-positive:", (valid["predicted_rate"] <= 0).sum())

    # fill template and save it
    template = pd.read_csv("../data/validation-predictions-template.csv")
    submission = template[["load_id"]].merge(valid[["load_id", "predicted_rate"]], on="load_id", how="left")
    assert len(submission) == 12_000 and submission["predicted_rate"].notna().all()

    output_path = "../validation_predictions.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved {len(submission):,} rows -> {output_path}")
    print(submission.head())

    # fill missing December inputs
    december_path = "../data/december-chart-inputs.csv"
    december_raw = pd.read_csv(december_path)
    december = december_raw.drop(columns="predicted_rate").copy()
    december["date"] = pd.to_datetime(december["date"])

    # look each city coordinates from the training and validation data
    coords = pd.concat([
        df[[side, f"{side}_lat", f"{side}_lon"]].set_axis(["city", "lat", "lon"], axis=1)
        for df in (raw, valid_raw) for side in ("pickup", "delivery")
    ]).drop_duplicates("city").set_index("city")
    for side in ("pickup", "delivery"):
        december[f"{side}_lat"] = december[side].map(coords["lat"])
        december[f"{side}_lon"] = december[side].map(coords["lon"])

    daily_market = valid.groupby("date")["market_index"].median()
    december["market_index"] = december["date"].map(daily_market)

    assert december[["pickup_lat", "delivery_lat", "market_index"]].notna().all().all()
    print(december[["date", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "market_index"]].head())

    # predict december and fill inplace
    december_feat = add_features(clean(december, weight_medians)[0])
    december_raw["predicted_rate"] = blend_predict(december_feat, gbm_final, ridge_final).round(2)
    december_raw.to_csv(december_path, index=False)

    print(december_raw[["date", "predicted_rate"]].to_string(index=False))
    lane = raw[(raw["pickup"] == "Lexington") & (raw["delivery"] == "Fort Wayne") & (raw["equipment"] == "Dry Van")]
    print(f"\nDecember predictions: ${december_raw['predicted_rate'].min():.0f} - ${december_raw['predicted_rate'].max():.0f}")
    print(f"Historical Dry Van Lexington->Fort Wayne: ${lane['posted_rate'].min():.0f} - ${lane['posted_rate'].max():.0f} "
          f"(mean ${lane['posted_rate'].mean():.0f})")


if __name__ == "__main__":
    main()
