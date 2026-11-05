import numpy as np
import pandas as pd

EQUIPMENT = ["Dry Van", "Flatbed", "Reefer"]

BASE_FEATURES = [
    "distance", "log_distance", "haversine", "circuity",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon", "delta_lat", "delta_lon",
    "equipment_code", "weight", "weight_capped", "weight_was_negative", "market_index",
]
DATE_FEATURES = ["month", "day_of_year", "day_of_week"]
FEATURES = BASE_FEATURES + DATE_FEATURES
LINEAR_FEATURES = ["log_distance", "distance", "pickup_lat", "pickup_lon", "delivery_lat",
                   "delivery_lon", "weight", "weight_capped", "market_index", "month"]


def clean(df, weight_medians=None):
    df = df.copy()
    # use abs() to flip signs of negative weights
    df["weight_was_negative"] = (df["weight"] < 0).astype(int)
    df["weight"] = df["weight"].abs()
    df["weight_capped"] = (df["weight"] >= 47_500).astype(int)

    # Used median for missing weights
    if weight_medians is None:
        weight_medians = df.groupby("equipment")["weight"].median()
    df["weight"] = df["weight"].fillna(df["equipment"].map(weight_medians))

    # market_index is a daily market level, so day's median will be used to fill missing values
    df["market_index"] = df["market_index"].fillna(df.groupby("date")["market_index"].transform("median"))
    return df, weight_medians


# building the input model and excluding city names
def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def add_features(df):
    df = df.copy()
    df["log_distance"] = np.log(df["distance"])
    df["haversine"] = haversine_miles(df["pickup_lat"], df["pickup_lon"], df["delivery_lat"], df["delivery_lon"])
    df["circuity"] = df["distance"] / df["haversine"].clip(lower=1)
    df["delta_lat"] = df["delivery_lat"] - df["pickup_lat"]
    df["delta_lon"] = df["delivery_lon"] - df["pickup_lon"]
    df["equipment_code"] = pd.Categorical(df["equipment"], categories=EQUIPMENT).codes
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


def load_training_data():
    raw = pd.read_csv("../data/train_test.csv", parse_dates=["date"])
    train_all, weight_medians = clean(raw)
    train_all = add_features(train_all)
    train_all["rpm"] = train_all["posted_rate"] / train_all["distance"]

    # dropping the load prices and keep only those which are in between $1 and $5
    is_outlier = ~train_all["rpm"].between(1.0, 5.0)
    return raw, train_all, weight_medians, is_outlier
