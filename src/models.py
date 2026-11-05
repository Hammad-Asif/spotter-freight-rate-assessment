import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import FEATURES, LINEAR_FEATURES


#  using 3 error measure MAE, RMSE and MAPE percentage
def metrics(y_true, y_pred):
    err = y_pred - y_true
    return {"MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "MAPE%": 100 * np.mean(np.abs(err) / y_true)}


# trying linear model ridge regression first
def fit_ridge(frame):
    model = make_pipeline(
        ColumnTransformer([("num", StandardScaler(), LINEAR_FEATURES),
                           ("cat", OneHotEncoder(handle_unknown="ignore"), ["equipment"])]),
        Ridge(alpha=1.0),
    )
    return model.fit(frame, np.log(frame["rpm"]))


# now using the model to capture the non-linear effects
def fit_gbm(frame, features=FEATURES):
    model = HistGradientBoostingRegressor(
        loss="absolute_error", learning_rate=0.1, max_iter=3000, max_leaf_nodes=31,
        min_samples_leaf=40, l2_regularization=1.0, categorical_features=["equipment_code"],
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=50, random_state=42,
    )
    return model.fit(frame[features], np.log(frame["rpm"]))


def blend_predict(df, gbm_model, ridge_model):
    log_rpm = (gbm_model.predict(df[FEATURES]) + ridge_model.predict(df)) / 2
    return np.exp(log_rpm) * df["distance"].to_numpy()
