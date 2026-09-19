"""Shared paths, features and evaluation helpers.

The app, the training script and the plots all import from here so the model and the
reported numbers can never drift apart.
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    root_mean_squared_error,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_PATH = DATA_DIR / "continuous_dataset.csv"
CLEAN_PATH = DATA_DIR / "cleaned_data.csv"
PREDISPATCH_PATH = DATA_DIR / "weekly_pre-dispatch_forecast.csv"
METRICS_PATH = ROOT / "metrics.json"
HOLDOUT_PATH = DATA_DIR / "holdout_predictions.csv"

FEATURES = [
    "temperature",
    "humidity",
    "holiday",
    "school",
    "hour",
    "month",
    "day",
    "day_of_week",
]
TARGET = "demand"

# Time-based split: learn from 2015-2018, test on all of 2019 (before COVID-19 hit 2020 demand).
TRAIN_END = "2019-01-01"
TEST_END = "2020-01-01"


def load_clean() -> pd.DataFrame:
    return pd.read_csv(CLEAN_PATH, parse_dates=["datetime"])


def load_predispatch() -> pd.DataFrame:
    """Official weekly pre-dispatch forecast from the grid operator (the benchmark to beat)."""
    pf = pd.read_csv(PREDISPATCH_PATH, encoding="utf-8-sig")
    pf["datetime"] = pd.to_datetime(pf["datetime"], format="%m/%d/%Y %H:%M")
    return pf.drop_duplicates("datetime")


def make_model() -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)


def score(y_true, y_pred) -> dict:
    return {
        "r2": round(float(r2_score(y_true, y_pred)), 3),
        "mae_mw": round(float(mean_absolute_error(y_true, y_pred)), 1),
        "rmse_mw": round(float(root_mean_squared_error(y_true, y_pred)), 1),
        "mape_pct": round(float(mean_absolute_percentage_error(y_true, y_pred)) * 100, 2),
    }


def evaluate_holdout(df: pd.DataFrame | None = None):
    """Train on data before TRAIN_END, test on the following year.

    Returns (model, test_frame, metrics). test_frame has the columns
    `rf_forecast` and `load_forecast` (the operator's pre-dispatch forecast). Both are
    scored on exactly the same rows.
    """
    df = load_clean() if df is None else df
    train = df[df["datetime"] < TRAIN_END]
    test = df[(df["datetime"] >= TRAIN_END) & (df["datetime"] < TEST_END)]
    test = test.merge(load_predispatch(), on="datetime", how="inner").copy()

    model = make_model().fit(train[FEATURES], train[TARGET])
    test["rf_forecast"] = model.predict(test[FEATURES])

    metrics = {
        "train_period": f"{train['datetime'].min():%Y-%m-%d} to {train['datetime'].max():%Y-%m-%d}",
        "test_period": f"{test['datetime'].min():%Y-%m-%d} to {test['datetime'].max():%Y-%m-%d}",
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "random_forest": score(test[TARGET], test["rf_forecast"]),
        "operator_pre_dispatch": score(test[TARGET], test["load_forecast"]),
    }
    # For reference only: a shuffled 80/20 split lets the model "peek" at the hours right next to
    # every test hour, which inflates the score. NOT a fair estimate of forecasting skill.
    X_tr, X_te, y_tr, y_te = train_test_split(df[FEATURES], df[TARGET], test_size=0.2, random_state=42)
    shuffled = make_model().fit(X_tr, y_tr)
    metrics["random_split_r2_leaky"] = score(y_te, shuffled.predict(X_te))["r2"]

    return model, test, metrics
