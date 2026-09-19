"""Train + evaluate the Random Forest; writes metrics.json and data/holdout_predictions.csv
(pre-computed so the dashboard starts fast).

Evaluation is a *time-based* split (train 2015-2018, test 2019) and is compared against the
grid operator's own weekly pre-dispatch forecast on the same hours.
"""
import json

from common import FEATURES, HOLDOUT_PATH, METRICS_PATH, TARGET, evaluate_holdout, load_clean, make_model


def main() -> None:
    df = load_clean()
    _, test, metrics = evaluate_holdout(df)
    test[["datetime", "demand", "rf_forecast", "load_forecast"]].to_csv(HOLDOUT_PATH, index=False)

    # Final model on all data -> feature importance
    final = make_model().fit(df[FEATURES], df[TARGET])
    importance = dict(sorted(zip(FEATURES, final.feature_importances_.round(3).tolist()), key=lambda kv: -kv[1]))
    metrics["feature_importance"] = importance

    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
