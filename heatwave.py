"""What-if: how much does the model think demand rises when afternoons get hotter?

Takes every historical non-holiday weekday 15:00 observation, forces the temperature to a chosen
value (everything else unchanged), and averages the model's predictions.

Note: a Random Forest cannot extrapolate. The hottest temperature in the data is ~35 C, so
scenarios above that would just repeat the 35 C answer. Scenarios here stay inside the observed range.
"""
import pandas as pd

from common import FEATURES, TARGET, load_clean, make_model


def main() -> None:
    df = load_clean()
    model = make_model().fit(df[FEATURES], df[TARGET])

    base = df[(df["hour"] == 15) & (df["day_of_week"] < 5) & (df["holiday"] == 0)].copy()
    print(f"Observed temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f} C\n")

    rows = []
    for temp in [26, 28, 30, 32, 34]:
        scenario = base.copy()
        scenario["temperature"] = temp
        rows.append({"temperature_c": temp, "avg_predicted_demand_mw": model.predict(scenario[FEATURES]).mean()})

    out = pd.DataFrame(rows)
    out["change_vs_28c_mw"] = out["avg_predicted_demand_mw"] - out.loc[out["temperature_c"] == 28, "avg_predicted_demand_mw"].iloc[0]
    print(out.round(1).to_string(index=False))

    hot = out.loc[out["temperature_c"] == 34, "avg_predicted_demand_mw"].iloc[0]
    mild = out.loc[out["temperature_c"] == 28, "avg_predicted_demand_mw"].iloc[0]
    print(f"\nHeatwave (34 C) vs normal (28 C) at 15:00 on weekdays: {hot - mild:+.0f} MW ({(hot / mild - 1) * 100:+.1f}%)")


if __name__ == "__main__":
    main()
