"""Streamlit dashboard: hourly electricity demand analysis + forecasting (Panama national grid)."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))
from common import (  # noqa: E402
    FEATURES,
    HOLDOUT_PATH,
    METRICS_PATH,
    TARGET,
    TRAIN_END,
    evaluate_holdout,
    load_clean,
    make_model,
)

st.set_page_config(page_title="Energy Demand Forecasting", page_icon="⚡", layout="wide")

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ---------------------------------------------------------------- data + model (cached)
@st.cache_data
def get_data() -> pd.DataFrame:
    return load_clean()


@st.cache_data
def get_holdout():
    """Fair evaluation: train on 2015-2018, test on 2019, compared with the operator's forecast.

    Uses the files written by `python src/train_model.py` when present (fast start);
    otherwise recomputes them.
    """
    if METRICS_PATH.exists() and HOLDOUT_PATH.exists():
        return pd.read_csv(HOLDOUT_PATH, parse_dates=["datetime"]), json.loads(METRICS_PATH.read_text())
    _, test, metrics = evaluate_holdout(get_data())
    return test, metrics


@st.cache_resource
def get_final_model():
    """Model used by the interactive predictor: trained on all available data."""
    df = get_data()
    return make_model().fit(df[FEATURES], df[TARGET])


df = get_data()
test, metrics = get_holdout()
model = get_final_model()
rf, op = metrics["random_forest"], metrics["operator_pre_dispatch"]


def show(fig):
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------- sidebar
st.sidebar.title("⚡ Project Information")
st.sidebar.markdown(
    f"""
### Energy Demand Forecasting

**Team Members**

• Aaron Parampog (24MT7030)

• Aryan Puthran (24MT7034)

• Jay Suthar (24MT7045)

---

**Data:** hourly national demand, Panama (2015 – Jun 2020)

**Algorithm:** Random Forest Regressor

**Validation:** train {df['datetime'].dt.year.min()}–{int(TRAIN_END[:4]) - 1}, test on {TRAIN_END[:4]} (time-based split)

**Holdout R²:** {rf['r2']:.2f}  ·  **MAPE:** {rf['mape_pct']:.1f}%

**Objective:** forecast electricity demand from weather and calendar variables.
"""
)

# ---------------------------------------------------------------- header + KPIs
st.title("⚡ Energy Demand Forecasting")
st.markdown(
    "Explore hourly electricity demand patterns and forecast demand from weather and calendar factors. "
    "The model is validated on a full year it never saw (2019) and benchmarked against the grid operator's "
    "own weekly pre-dispatch forecast."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Average demand", f"{df['demand'].mean():,.0f} MW")
c2.metric("Peak demand", f"{df['demand'].max():,.0f} MW")
c3.metric("Holdout R² (2019)", f"{rf['r2']:.2f}", delta=f"operator: {op['r2']:.2f}", delta_color="off")
c4.metric("Holdout MAPE (2019)", f"{rf['mape_pct']:.1f}%", delta=f"operator: {op['mape_pct']:.1f}%", delta_color="off")

# ---------------------------------------------------------------- exploration charts
st.subheader("📈 Demand trend")
daily = df.set_index("datetime")["demand"].resample("D").mean().reset_index()
show(px.line(daily, x="datetime", y="demand", title="Daily average demand (MW)", labels={"datetime": "", "demand": "MW"}))

left, right = st.columns(2)

with left:
    st.subheader("🕒 Demand by hour")
    hourly = (
        df.assign(day_type=df["day_of_week"].map(lambda d: "Weekend" if d >= 5 else "Weekday"))
        .groupby(["day_type", "hour"], as_index=False)["demand"]
        .mean()
    )
    show(px.line(hourly, x="hour", y="demand", color="day_type", markers=True, title="Average hourly profile"))

with right:
    st.subheader("🌡 Temperature vs demand")
    show(
        px.scatter(
            df.sample(3000, random_state=42),
            x="temperature",
            y="demand",
            opacity=0.35,
            title="Each dot = one hour (random sample)",
        )
    )

left, right = st.columns(2)

with left:
    st.subheader("📅 Demand by month")
    monthly = df.groupby("month", as_index=False)["demand"].mean()
    monthly["month_name"] = monthly["month"].map(lambda m: MONTHS[m - 1])
    show(px.bar(monthly, x="month_name", y="demand", title="Average demand by month"))

with right:
    st.subheader("📊 Demand distribution")
    show(px.histogram(df, x="demand", nbins=40, title="Distribution of hourly demand"))

# ---------------------------------------------------------------- model
st.subheader("🎯 What drives the forecast?")
importance = (
    pd.DataFrame({"Feature": FEATURES, "Importance": model.feature_importances_})
    .sort_values("Importance", ascending=False)
)
show(px.bar(importance, x="Feature", y="Importance", title="Random Forest feature importance"))

st.subheader("🔍 Forecast vs actual (held-out 2019)")
lo, hi = test["datetime"].min().date(), test["datetime"].max().date() - timedelta(days=6)
start = st.date_input("Week starting", value=date(2019, 4, 13), min_value=lo, max_value=hi)
week = test[(test["datetime"] >= pd.Timestamp(start)) & (test["datetime"] < pd.Timestamp(start) + pd.Timedelta(days=7))]
week = week.rename(columns={"demand": "Actual", "rf_forecast": "Random Forest", "load_forecast": "Operator pre-dispatch"})
show(
    px.line(
        week,
        x="datetime",
        y=["Actual", "Random Forest", "Operator pre-dispatch"],
        labels={"datetime": "", "value": "MW", "variable": ""},
        title="Model forecast vs actual demand vs operator forecast",
    )
)

# ---------------------------------------------------------------- interactive predictor
st.subheader("🔮 Try a scenario")
t_min, t_max = float(df["temperature"].min()), float(df["temperature"].max())
h_min, h_max = float(df["humidity"].min()), float(df["humidity"].max())

col_a, col_b = st.columns(2)
with col_a:
    when = st.date_input("Date", value=date(2019, 4, 15), key="scenario_date")
    hour = st.slider("Hour of day", 0, 23, 15)
    holiday = st.checkbox("Public holiday", value=False)
    school = st.checkbox("School in session", value=True)
with col_b:
    temp = st.slider("Temperature (°C)", round(t_min, 1), round(t_max, 1), 29.0, step=0.5)
    humidity = st.slider(
        "Specific humidity (kg/kg)", round(h_min, 4), round(h_max, 4), round(float(df["humidity"].median()), 4), step=0.0005, format="%.4f"
    )

scenario = pd.DataFrame(
    [
        {
            "temperature": temp,
            "humidity": humidity,
            "holiday": int(holiday),
            "school": int(school),
            "hour": hour,
            "month": when.month,
            "day": when.day,
            "day_of_week": when.weekday(),
        }
    ]
)[FEATURES]

st.metric(f"Predicted demand · {DAYS[when.weekday()]} {hour:02d}:00", f"{model.predict(scenario)[0]:,.0f} MW")
st.caption(
    f"Sliders stay inside the range the model has seen ({t_min:.0f}–{t_max:.0f} °C): Random Forests can't extrapolate beyond their training data."
)

# ---------------------------------------------------------------- insights
bias = (test["rf_forecast"] - test["demand"]).mean()
st.subheader("📌 Key insights")
st.markdown(
    f"""
- **Time of day is the strongest driver** (~{importance.iloc[0]['Importance']:.0%} of the model's importance), then day of week. Weather matters, but less.
- **Honest validation matters.** A shuffled train/test split gives R² ≈ {metrics['random_split_r2_leaky']:.2f}, but that leaks neighbouring hours into training. Training on 2015–2018 and testing on 2019 gives R² = **{rf['r2']:.2f}** (MAPE {rf['mape_pct']:.1f}%).
- **The operator's forecast is still better** (R² {op['r2']:.2f}, MAPE {op['mape_pct']:.1f}%). It also knows recent demand; this model only sees weather and calendar.
- **The model under-forecasts 2019 by ~{abs(bias):.0f} MW on average**: demand grew year over year and tree models can't extrapolate a trend.

**Where this applies:** peak-load planning, grid infrastructure planning, utility demand forecasting, energy resource optimisation.
"""
)
