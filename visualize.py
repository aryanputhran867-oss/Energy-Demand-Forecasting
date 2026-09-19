"""Save the README figures to assets/ (no blocking plt.show())."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from common import ROOT, evaluate_holdout, load_clean

ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def hourly_profile(df):
    df = df.assign(day_type=df["day_of_week"].map(lambda d: "Weekend" if d >= 5 else "Weekday"))
    prof = df.groupby(["day_type", "hour"])["demand"].mean().unstack(0)
    ax = prof.plot(figsize=(7, 4), lw=2.2, color={"Weekday": "#1f77b4", "Weekend": "#ff7f0e"})
    ax.set(xlabel="Hour of day", ylabel="Average demand (MW)", title="Demand follows the daily routine")
    ax.legend(title=None, frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(ASSETS / "hourly_profile.png")
    plt.close(ax.figure)


def temp_vs_demand(df):
    fig, ax = plt.subplots(figsize=(7, 4))
    hb = ax.hexbin(df["temperature"], df["demand"], gridsize=40, cmap="viridis", mincnt=1)
    fig.colorbar(hb, label="Hours observed")
    ax.set(xlabel="Temperature (°C)", ylabel="Demand (MW)", title="Hotter hours tend to be higher-demand hours")
    fig.tight_layout()
    fig.savefig(ASSETS / "temp_vs_demand.png")
    plt.close(fig)


def forecast_vs_actual(test, start="2019-04-13", days=7):
    end = pd.Timestamp(start) + pd.Timedelta(days=days)
    week = test[(test["datetime"] >= start) & (test["datetime"] < end)]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(week["datetime"], week["demand"], color="black", lw=2, label="Actual")
    ax.plot(week["datetime"], week["rf_forecast"], color="#d62728", lw=1.6, label="Random Forest")
    ax.plot(week["datetime"], week["load_forecast"], color="#2ca02c", lw=1.6, ls="--", label="Operator pre-dispatch")
    ax.set(ylabel="Demand (MW)", title=f"Week of {start}: actual vs forecasts (held-out 2019 data)")
    ax.legend(frameon=False, ncol=3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(ASSETS / "forecast_vs_actual.png")
    plt.close(fig)


def main() -> None:
    df = load_clean()
    _, test, _ = evaluate_holdout(df)
    hourly_profile(df)
    temp_vs_demand(df)
    forecast_vs_actual(test)
    print(f"Saved figures to {ASSETS}")


if __name__ == "__main__":
    main()
