"""Raw Panama dataset -> data/cleaned_data.csv (model-ready columns + time features)."""
import pandas as pd

from common import CLEAN_PATH, RAW_PATH

KEEP = {
    "datetime": "datetime",
    "nat_demand": "demand",         # national hourly demand (MW)
    "T2M_toc": "temperature",       # 2 m air temperature, Tocumen (Panama City), deg C
    "QV2M_toc": "humidity",         # 2 m specific humidity, Tocumen, kg/kg
    "holiday": "holiday",           # 1 = public holiday
    "school": "school",             # 1 = school in session
}


def main() -> None:
    df = pd.read_csv(RAW_PATH, usecols=list(KEEP)).rename(columns=KEEP)
    df["datetime"] = pd.to_datetime(df["datetime"])

    missing = df.isnull().sum().sum()
    df = df.dropna().sort_values("datetime").reset_index(drop=True)

    # Sanity checks: no duplicate timestamps, no gaps in the hourly series.
    assert not df["datetime"].duplicated().any(), "duplicate timestamps"
    assert (df["datetime"].diff().dropna() == pd.Timedelta("1h")).all(), "gaps in hourly series"

    df["hour"] = df["datetime"].dt.hour
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["day_of_week"] = df["datetime"].dt.dayofweek  # Monday = 0

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Saved {len(df):,} rows ({missing} missing values dropped) -> {CLEAN_PATH}")
    print(f"Range: {df['datetime'].min()} to {df['datetime'].max()}")


if __name__ == "__main__":
    main()
