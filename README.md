# ⚡ Energy Demand Forecasting

Forecasting **hourly national electricity demand in Panama** from weather and calendar features with a Random Forest, wrapped in an interactive Streamlit dashboard, and benchmarked against the grid operator's own forecast.

![Forecast vs actual](assets/forecast_vs_actual.png)

## Results

Trained on 2015–2018, tested on **all of 2019** (8,760 hours the model never saw), scored on the same hours as the operator's weekly pre-dispatch forecast:

| Model | R² | MAE (MW) | RMSE (MW) | MAPE |
|---|---|---|---|---|
| Random Forest (weather + calendar) | 0.74 | 81.8 | 96.8 | 6.8% |
| Operator weekly pre-dispatch forecast | 0.85 | 54.9 | 73.6 | 4.8% |

**A note on validation.** A shuffled 80/20 split gives R² = 0.885, but that number is inflated: with hourly data, the hours right before and after every test hour end up in the training set. Splitting by time is the honest estimate, so that's what this project reports.

## Key findings

- **Time of day dominates** (~67% of feature importance), followed by day of week (~14%). Temperature and humidity matter, but far less.
- **Weekdays and weekends have very different profiles**: weekday demand peaks around midday, weekends stay flatter.
- **Hotter afternoons mean higher demand.** Holding everything else fixed on non-holiday weekdays at 15:00, the model predicts about **+100 MW (+7%)** at 34 °C versus 28 °C (`src/heatwave.py`). This is what the model learned from data, not a controlled experiment.

## Data

Hourly national demand, weather (Tocumen / Panama City) and holiday and school calendars, Jan 2015 – Jun 2020, plus the operator's weekly pre-dispatch forecast.

Aguilar Madrid, E. (2021). *Short-term electricity load forecasting (Panama case study)*. Mendeley Data. https://doi.org/10.17632/byx7sztj59.1
Check the dataset's licence on Mendeley before redistributing the raw files.

## Project structure

```
app.py                     Streamlit dashboard
src/common.py              Shared paths, features, model and evaluation
src/clean_data.py          Raw CSV -> data/cleaned_data.csv
src/train_model.py         Time-based evaluation -> metrics.json, data/holdout_predictions.csv
src/heatwave.py            Temperature what-if analysis
src/visualize.py           Saves README figures to assets/
data/                      Raw + cleaned data, pre-dispatch benchmark
```

## Run it

```bash
pip install -r requirements.txt      # Python 3.10+
python src/clean_data.py             # optional: data/cleaned_data.csv is already included
python src/train_model.py            # optional: metrics.json is already included
streamlit run app.py
```

## Limitations and next steps

- **The model under-forecasts 2019 by ~75 MW on average.** Demand grew year over year and Random Forests can't extrapolate a trend.
- **It can't extrapolate in temperature either**: nothing above ~35 °C (the hottest hour in the data) is predicted differently.
- **It only sees weather and calendar.** The operator's forecast is better, likely because it also knows recent demand. Adding lagged demand (e.g. the same hour last week) is the obvious next step.
- COVID-19 changed demand patterns in spring 2020, so the test year deliberately stops at Dec 2019.

## Team

Aaron Parampog · Aryan Puthran · Jay Suthar
