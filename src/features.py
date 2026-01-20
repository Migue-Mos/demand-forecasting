FEATURE_COLUMNS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_14", "rolling_std_14",
    "rolling_mean_28", "rolling_std_28",
    "day_of_week", "week_of_year", "month", "year"
]

TARGET_COLUMN = "sales"

def select_features(df, target):
    X = df[FEATURE_COLUMNS]
    y = df[target]
    return X, y