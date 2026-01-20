from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def train_linear_regression():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression()),
    ])


def train_random_forest():
    return RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )


