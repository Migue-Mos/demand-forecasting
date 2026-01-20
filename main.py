# main.py

import pandas as pd
from pathlib import Path

from src.load_data import load_processed_data
from src.features import select_features
from src.models import train_linear_regression, train_random_forest
from src.evaluation import evaluate_regression
from src.save_model import save_model


def main():
    """
    Main execution pipeline for demand forecasting project.
    """

    # -----------------------------
    # 1. Paths & configuration
    # -----------------------------
    DATA_DIR = Path("data/processed")
    TARGET = "sales"

    # -----------------------------
    # 2. Load data
    # -----------------------------
    df = load_processed_data(DATA_DIR / "demand_forecasting_features.pkl")

    print(f"Dataset loaded with shape: {df.shape}")

    # -----------------------------
    # 3. Feature selection
    # -----------------------------
    X, y = select_features(df, target=TARGET)

    # -----------------------------
    # 4. Temporal train-test split
    # -----------------------------
    split_date = "2017-01-01"

    X_train = X[df["date"] < split_date]
    y_train = y[df["date"] < split_date]

    X_test = X[df["date"] >= split_date]
    y_test = y[df["date"] >= split_date]

    print(f"Train size: {X_train.shape}")
    print(f"Test size: {X_test.shape}")

    # -----------------------------
    # 5. Train models
    # -----------------------------
    lr_model = train_linear_regression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)

    save_model(lr_model, "linear_regression")



    rf_model = train_random_forest()
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)

    save_model(rf_model, "random_forest")


    # -----------------------------
    # 6. Evaluate models
    # -----------------------------
    print("\nLinear Regression performance:")
    mae_lr, rmse_lr = evaluate_regression(y_test, y_pred_lr)
    print(f"Linear Regression MAE: {mae_lr:.2f}")
    print(f"Linear Regression RMSE: {rmse_lr:.2f}")

    print("\nRandom Forest performance:")
    mae_rf, rmse_rf = evaluate_regression(y_test, y_pred_rf)
    print(f"Random Forest MAE: {mae_rf:.2f}")
    print(f"Random Forest RMSE: {rmse_rf:.2f}")


if __name__ == "__main__":
    main()
