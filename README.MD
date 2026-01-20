# Demand Forecasting with Machine Learning

## Project Overview
This project focuses on **time series demand forecasting** using historical sales data from the *Store Item Demand Forecasting Challenge*.  
The objective is to predict daily item-level sales while applying **sound data science practices**, including proper time-based validation, strong baselines, and interpretable machine learning models.

The project is designed as a **portfolio-ready, end-to-end pipeline**, covering data exploration, feature engineering, modeling, evaluation, and production-style code structure.

---

## Dataset
**Source:** Kaggle – Store Item Demand Forecasting Challenge  
**Granularity:** Daily sales per store and item  
**Target variable:** `sales`

Each observation represents:
- a specific **date**
- a **store**
- an **item**
- the number of units sold

---

## Project Structure

```text
demand-forecasting/
├── data/
│   ├── raw/                # Original dataset (ignored by git)
│   └── processed/          # Generated features and outputs (ignored by git)
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_ml_models.ipynb
│   └── 05_anomaly_detection.ipynb
│
├── models/
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── models.py
│   ├── evaluation.py
│   └── utils.py
│
├── main.py                 # End-to-end pipeline
├── requirements.txt
└── README.md
```
## Notebooks Description

01 – Exploratory Data Analysis
- Sales distribution analysis
- Trend and seasonality detection
- Store- and item-level demand patterns

02 – Feature Engineering
- Lag features (1, 7, 14, 28 days)
- Rolling statistics (mean and standard deviation)
- Calendar-based features (day of week, month, year)

03 – Baseline Models
- Naive forecast
- Seasonal naive (lag-7)
- Rolling mean baseline
- Performance comparison using MAE and RMSE

04 – Machine Learning Models
- Linear Regression with preprocessing pipeline
- Missing value imputation
- Time-based train/test split
- Model evaluation and comparison

05 – Anomaly Detection
- Statistical anomaly detection (Z-score)
- Isolation Forest for extreme demand events
- Interpretation of detected anomalies

## Modeling Approach

- Train/Test Split: strictly time-based (no data leakage)
- Baseline First: simple heuristics before machine learning
- Model Choice: interpretable regression models
- Missing Values: handled via imputation
- Evaluation Metrics: MAE and RMSE

This approach prioritizes robustness, interpretability, and reproducibility.

## Running the Project

1. Install dependencies:

pip install -r requirements.txt

2. Run the full pipeline

python main.py

This will:
- Load the raw dataset
- Generate features
- Train regression models
- Evaluate performance
- Save results to data/processed/

## Key Results
- Seasonal baselines significantly outperform naive forecasts
- Machine learning models further reduce prediction error
- Each item/store shows stable, learnable demand patterns
- Anomaly detection highlights extreme demand events

## Future Improvements
- Gradient boosting models (XGBoost, LightGBM)
- Hierarchical forecasting (store → item)
- Probabilistic forecasting
- Production monitoring and retraining strategies