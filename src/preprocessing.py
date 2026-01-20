"""
Data preprocessing module for time series demand forecasting.

This module contains functions for cleaning, transforming, and preparing
raw time series data for analysis and modeling.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw time series data from CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing the loaded data
    """
    df = pd.read_csv(filepath)
    return df


def handle_missing_values(df: pd.DataFrame, method: str = 'forward_fill') -> pd.DataFrame:
    """
    Handle missing values in the time series data.
    
    Args:
        df: Input DataFrame
        method: Method to handle missing values ('forward_fill', 'interpolate', 'drop')
        
    Returns:
        DataFrame with missing values handled
    """
    if method == 'forward_fill':
        return df.fillna(method='ffill').fillna(method='bfill')
    elif method == 'interpolate':
        return df.interpolate(method='linear')
    elif method == 'drop':
        return df.dropna()
    else:
        raise ValueError(f"Unknown method: {method}")


def remove_outliers(df: pd.DataFrame, column: str, method: str = 'iqr') -> pd.DataFrame:
    """
    Remove outliers from time series data.
    
    Args:
        df: Input DataFrame
        column: Column name to check for outliers
        method: Method to detect outliers ('iqr' or 'zscore')
        
    Returns:
        DataFrame with outliers removed
    """
    if method == 'iqr':
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        return df[(df[column] >= Q1 - 1.5 * IQR) & (df[column] <= Q3 + 1.5 * IQR)]
    elif method == 'zscore':
        from scipy import stats
        z_scores = np.abs(stats.zscore(df[column]))
        return df[z_scores < 3]
    else:
        raise ValueError(f"Unknown method: {method}")


def normalize_data(df: pd.DataFrame, columns: list) -> Tuple[pd.DataFrame, dict]:
    """
    Normalize specified columns in the DataFrame.
    
    Args:
        df: Input DataFrame
        columns: List of column names to normalize
        
    Returns:
        Normalized DataFrame and scaling parameters for inverse transformation
    """
    scaling_params = {}
    df_normalized = df.copy()
    
    for col in columns:
        min_val = df[col].min()
        max_val = df[col].max()
        df_normalized[col] = (df[col] - min_val) / (max_val - min_val)
        scaling_params[col] = {'min': min_val, 'max': max_val}
    
    return df_normalized, scaling_params


def split_train_test(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets.
    
    Args:
        df: Input DataFrame
        test_size: Proportion of data to use for testing (0-1)
        
    Returns:
        Train and test DataFrames
    """
    split_idx = int(len(df) * (1 - test_size))
    return df[:split_idx], df[split_idx:]
