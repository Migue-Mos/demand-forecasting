import pandas as pd


def temporal_train_val_split(df: pd.DataFrame, split_date: str):
    """
    Split dataset into train and validation sets using a date threshold.
    """
    train = df[df["date"] < split_date]
    val = df[df["date"] >= split_date]
    return train, val