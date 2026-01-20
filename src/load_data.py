import pandas as pd

def load_processed_data(path: str) -> pd.DataFrame:
    """
    Load processed feature dataset for modeling.
    """
    df = pd.read_pickle(path)
    return df