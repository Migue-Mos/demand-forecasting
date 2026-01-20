from pathlib import Path
import joblib


def save_model(model, model_name: str, output_dir: Path = Path("models")):
    """
    Save a trained model to disk using joblib.

    Parameters
    ----------
    model : object
        Trained model or pipeline.
    model_name : str
        Name of the model (used for filename).
    output_dir : Path, optional
        Directory where models will be saved.
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{model_name}.joblib"
    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")
