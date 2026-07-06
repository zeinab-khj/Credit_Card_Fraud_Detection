import joblib
from pathlib import Path


def save_model(model, model_name="svm_pipeline.pkl"):

    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    model_path = model_dir / model_name

    joblib.dump(model, model_path)

    return model_path

def load_model(model_name="RF_pipeline.pkl"):

    model_path = Path("models") / model_name

    return joblib.load(model_path)
