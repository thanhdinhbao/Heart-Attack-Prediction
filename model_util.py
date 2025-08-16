from pathlib import Path
import joblib
import pandas as pd
from typing import Dict, List, Optional

MODELS_DIR = Path(__file__).parent / "models"
FEATURES = ["age","sex","cp","trestbps","chol","fbs","restecg",
            "thalach","exang","oldpeak","slope","ca","thal"]

_model_cache = {}

def list_models() -> List[str]:
    MODELS_DIR.mkdir(exist_ok=True)
    return sorted([p.name for p in MODELS_DIR.glob("*.pkl")])

def load_model(model_name: Optional[str] = None):
    """
    model_name: tên file trong thư mục models (vd: 'random_forest.pkl').
    Nếu None: lấy model đầu tiên theo thứ tự alphabet.
    """
    MODELS_DIR.mkdir(exist_ok=True)
    if model_name is None:
        all_models = list_models()
        if not all_models:
            raise FileNotFoundError("Thư mục models/ chưa có file .pkl")
        model_name = all_models[0]

    key = str(model_name)
    if key in _model_cache:
        return _model_cache[key], model_name

    path = MODELS_DIR / model_name
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {path}")
    model = joblib.load(path)
    _model_cache[key] = model
    return model, model_name

def to_dataframe_one(payload: Dict) -> pd.DataFrame:
    """Chuyển dict 1 bệnh nhân -> DataFrame đúng thứ tự cột FEATURES."""
    data = {k: [payload[k]] for k in FEATURES}
    return pd.DataFrame(data)

def to_dataframe_batch(items: List[Dict]) -> pd.DataFrame:
    """Chuyển list bệnh nhân -> DataFrame đúng thứ tự cột FEATURES."""
    rows = []
    for it in items:
        rows.append([it[k] for k in FEATURES])
    return pd.DataFrame(rows, columns=FEATURES)

def predict_one(model, x_df: pd.DataFrame, threshold: float = 0.5):
    if hasattr(model, "predict_proba"):
        p1 = float(model.predict_proba(x_df)[:,1][0])
    else:
        # fallback
        y = int(model.predict(x_df)[0])
        p1 = 0.9 if y==1 else 0.1
    label = int(p1 >= threshold)
    return label, p1
