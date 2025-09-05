from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import logging

# ------------------------
# Logging setup
# ------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("uvicorn")

# ------------------------
# FastAPI app
# ------------------------
app = FastAPI(title="Iris Classifier API", version="1.0")

# ------------------------
# Input validation with Pydantic
# ------------------------
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., gt=0, description="Sepal length in cm")
    sepal_width: float = Field(..., gt=0, description="Sepal width in cm")
    petal_length: float = Field(..., gt=0, description="Petal length in cm")
    petal_width: float = Field(..., gt=0, description="Petal width in cm")

# ------------------------
# Load model
# ------------------------
model, class_names, model_accuracy = None, None, None

try:
    data = joblib.load("model.pkl")
    model = data.get("model")
    class_names = data.get("class_names")
    model_accuracy = data.get("accuracy")
    if model is None or class_names is None:
        raise ValueError("Model or class_names not found in model.pkl")
    logger.info("✅ Model loaded successfully")
except Exception as e:
    logger.error(f"❌ Could not load model.pkl: {e}")

# ------------------------
# Health check
# ------------------------
@app.get("/")
def health_check():
    if model is None:
        return {"status": "error", "message": "Model not loaded"}
    return {"status": "ok", "message": "Model loaded and ready"}

# ------------------------
# Predict single sample
# ------------------------
@app.post("/predict")
def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = np.array([[features.sepal_length, features.sepal_width,
                   features.petal_length, features.petal_width]])
    try:
        pred_idx = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
        return {
            "species": class_names[pred_idx],
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}")

# ------------------------
# Model info
# ------------------------
@app.get("/model-info")
def model_info():
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {
        "model": "Logistic Regression (with StandardScaler)",
        "dataset": "Iris (150 samples, 3 classes)",
        "classes": class_names,
        "accuracy": model_accuracy
    }
