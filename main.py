import pandas as pd
import joblib
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from schemas import ClaimData

# --------------------
# FastAPI App
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080"],  # frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------
# Load ML Model
# --------------------
MODEL_PATH = Path(
    r"C:\Users\AK\OneDrive\Desktop\FraudDetection\fraudradar-app\fraudradar-app-main\backend-app-main\model\trained_model.joblib"
)
print(">>> Loading model from:", MODEL_PATH)

try:
    model = joblib.load(MODEL_PATH)

    # ✅ Extract feature names from model (instead of hardcoding)
    if hasattr(model, "feature_names_in_"):
        feature_columns = model.feature_names_in_.tolist()
        print(">>> Loaded feature columns from model:", len(feature_columns))
    else:
        raise RuntimeError("❌ Model does not expose feature_names_in_ attribute. Save pipeline with preprocessing.")
except Exception as e:
    raise RuntimeError(f"❌ Failed to load model: {e}")

# --------------------
# Model Config
# --------------------
threshold = 0.5

# --------------------
# Endpoints
# --------------------
@app.get("/")
def root():
    return {"message": "✅ Fraud Detection API is running (no Kafka/DB)"}


# ---- Single prediction ----
@app.post("/predict/")
def predict_fraud(data: ClaimData):
    if len(data.features) != len(feature_columns):
        raise HTTPException(
            status_code=400,
            detail=f"Incorrect number of features. Expected {len(feature_columns)}, got {len(data.features)}",
        )

    X_input = pd.DataFrame([data.features], columns=feature_columns)
    prob = model.predict_proba(X_input)[0][1]
    prediction_int = int(prob >= threshold)
    prediction_str = "Fraudulent" if prediction_int == 1 else "Non-Fraudulent"
    risk_factors = ["Unusual billing patterns"] if prob > 0.8 else []

    return {
        "provider_id": data.provider_id,
        "prediction": prediction_str,
        "probability": round(prob, 4),
        "risk_factors": risk_factors,
    }


@app.post("/predict/batch/")
async def predict_batch(file: UploadFile = File(...)):
    import io, traceback

    try:
        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        print(">>> Uploaded CSV shape:", df.shape)
        print(">>> CSV columns:", list(df.columns))

        # ✅ Ensure same columns as model expects
        df = df[feature_columns]

        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= threshold).astype(int)

        results = []
        for i, prob in enumerate(probs):
            results.append({
                "provider_id": str(df.index[i]),
                "prediction": "Fraud" if preds[i] == 1 else "Not Fraud",
                "probability": float(prob),
                "threshold": threshold,
                "risk_factors": []
            })

        return {"predictions": results}

    except Exception as e:
        print(">>> ERROR in /predict/batch/:", e)
        traceback.print_exc()
        raise e
