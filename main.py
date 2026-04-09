from fastapi import FastAPI
import os
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title="ML Model API")

# -------------------------
# 📁 PATH SETUP
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

# -------------------------
# 🤖 LOAD MODEL
# -------------------------
model = None

if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"✅ Model loaded from: {MODEL_PATH}")
else:
    print(f"❌ Model not found at: {MODEL_PATH}")


# -------------------------
# 📥 INPUT SCHEMA
# -------------------------
class InputData(BaseModel):
    brand: str
    model: str
    year: int
    km_driven: float
    fuel: str
    transmission: str
    owner: int
    location: str
    mileage: float
    engine: float
    max_power: float
    seats: int
    seller_type: str
# -------------------------
# 🏠 HOME ROUTE
# -------------------------
@app.get("/")
def home():
    return {"message": "ML Model API is running successfully"}


# -------------------------
# 🔮 PREDICTION ROUTE
# -------------------------
@app.post("/predict")
def predict(data: InputData):

    if model is None:
        return {"error": "Model not loaded. Train model first."}

    try:
        # 🔥 Convert to DataFrame (VERY IMPORTANT)
        input_df = pd.DataFrame([data.dict()])

        prediction = model.predict(input_df)

        return {
            "input": data.dict(),
            "predicted_price": float(prediction[0])
        }

    except Exception as e:
        return {"error": str(e)}