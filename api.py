import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
import os

# Initialize App
app = FastAPI(
    title="Fraud Detection API",
    description="API for detecting credit card fraud using Random Forest models.",
    version="1.0.0"
)

# Global variables for models and scalers
models = {}

# Load resources on startup
@app.on_event("startup")
def load_resources():
    global models
    try:
        print("Loading models and scalers...")
        models['full_model'] = joblib.load('random_forest_model.pkl')
        models['scaler_amount'] = joblib.load('scaler_amount.pkl')
        models['scaler_time'] = joblib.load('scaler_time.pkl')
        
        if os.path.exists('simple_model.pkl'):
            models['simple_model'] = joblib.load('simple_model.pkl')
        else:
            models['simple_model'] = None
            print("Warning: simple_model.pkl not found.")
            
        print("Resources loaded successfully.")
    except Exception as e:
        print(f"Error loading resources: {e}")
        # In a real app, you might want to crash here if models are critical
        pass

# Input Schemas
class TransactionSimple(BaseModel):
    time: float
    amount: float

class TransactionFull(BaseModel):
    time: float
    amount: float
    v1: float
    v2: float
    v3: float
    v4: float
    v5: float
    v6: float
    v7: float
    v8: float
    v9: float
    v10: float
    v11: float
    v12: float
    v13: float
    v14: float
    v15: float
    v16: float
    v17: float
    v18: float
    v19: float
    v20: float
    v21: float
    v22: float
    v23: float
    v24: float
    v25: float
    v26: float
    v27: float
    v28: float

# Endpoints

@app.get("/")
def home():
    return {"message": "Welcome to the Fraud Detection API. Use /predict or /predict_simple to check transactions."}

@app.get("/health")
def health():
    if 'full_model' in models and models['full_model'] is not None:
        return {"status": "healthy", "models_loaded": list(models.keys())}
    else:
        raise HTTPException(status_code=503, detail="Models not loaded")

@app.post("/predict_simple")
def predict_simple(transaction: TransactionSimple):
    if not models.get('simple_model'):
        raise HTTPException(status_code=503, detail="Simple model not available.")
    
    try:
        # Preprocess
        scaler_amount = models['scaler_amount']
        scaler_time = models['scaler_time']
        
        scaled_amount = scaler_amount.transform([[transaction.amount]])
        scaled_time = scaler_time.transform([[transaction.time]])
        
        # Create DataFrame for prediction
        input_data = pd.DataFrame({
            'scaled_amount': scaled_amount.flatten(),
            'scaled_time': scaled_time.flatten()
        })
        
        # Predict
        model = models['simple_model']
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]
        
        return {
            "prediction": int(prediction),
            "is_fraud": bool(prediction == 1),
            "probability": float(probability),
            "model_used": "Simple Model (Time & Amount)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict")
def predict_full(transaction: TransactionFull):
    if not models.get('full_model'):
        raise HTTPException(status_code=503, detail="Full model not available.")
    
    try:
        # Preprocess
        scaler_amount = models['scaler_amount']
        scaler_time = models['scaler_time']
        
        scaled_amount = scaler_amount.transform([[transaction.amount]])
        scaled_time = scaler_time.transform([[transaction.time]])
        
        # Prepare features V1-V28
        # Extract v1..v28 from the input object
        features = [getattr(transaction, f'v{i}') for i in range(1, 29)]
        
        # Create DataFrame
        # Note: The model expects columns V1..V28, scaled_amount, scaled_time
        input_df = pd.DataFrame([features], columns=[f'V{i}' for i in range(1, 29)])
        input_df['scaled_amount'] = scaled_amount.flatten()
        input_df['scaled_time'] = scaled_time.flatten()
        
        # Reorder columns to match training
        cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
        final_input = input_df[cols]
        
        # Predict
        model = models['full_model']
        prediction = model.predict(final_input)[0]
        probability = model.predict_proba(final_input)[0][1]
        
        return {
            "prediction": int(prediction),
            "is_fraud": bool(prediction == 1),
            "probability": float(probability),
            "model_used": "Full Model (V1-V28 + Time + Amount)"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
