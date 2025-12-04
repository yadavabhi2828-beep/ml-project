import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import os
from config import Config, setup_logging

# Setup logging
logger = setup_logging(__name__)

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
        logger.info("Loading models and scalers...")
        models['full_model'] = joblib.load(Config.FULL_MODEL_PATH)
        models['scaler_amount'] = joblib.load(Config.SCALER_AMOUNT_PATH)
        models['scaler_time'] = joblib.load(Config.SCALER_TIME_PATH)
        
        if os.path.exists(Config.SIMPLE_MODEL_PATH):
            models['simple_model'] = joblib.load(Config.SIMPLE_MODEL_PATH)
            logger.info("Simple model loaded successfully")
        else:
            models['simple_model'] = None
            logger.warning("simple_model.pkl not found")
            
        logger.info("Resources loaded successfully")
    except Exception as e:
        logger.error(f"Error loading resources: {e}")
        raise  # Fail fast if models can't be loaded

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

class BatchTransactionSimple(BaseModel):
    transactions: List[TransactionSimple] = Field(..., max_items=Config.MAX_BATCH_SIZE)
    
    @validator('transactions')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Batch must contain at least one transaction")
        return v

class BatchTransactionFull(BaseModel):
    transactions: List[TransactionFull] = Field(..., max_items=Config.MAX_BATCH_SIZE)
    
    @validator('transactions')
    def validate_batch_size(cls, v):
        if len(v) == 0:
            raise ValueError("Batch must contain at least one transaction")
        return v

class PredictionResponse(BaseModel):
    prediction: int
    is_fraud: bool
    probability: float
    model_used: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    total_transactions: int
    fraud_count: int
    processing_time_ms: float

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

@app.post("/predict_batch", response_model=BatchPredictionResponse)
def predict_full_batch(batch: BatchTransactionFull):
    """Batch prediction endpoint for full model (V1-V28 + Time + Amount)"""
    if not models.get('full_model'):
        raise HTTPException(status_code=503, detail="Full model not available")
    
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Processing batch of {len(batch.transactions)} transactions with full model")
        
        # Convert batch to DataFrame for vectorized processing
        data = []
        for txn in batch.transactions:
            row = {
                'Time': txn.time,
                'Amount': txn.amount,
                **{f'V{i}': getattr(txn, f'v{i}') for i in range(1, 29)}
            }
            data.append(row)
        
        df = pd.DataFrame(data)
        
        # Vectorized scaling
        scaler_amount = models['scaler_amount']
        scaler_time = models['scaler_time']
        
        scaled_amount = scaler_amount.transform(df[['Amount']])
        scaled_time = scaler_time.transform(df[['Time']])
        
        # Prepare features
        input_df = df[[f'V{i}' for i in range(1, 29)]].copy()
        input_df['scaled_amount'] = scaled_amount.flatten()
        input_df['scaled_time'] = scaled_time.flatten()
        
        # Reorder columns
        cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
        final_input = input_df[cols]
        
        # Batch prediction
        model = models['full_model']
        predictions = model.predict(final_input)
        probabilities = model.predict_proba(final_input)[:, 1]
        
        # Build response
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(PredictionResponse(
                prediction=int(pred),
                is_fraud=bool(pred == 1),
                probability=float(prob),
                model_used="Full Model (V1-V28 + Time + Amount)"
            ))
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        fraud_count = int(predictions.sum())
        
        logger.info(f"Batch processed in {processing_time:.2f}ms. Fraud detected: {fraud_count}/{len(batch.transactions)}")
        
        return BatchPredictionResponse(
            predictions=results,
            total_transactions=len(batch.transactions),
            fraud_count=fraud_count,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.post("/predict_simple_batch", response_model=BatchPredictionResponse)
def predict_simple_batch(batch: BatchTransactionSimple):
    """Batch prediction endpoint for simple model (Time + Amount only)"""
    if not models.get('simple_model'):
        raise HTTPException(status_code=503, detail="Simple model not available")
    
    try:
        import time
        start_time = time.time()
        
        logger.info(f"Processing batch of {len(batch.transactions)} transactions with simple model")
        
        # Convert batch to DataFrame
        data = [{'Time': txn.time, 'Amount': txn.amount} for txn in batch.transactions]
        df = pd.DataFrame(data)
        
        # Vectorized scaling
        scaler_amount = models['scaler_amount']
        scaler_time = models['scaler_time']
        
        scaled_amount = scaler_amount.transform(df[['Amount']])
        scaled_time = scaler_time.transform(df[['Time']])
        
        # Prepare input
        input_df = pd.DataFrame({
            'scaled_amount': scaled_amount.flatten(),
            'scaled_time': scaled_time.flatten()
        })
        
        # Batch prediction
        model = models['simple_model']
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)[:, 1]
        
        # Build response
        results = []
        for pred, prob in zip(predictions, probabilities):
            results.append(PredictionResponse(
                prediction=int(pred),
                is_fraud=bool(pred == 1),
                probability=float(prob),
                model_used="Simple Model (Time & Amount)"
            ))
        
        processing_time = (time.time() - start_time) * 1000
        fraud_count = int(predictions.sum())
        
        logger.info(f"Batch processed in {processing_time:.2f}ms. Fraud detected: {fraud_count}/{len(batch.transactions)}")
        
        return BatchPredictionResponse(
            predictions=results,
            total_transactions=len(batch.transactions),
            fraud_count=fraud_count,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host=Config.API_HOST, port=Config.API_PORT)
