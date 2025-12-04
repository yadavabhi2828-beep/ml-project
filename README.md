# Credit Card Fraud Detection

## Overview
This project implements a machine learning system to detect fraudulent credit card transactions. It uses a dataset of transactions, preprocesses the data (scaling, handling imbalance with SMOTE), and trains multiple models (Logistic Regression, Random Forest, XGBoost) to classify transactions as fraudulent or legitimate.

## ✨ New Features
- **Batch Prediction API**: Process multiple transactions in a single request (10-100x faster)
- **Centralized Configuration**: All settings in `config.py` for easy management
- **Proper Logging**: Structured logging throughout the application
- **Optimized Performance**: Vectorized batch processing for improved throughput

## Files
- **`config.py`**: Centralized configuration for all settings
- **`fraud_train.py`**: Main script for training and evaluating models
- **`api.py`**: FastAPI server with batch prediction endpoints
- **`streamlit_app.py`**: Streamlit web interface with batch upload support
- **`test_batch_api.py`**: Test script for batch API endpoints
- **`test_api.py`**: Script to test individual predictions
- **`cleaned_creditcard.csv`**: Dataset used for training
- **`requirements.txt`**: Pinned Python dependencies
- **`random_forest_model.pkl`**: Saved Random Forest model
- **`simple_model.pkl`**: Saved simple model (Time + Amount only)
- **`scaler_amount.pkl`, `scaler_time.pkl`**: Saved scalers
- **`roc_curves.png`**: ROC curves comparing model performance
- **`feature_importance_*.png`**: Feature importance plots

## Setup & Usage

### 1. Install Dependencies
Ensure you have Python 3.8+ installed. Install the required packages:

```bash
pip install -r requirements.txt
```

*Note for macOS users: If you encounter issues with XGBoost, install `libomp`:*
```bash
brew install libomp
```

### 2. Train Models
Execute the training script to train models and generate results:

```bash
python3 fraud_train.py
```

This will:
- Load and preprocess the dataset
- Train multiple models (Logistic Regression, Random Forest, XGBoost)
- Save trained models and scalers
- Generate ROC curves and feature importance plots

### 3. Run the API
Start the FastAPI server:

```bash
python3 api.py
```

The API will be available at `http://localhost:8000`

**API Endpoints:**
- `GET /health` - Check API health and loaded models
- `POST /predict` - Single prediction (full model)
- `POST /predict_simple` - Single prediction (simple model)
- `POST /predict_batch` - **NEW!** Batch prediction (full model)
- `POST /predict_simple_batch` - **NEW!** Batch prediction (simple model)

### 4. Test the API
Test individual predictions:
```bash
python3 test_api.py
```

Test batch predictions:
```bash
python3 test_batch_api.py
```

### 5. Run the Streamlit App
Start the interactive web interface:

```bash
streamlit run streamlit_app.py
```

Features:
- Manual transaction entry with fraud prediction
- CSV batch upload with optimized batch processing
- Model selection (Full vs Simple)
- Download results as CSV
- Real-time performance metrics

## API Usage Examples

### Single Prediction
```python
import requests

# Simple model (Time + Amount only)
response = requests.post("http://localhost:8000/predict_simple", json={
    "time": 100.0,
    "amount": 250.5
})

# Full model (requires V1-V28 features)
response = requests.post("http://localhost:8000/predict", json={
    "time": 100.0,
    "amount": 250.5,
    "v1": -1.5, "v2": 0.3, # ... v3-v28
})
```

### Batch Prediction (NEW!)
```python
import requests

# Batch prediction - much faster for multiple transactions
response = requests.post("http://localhost:8000/predict_simple_batch", json={
    "transactions": [
        {"time": 0.0, "amount": 100.0},
        {"time": 100.0, "amount": 250.5},
        {"time": 200.0, "amount": 50.0}
    ]
})

result = response.json()
print(f"Processed {result['total_transactions']} in {result['processing_time_ms']:.2f}ms")
print(f"Fraud detected: {result['fraud_count']}")
```

## Performance Improvements

### Batch Processing
- **Before**: Individual API calls for each transaction (~50-100ms each)
- **After**: Batch processing with vectorized operations (~1-5ms per transaction)
- **Speedup**: 10-100x faster for large datasets

### Example Performance
- 100 transactions: ~50ms (2000 transactions/sec)
- 1000 transactions: ~300ms (3300 transactions/sec)

## Configuration

All settings are centralized in `config.py`:
- Model file paths
- API host and port
- Batch processing limits
- Logging configuration
- Training parameters

## Results
The system evaluates models using Precision, Recall, F1-score, and ROC-AUC:
- **Random Forest** and **XGBoost** typically achieve ROC-AUC > 0.96
- **Simple Model** (Time + Amount only) achieves ROC-AUC > 0.85
- Models are automatically saved for deployment
- Comprehensive logging for monitoring and debugging

## Logging
All components use structured logging:
- Training progress and metrics
- API requests and responses
- Batch processing performance
- Error tracking and debugging

Logs are output to console with timestamps and log levels.

## Next Steps
See `project_improvements.md` for additional enhancement recommendations including:
- Model compression and optimization
- Real-time fraud detection
- Enhanced monitoring and alerting
- Docker deployment
- Advanced ML techniques
