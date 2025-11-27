# Credit Card Fraud Detection

## Overview
This project implements a machine learning system to detect fraudulent credit card transactions. It uses a dataset of transactions, preprocesses the data (scaling, handling imbalance with SMOTE), and trains multiple models (Logistic Regression, Random Forest, XGBoost) to classify transactions as fraudulent or legitimate.

## Files
- **`fraud_train.py`**: The main script for training and evaluating models.
- **`app.py`**: Flask API for serving the model.
- **`streamlit_app.py`**: Streamlit web interface.
- **`test_api.py`**: Script to test the API.
- **`cleaned_creditcard.csv`**: The dataset used for training.
- **`requirements.txt`**: List of Python dependencies.
- **`project_files.json`**: JSON archive of the project source code.
- **`random_forest_model.pkl`**: The saved Random Forest model.
- **`scaler_amount.pkl`, `scaler_time.pkl`**: Saved scalers.
- **`roc_curves.png`**: ROC curves comparing model performance.
- **`feature_importance_*.png`**: Feature importance plots.

## Setup & Usage

### 1. Install Dependencies
Ensure you have Python installed. Install the required packages using pip:

```bash
pip install -r requirements.txt
```

*Note for macOS users: If you encounter issues with XGBoost, you may need to install `libomp`:*
```bash
brew install libomp
```

### 2. Run the Training Script
Execute the main script to train the models and generate results:

```bash
python3 fraud_train.py
```

### 3. Run the API
Start the Flask server:

```bash
python3 app.py
```

Send a test request:
```bash
python3 test_api.py
```

### 4. Run the Streamlit App
Start the interactive UI:

```bash
streamlit run streamlit_app.py
```

## Results
The system evaluates models using Precision, Recall, F1-score, and ROC-AUC.
- **Random Forest** and **XGBoost** typically achieve high ROC-AUC scores (>0.96).
- The trained Random Forest model is automatically saved as `random_forest_model.pkl`.
- The API provides real-time predictions for new transactions.
