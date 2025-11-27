from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load model and scalers
try:
    model = joblib.load('random_forest_model.pkl')
    scaler_amount = joblib.load('scaler_amount.pkl')
    scaler_time = joblib.load('scaler_time.pkl')
    print("Model and scalers loaded successfully.")
except Exception as e:
    print(f"Error loading model or scalers: {e}")
    model = None
    scaler_amount = None
    scaler_time = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler_amount or not scaler_time:
        return jsonify({'error': 'Model or scalers not loaded'}), 500

    try:
        data = request.get_json()
        
        # Create DataFrame from input
        input_df = pd.DataFrame([data])
        
        # Check if all required columns are present
        required_columns = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing_cols = [col for col in required_columns if col not in input_df.columns]
        if missing_cols:
             return jsonify({'error': f'Missing columns: {missing_cols}'}), 400

        # Preprocess
        input_df['scaled_amount'] = scaler_amount.transform(input_df['Amount'].values.reshape(-1,1))
        input_df['scaled_time'] = scaler_time.transform(input_df['Time'].values.reshape(-1,1))
        
        input_df.drop(['Time', 'Amount'], axis=1, inplace=True)
        
        # Ensure column order matches training (scaled_amount, scaled_time, V1...V28)
        # In fraud_train.py, we dropped Time and Amount, and added scaled_amount and scaled_time.
        # The order depends on where they were inserted or if they were appended.
        # "df['scaled_amount'] = ..." appends to the end.
        # So the order in X was: V1...V28, scaled_amount, scaled_time (since Time/Amount were dropped).
        # Wait, let's check fraud_train.py logic again.
        # df.drop(['Time', 'Amount'], axis=1, inplace=True)
        # X = df.drop('Class', axis=1)
        # So X columns are V1...V28, scaled_amount, scaled_time.
        
        # Let's reorder input_df to match
        cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
        input_df = input_df[cols]
        
        # Predict
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        
        result = {
            'prediction': int(prediction),
            'probability': float(probability),
            'status': 'Fraud' if prediction == 1 else 'Legitimate'
        }
        
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
