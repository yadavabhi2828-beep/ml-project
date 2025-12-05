import streamlit as st
import pandas as pd
import numpy as np
import requests
from config import Config, setup_logging

# Setup logging
logger = setup_logging(__name__)

# Set page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .stSuccess {
        background-color: #d4edda;
        color: #155724;
        padding: 10px;
        border-radius: 5px;
    }
    .stError {
        background-color: #f8d7da;
        color: #721c24;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# API Configuration
API_BASE_URL = f"http://127.0.0.1:{Config.API_PORT}"

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.info("This application uses a Random Forest model to detect fraudulent credit card transactions.")

# Model Selection
st.sidebar.subheader("Model Settings")
model_choice = st.sidebar.radio(
    "Select Model",
    ["Full Model (Best Accuracy)", "Simple Model (Time & Amount Only)", "Anomaly Detection (Isolation Forest)"],
    help="Full Model requires V1-V28 features. Simple Model only needs Time and Amount. Anomaly Detection is unsupervised."
)

use_simple_model = "Simple Model" in model_choice
use_anomaly_model = "Anomaly Detection" in model_choice



# Main content
st.title("🛡️ Credit Card Fraud Detection")

tab1, tab2 = st.tabs(["Manual Entry", "Batch Upload (CSV)"])

with tab1:
    st.markdown("Enter transaction details below to check for potential fraud.")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")
        time_val = st.number_input("Time (Seconds since first transaction)", min_value=0.0, value=0.0)
        amount_val = st.number_input("Amount ($)", min_value=0.0, value=100.0)

    features = []
    if not use_simple_model:
        with col2:
            st.subheader("Anonymized Features (V1-V28)")
            st.caption("These features are PCA-transformed for privacy. You can generate random values for testing.")
            
            if st.button("Generate Random Features"):
                random_features = np.random.randn(28)
                st.session_state['features'] = random_features
            
            if 'features' not in st.session_state:
                st.session_state['features'] = np.zeros(28)
            
            # Display features in an expander to save space
            with st.expander("View/Edit Features V1-V28"):
                for i in range(28):
                    val = st.number_input(f"V{i+1}", value=float(st.session_state['features'][i]), key=f"v{i+1}")
                    features.append(val)
    else:
        with col2:
            st.info("Simple Model selected. V1-V28 features are not required.")

    if st.button("Analyze Transaction", type="primary"):
        try:
            if use_simple_model:
                # Call Simple Model API
                payload = {
                    'Time': time_val,
                    'Amount': amount_val
                }
                response = requests.post(f"{API_BASE_URL}/predict_simple", json=payload)
            elif use_anomaly_model:
                # Call Anomaly Model API
                payload = {
                    'Time': time_val,
                    'Amount': amount_val,
                    **{f'V{i}': features[i-1] for i in range(1, 29)}
                }
                response = requests.post(f"{API_BASE_URL}/predict_anomaly", json=payload)
            else:
                # Call Full Model API
                payload = {
                    'Time': time_val,
                    'Amount': amount_val,
                    **{f'V{i}': features[i-1] for i in range(1, 29)}
                }
                response = requests.post(f"{API_BASE_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['prediction']
                probability = result['probability']
                
                logger.info(f"Manual prediction: {'Fraud' if prediction == 1 else 'Legitimate'} (prob: {probability:.4f})")
                
                st.markdown("---")
                st.subheader("Analysis Result")
                
                if prediction == 1:
                    st.error(f"🚨 FRAUD DETECTED! (Probability: {probability:.2%})")
                    st.markdown("This transaction shows patterns consistent with fraudulent activity.")
                else:
                    st.success(f"✅ Legitimate Transaction (Probability of Fraud: {probability:.2%})")
                    st.markdown("This transaction appears to be safe.")
            else:
                st.error(f"API Error: {response.json().get('error', 'Unknown error')}")
                logger.error(f"API error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            st.error(f"❌ Cannot connect to API. Please make sure API is running on port {Config.API_PORT}.")
            logger.error("API connection error")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with tab2:
    st.header("Upload Transactions")
    if use_simple_model:
        st.markdown("Upload a CSV file. Required columns: `Time`, `Amount`.")
    elif use_anomaly_model:
        st.markdown("Upload a CSV file. Required columns: `Time`, `Amount`, `V1`...`V28`.")
    else:
        st.markdown("Upload a CSV file. Required columns: `Time`, `Amount`, `V1`...`V28`.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", df.head())
            
            # Validation
            if use_simple_model:
                required_cols = ['Time', 'Amount']
            else:
                # Both Full Model and Anomaly Model require all features
                required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
                
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                if st.button("Analyze Batch"):
                    try:
                        with st.spinner("Analyzing transactions..."):
                            # Prepare batch payload
                            transactions = []
                            for idx, row in df.iterrows():
                                if use_simple_model:
                                    txn = {
                                        'time': float(row['Time']),
                                        'amount': float(row['Amount'])
                                    }
                                else:
                                    txn = {
                                        'time': float(row['Time']),
                                        'amount': float(row['Amount']),
                                        **{f'v{i}': float(row[f'V{i}']) for i in range(1, 29)}
                                    }
                                transactions.append(txn)
                            
                            # Process in chunks if needed
                            all_predictions = []
                            all_probabilities = []
                            total_time = 0
                            
                            chunk_size = Config.BATCH_CHUNK_SIZE
                            for i in range(0, len(transactions), chunk_size):
                                chunk = transactions[i:i+chunk_size]
                                
                                # Call batch API
                                if use_simple_model:
                                    endpoint = f"{API_BASE_URL}/predict_simple_batch"
                                elif use_anomaly_model:
                                    endpoint = f"{API_BASE_URL}/predict_anomaly_batch"
                                else:
                                    endpoint = f"{API_BASE_URL}/predict_batch"
                                
                                payload = {'transactions': chunk}
                                response = requests.post(endpoint, json=payload)
                                
                                if response.status_code == 200:
                                    result = response.json()
                                    for pred_result in result['predictions']:
                                        all_predictions.append(pred_result['prediction'])
                                        all_probabilities.append(pred_result['probability'])
                                    total_time += result['processing_time_ms']
                                    
                                    # Show progress
                                    progress = min((i + chunk_size) / len(transactions), 1.0)
                                    st.progress(progress)
                                else:
                                    st.error(f"API Error: {response.json().get('detail', 'Unknown error')}")
                                    logger.error(f"Batch API error: {response.text}")
                                    break
                            
                            if len(all_predictions) == len(df):
                                # Add results to original df
                                df['Fraud_Prediction'] = all_predictions
                                df['Fraud_Probability'] = all_probabilities
                                df['Status'] = df['Fraud_Prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Legitimate')
                                
                                st.success(f"✅ Analysis Complete! Processed in {total_time:.2f}ms")
                                st.dataframe(df)
                                
                                # Download button
                                csv = df.to_csv(index=False).encode('utf-8')
                                st.download_button(
                                    label="Download Results as CSV",
                                    data=csv,
                                    file_name='fraud_predictions.csv',
                                    mime='text/csv',
                                )
                                
                                # Summary metrics
                                fraud_count = df['Fraud_Prediction'].sum()
                                col_m1, col_m2, col_m3 = st.columns(3)
                                col_m1.metric("Total Transactions", len(df))
                                col_m2.metric("Fraudulent Detected", int(fraud_count))
                                col_m3.metric("Processing Time", f"{total_time:.0f}ms")
                                
                                logger.info(f"Batch analysis complete: {len(df)} transactions, {fraud_count} frauds detected")
                        
                    except requests.exceptions.ConnectionError:
                        st.error(f"❌ Cannot connect to API. Please make sure API is running on port {Config.API_PORT}.")
                        logger.error("API connection error")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Batch processing error: {str(e)}")
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")
