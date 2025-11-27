import streamlit as st
import joblib
import pandas as pd
import numpy as np

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
    .main {
        background-color: #f5f5f5;
    }
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

# Load model and scalers
@st.cache_resource
def load_resources():
    try:
        model = joblib.load('random_forest_model.pkl')
        scaler_amount = joblib.load('scaler_amount.pkl')
        scaler_time = joblib.load('scaler_time.pkl')
        return model, scaler_amount, scaler_time
    except Exception as e:
        st.error(f"Error loading resources: {e}")
        return None, None, None

model, scaler_amount, scaler_time = load_resources()

# Sidebar
st.sidebar.title("Navigation")
st.sidebar.info("This application uses a Random Forest model to detect fraudulent credit card transactions.")

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
            features = []
            for i in range(28):
                val = st.number_input(f"V{i+1}", value=float(st.session_state['features'][i]), key=f"v{i+1}")
                features.append(val)

    if st.button("Analyze Transaction", type="primary"):
        if model and scaler_amount and scaler_time:
            # Prepare input data
            input_data = pd.DataFrame([features], columns=[f'V{i}' for i in range(1, 29)])
            
            # Scale Time and Amount
            scaled_amount = scaler_amount.transform([[amount_val]])
            scaled_time = scaler_time.transform([[time_val]])
            
            input_data['scaled_amount'] = scaled_amount.flatten()
            input_data['scaled_time'] = scaled_time.flatten()
            
            # Reorder
            cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
            final_input = input_data[cols]
            
            # Predict
            prediction = model.predict(final_input)[0]
            probability = model.predict_proba(final_input)[0][1]
            
            st.markdown("---")
            st.subheader("Analysis Result")
            
            if prediction == 1:
                st.error(f"🚨 FRAUD DETECTED! (Probability: {probability:.2%})")
                st.markdown("This transaction shows patterns consistent with fraudulent activity.")
            else:
                st.success(f"✅ Legitimate Transaction (Probability of Fraud: {probability:.2%})")
                st.markdown("This transaction appears to be safe.")
                
        else:
            st.error("Model or scalers not loaded. Please check the server logs.")

with tab2:
    st.header("Upload Transactions")
    st.markdown("Upload a CSV file containing transaction details. Required columns: `Time`, `Amount`, `V1`...`V28`.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", df.head())
            
            # Validation
            required_cols = ['Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing columns: {missing_cols}")
            else:
                if st.button("Analyze Batch"):
                    if model and scaler_amount and scaler_time:
                        # Preprocess
                        process_df = df.copy()
                        
                        # Scale
                        process_df['scaled_amount'] = scaler_amount.transform(process_df['Amount'].values.reshape(-1,1))
                        process_df['scaled_time'] = scaler_time.transform(process_df['Time'].values.reshape(-1,1))
                        
                        # Select and reorder for model
                        cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'scaled_time']
                        X = process_df[cols]
                        
                        # Predict
                        predictions = model.predict(X)
                        probabilities = model.predict_proba(X)[:, 1]
                        
                        # Add results to original df
                        df['Fraud_Prediction'] = predictions
                        df['Fraud_Probability'] = probabilities
                        df['Status'] = df['Fraud_Prediction'].apply(lambda x: 'Fraud' if x == 1 else 'Legitimate')
                        
                        st.success("Analysis Complete!")
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
                        col_m1, col_m2 = st.columns(2)
                        col_m1.metric("Total Transactions", len(df))
                        col_m2.metric("Fraudulent Transactions Detected", int(fraud_count))
                        
                    else:
                        st.error("Model resources not loaded.")
                        
        except Exception as e:
            st.error(f"Error processing file: {e}")
