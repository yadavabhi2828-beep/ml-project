import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve
from imblearn.over_sampling import SMOTE
import os
import joblib
from config import Config, setup_logging

# Setup logging
logger = setup_logging(__name__)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    logger.warning("XGBoost not installed or failed to import. Skipping XGBoost model.")
except Exception as e:
    XGBOOST_AVAILABLE = False
    logger.warning(f"XGBoost failed to import: {e}. Skipping XGBoost model.")

# 1. Data Loading
def load_data(filepath=Config.DATA_PATH):
    if os.path.exists(filepath):
        logger.info(f"Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
    else:
        logger.warning("Dataset not found. Generating synthetic data for demonstration...")
        # Generate synthetic data mimicking the structure
        n_samples = 10000
        n_features = 30 # V1-V28 + Time + Amount
        
        # Create random features
        X = np.random.randn(n_samples, n_features)
        
        # Create imbalanced target
        y = np.zeros(n_samples)
        n_fraud = int(n_samples * 0.002) # 0.2% fraud
        fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
        y[fraud_indices] = 1
        
        columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
        
        # Adjust Time and Amount to look somewhat realistic
        X[:, 0] = np.cumsum(np.random.exponential(10, n_samples)) # Time
        X[:, -1] = np.random.exponential(100, n_samples) # Amount
        
        data = np.column_stack((X, y))
        df = pd.DataFrame(data, columns=columns)
        
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Class distribution:\n{df['Class'].value_counts(normalize=True)}")
    return df

# 2. Preprocessing
def preprocess_data(df):
    logger.info("Preprocessing data...")
    
    # Handle missing values (simple imputation if any)
    if df.isnull().sum().sum() > 0:
        logger.info("Handling missing values...")
        df = df.fillna(df.mean())

    # Normalize Time and Amount
    # RobustScaler is less prone to outliers
    rob_scaler_amount = RobustScaler()
    rob_scaler_time = RobustScaler()
    
    df['scaled_amount'] = rob_scaler_amount.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler_time.fit_transform(df['Time'].values.reshape(-1,1))
    
    df.drop(['Time', 'Amount'], axis=1, inplace=True)
    
    # Move scaled columns to front for easier slicing if needed, or just keep as is
    # Let's just use all columns except Class as features
    
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    return X, y, rob_scaler_amount, rob_scaler_time

# 3. Splitting and Balancing
def split_and_balance(X, y):
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y)
    
    logger.info("Applying SMOTE to training set...")
    sm = SMOTE(random_state=Config.SMOTE_RANDOM_STATE)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    logger.info(f"Original train shape: {y_train.shape}, Fraud count: {sum(y_train)}")
    logger.info(f"Resampled train shape: {y_train_res.shape}, Fraud count: {sum(y_train_res)}")
    
    return X_train_res, X_test, y_train_res, y_test

# 4. Model Training and Evaluation
def train_evaluate_models(X_train, X_test, y_train, y_test):
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    results = {}
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Metrics
        logger.info(f"--- {name} Evaluation ---")
        logger.info(f"\n{classification_report(y_test, y_pred)}")
        auc = roc_auc_score(y_test, y_pred_proba)
        logger.info(f"ROC AUC: {auc:.4f}")
        
        results[name] = {
            'model': model,
            'auc': auc,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Confusion Matrix:\n{cm}")

        # Save Random Forest model
        if name == 'Random Forest':
            joblib.dump(model, Config.FULL_MODEL_PATH)
            logger.info(f"Random Forest model saved to {Config.FULL_MODEL_PATH}")

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig('roc_curves.png')
    logger.info("ROC Curves saved to roc_curves.png")
    
    return results


# 5. Feature Importance
def plot_feature_importance(results, feature_names):
    logger.info("Plotting feature importances...")
    
    for name, result in results.items():
        model = result['model']
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 6))
            plt.title(f"Feature Importances - {name}")
            plt.bar(range(len(indices)), importances[indices], align="center")
            # feature_names is a pandas Index, so we can index it directly
            # Handle case where feature_names might be different for simple model
            current_feat_names = feature_names if len(feature_names) == len(importances) else ['scaled_amount', 'scaled_time']
            
            plt.xticks(range(len(indices)), current_feat_names[indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.replace(" ", "_")}.png')
            logger.info(f"Saved feature_importance_{name.replace(' ', '_')}.png")

def train_simple_model(X, y):
    logger.info("--- Training Simple Model (Time & Amount Only) ---")
    # Use only scaled_amount and scaled_time
    X_simple = X[['scaled_amount', 'scaled_time']]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42, stratify=y)
    
    # Balance
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    
    # Train Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_res, y_train_res)
    
    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    logger.info("Simple Model Evaluation:")
    logger.info(f"\n{classification_report(y_test, y_pred)}")
    auc = roc_auc_score(y_test, y_pred_proba)
    logger.info(f"Simple Model ROC AUC: {auc:.4f}")
    
    # Save
    joblib.dump(model, Config.SIMPLE_MODEL_PATH)
    logger.info(f"Simple Model saved to {Config.SIMPLE_MODEL_PATH}")
    
    return {'Simple Model': {'model': model, 'auc': auc, 'y_pred': y_pred, 'y_pred_proba': y_pred_proba}}

def main():
    df = load_data()
    X, y, scaler_amount, scaler_time = preprocess_data(df)
    
    # Save scalers
    joblib.dump(scaler_amount, Config.SCALER_AMOUNT_PATH)
    joblib.dump(scaler_time, Config.SCALER_TIME_PATH)
    logger.info(f"Scalers saved to {Config.SCALER_AMOUNT_PATH} and {Config.SCALER_TIME_PATH}")
    
    # 1. Train Full Models (V1-V28 + Time + Amount)
    logger.info("=== Training Full Models ===")
    X_train, X_test, y_train, y_test = split_and_balance(X, y)
    results = train_evaluate_models(X_train, X_test, y_train, y_test)
    plot_feature_importance(results, X.columns)
    
    # 2. Train Simple Model (Time + Amount Only)
    logger.info("=== Training Simple Model ===")
    simple_results = train_simple_model(X, y)
    # Plot importance for simple model (pass correct feature names)
    plot_feature_importance(simple_results, np.array(['scaled_amount', 'scaled_time']))

    logger.info("Training complete!")

if __name__ == "__main__":
    main()
