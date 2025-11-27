import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Loads the dataset from the given filepath."""
    print(f"Loading data from {filepath}...")
    try:
        df = pd.read_csv(filepath)
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def preprocess_data(df):
    """Preprocesses the data: scaling and splitting."""
    print("Preprocessing data...")
    
    # RobustScaler is less prone to outliers
    rob_scaler = RobustScaler()

    df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
    df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

    df.drop(['Time','Amount'], axis=1, inplace=True)
    
    # Move scaled columns to the front for easier access (optional but good for inspection)
    scaled_amount = df['scaled_amount']
    scaled_time = df['scaled_time']
    
    df.drop(['scaled_amount', 'scaled_time'], axis=1, inplace=True)
    df.insert(0, 'scaled_amount', scaled_amount)
    df.insert(1, 'scaled_time', scaled_time)

    X = df.drop('Class', axis=1)
    y = df['Class']

    return X, y

def train_models(X_train, y_train):
    """Trains Logistic Regression, Random Forest, and XGBoost models."""
    print("Training models...")
    
    models = {}
    
    # Logistic Regression
    print("Training Logistic Regression...")
    lr = LogisticRegression(solver='liblinear')
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # Random Forest
    print("Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf
    
    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    return models

def evaluate_models(models, X_test, y_test):
    """Evaluates models and plots ROC curves."""
    print("Evaluating models...")
    
    plt.figure(figsize=(10, 8))
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print(classification_report(y_test, y_pred))
        print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.savefig('roc_curves.png')
    print("ROC Curves saved to roc_curves.png")

def plot_feature_importance(models, feature_names):
    """Plots feature importance for tree-based models."""
    print("Plotting feature importance...")
    
    for name, model in models.items():
        if name in ['Random Forest', 'XGBoost']:
            plt.figure(figsize=(12, 6))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.title(f'Feature Importances - {name}')
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.savefig(f'feature_importance_{name.replace(" ", "_")}.png')
            print(f"Feature importance for {name} saved.")

def main():
    # Filepath
    filepath = 'cleaned_creditcard.csv' # Assuming the file is in the same directory
    
    # Load Data
    df = load_data(filepath)
    if df is None:
        return

    # Preprocess
    X, y = preprocess_data(df)
    
    # Split Data
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE
    print("Applying SMOTE...")
    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
    print(f"Original dataset shape {y_train.value_counts()}")
    print(f"Resampled dataset shape {y_train_res.value_counts()}")
    
    # Train Models
    models = train_models(X_train_res, y_train_res)
    
    # Evaluate
    evaluate_models(models, X_test, y_test)
    
    # Feature Importance
    plot_feature_importance(models, X.columns)
    
    print("Fraud detection pipeline completed.")

if __name__ == "__main__":
    main()
