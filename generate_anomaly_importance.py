import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split
from config import Config, setup_logging
from fraud_train import load_data, preprocess_data

# Setup logging
logger = setup_logging(__name__)

def anomaly_auc_scorer(estimator, X, y):
    """
    Custom scorer for Isolation Forest.
    IsolationForest decision_function returns:
    - Positive values for inliers (normal data)
    - Negative values for outliers (anomalies)
    
    We want to measure how well it detects frauds (y=1).
    So "fraud score" should be higher for frauds.
    We take the negative of decision_function so that:
    - Anomalies (originally negative) become positive (high fraud score)
    - Normals (originally positive) become negative (low fraud score)
    """
    # y is the true label (1 for fraud/anomaly, 0 for normal)
    # estimator.decision_function(X) returns anomaly score (higher = normal)
    # We negate it so higher = anomaly
    y_scores = -estimator.decision_function(X)
    return roc_auc_score(y, y_scores)

def main():
    logger.info("Starting Anomaly Model Feature Importance Analysis...")

    # 1. Load Data
    df = load_data()
    X, y, _, _ = preprocess_data(df)
    
    # 2. Load Model
    logger.info(f"Loading anomaly model from {Config.ANOMALY_MODEL_PATH}...")
    try:
        model = joblib.load(Config.ANOMALY_MODEL_PATH)
    except FileNotFoundError:
        logger.error("Anomaly model not found! Please run training first.")
        return

    # 3. Prepare Evaluation Subset
    # Permutation importance can be computationally expensive on the full dataset.
    # We'll use a representative subset (e.g., 20% or max 50k samples) to keep it fast.
    logger.info("Preparing evaluation subset...")
    
    # Use the same random state as training for consistency, or a new one appropriately.
    # We just need a good mix of normal and fraud data.
    # Since IF is often used for outlier detection, we want to see which features 
    # help distinguish the outliers (frauds) from the inliers.
    
    subset_size = min(50000, len(X))
    if len(X) > subset_size:
        _, X_eval, _, y_eval = train_test_split(
            X, y, test_size=subset_size, stratify=y, random_state=42
        )
    else:
        X_eval, y_eval = X, y
        
    logger.info(f"Using subset of {len(X_eval)} samples for permutation importance.")

    # 4. Compute Permutation Importance
    logger.info("Computing permutation importance (this may take a moment)...")
    # n_repeats=5 is usually sufficient for stable results
    result = permutation_importance(
        model, X_eval, y_eval,
        scoring=anomaly_auc_scorer,
        n_repeats=5,
        random_state=42,
        n_jobs=-1
    )

    # 5. Process and Plot Results
    sorted_idx = result.importances_mean.argsort()
    
    # Create DataFrame for easier viewing/saving
    importance_df = pd.DataFrame({
        'Feature': X.columns[sorted_idx],
        'Importance_Mean': result.importances_mean[sorted_idx],
        'Importance_Std': result.importances_std[sorted_idx]
    })
    
    # Print top 10 features
    logger.info("Top 10 Important Features for Anomaly Detection:")
    logger.info("\n" + str(importance_df.tail(10).iloc[::-1]))

    # Plot
    plt.figure(figsize=(12, 8))
    plt.boxplot(
        result.importances[sorted_idx].T,
        vert=False,
        labels=X.columns[sorted_idx]
    )
    plt.title("Permutation Importances (Test Set) - Isolation Forest")
    plt.xlabel("Decrease in ROC AUC Score")
    plt.tight_layout()
    
    output_plot = "feature_importance_anomaly.png"
    plt.savefig(output_plot)
    logger.info(f"Feature importance plot saved to {output_plot}")

if __name__ == "__main__":
    main()
