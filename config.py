"""
Configuration settings for the Fraud Detection System
"""
import os
import logging

class Config:
    """Application configuration"""
    
    # Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_DIR = BASE_DIR
    
    # Model files
    FULL_MODEL_PATH = os.path.join(MODEL_DIR, 'random_forest_model.pkl')
    SIMPLE_MODEL_PATH = os.path.join(MODEL_DIR, 'simple_model.pkl')
    SCALER_AMOUNT_PATH = os.path.join(MODEL_DIR, 'scaler_amount.pkl')
    SCALER_TIME_PATH = os.path.join(MODEL_DIR, 'scaler_time.pkl')
    ANOMALY_MODEL_PATH = os.path.join(MODEL_DIR, 'anomaly_model.pkl')
    
    # Data files
    DATA_PATH = os.path.join(BASE_DIR, 'cleaned_creditcard.csv')
    
    # API settings
    API_HOST = "0.0.0.0"
    API_PORT = 5001
    API_WORKERS = 1
    
    # Batch processing
    MAX_BATCH_SIZE = 1000
    BATCH_CHUNK_SIZE = 100  # Process in chunks for memory efficiency
    
    # Model training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    SMOTE_RANDOM_STATE = 42
    
    # Random Forest settings
    RF_N_ESTIMATORS = 100
    RF_N_JOBS = -1
    RF_RANDOM_STATE = 42
    
    # XGBoost settings
    XGB_RANDOM_STATE = 42
    XGB_USE_LABEL_ENCODER = False
    XGB_EVAL_METRIC = 'logloss'
    
    # Logging
    LOG_LEVEL = logging.INFO
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # Feature names
    V_FEATURES = [f'V{i}' for i in range(1, 29)]
    SCALED_FEATURES = ['scaled_amount', 'scaled_time']
    ALL_FEATURES = V_FEATURES + SCALED_FEATURES
    SIMPLE_FEATURES = SCALED_FEATURES

def setup_logging(name: str = __name__) -> logging.Logger:
    """
    Set up logging configuration
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=Config.LOG_LEVEL,
        format=Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT
    )
    logger = logging.getLogger(name)
    return logger
