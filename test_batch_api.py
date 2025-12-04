"""
Test script for batch prediction API endpoints
"""
import requests
import json
import time
from config import Config

API_BASE_URL = f"http://127.0.0.1:{Config.API_PORT}"

def test_batch_simple():
    """Test simple model batch prediction"""
    print("\n=== Testing Simple Model Batch Prediction ===")
    
    # Create sample transactions
    transactions = [
        {"time": 0.0, "amount": 100.0},
        {"time": 100.0, "amount": 250.5},
        {"time": 200.0, "amount": 50.0},
        {"time": 300.0, "amount": 1000.0},
        {"time": 400.0, "amount": 25.0},
    ]
    
    payload = {"transactions": transactions}
    
    start = time.time()
    response = requests.post(f"{API_BASE_URL}/predict_simple_batch", json=payload)
    elapsed = (time.time() - start) * 1000
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Processed {result['total_transactions']} transactions")
        print(f"   Fraud detected: {result['fraud_count']}")
        print(f"   API processing time: {result['processing_time_ms']:.2f}ms")
        print(f"   Total time (including network): {elapsed:.2f}ms")
        print(f"\n   Sample predictions:")
        for i, pred in enumerate(result['predictions'][:3]):
            print(f"   Transaction {i+1}: {'FRAUD' if pred['is_fraud'] else 'LEGITIMATE'} (prob: {pred['probability']:.4f})")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def test_batch_full():
    """Test full model batch prediction"""
    print("\n=== Testing Full Model Batch Prediction ===")
    
    # Create sample transactions with V1-V28 features
    import numpy as np
    np.random.seed(42)
    
    transactions = []
    for i in range(5):
        txn = {
            "time": float(i * 100),
            "amount": float(100 + i * 50),
            **{f"v{j}": float(np.random.randn()) for j in range(1, 29)}
        }
        transactions.append(txn)
    
    payload = {"transactions": transactions}
    
    start = time.time()
    response = requests.post(f"{API_BASE_URL}/predict_batch", json=payload)
    elapsed = (time.time() - start) * 1000
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Processed {result['total_transactions']} transactions")
        print(f"   Fraud detected: {result['fraud_count']}")
        print(f"   API processing time: {result['processing_time_ms']:.2f}ms")
        print(f"   Total time (including network): {elapsed:.2f}ms")
        print(f"\n   Sample predictions:")
        for i, pred in enumerate(result['predictions'][:3]):
            print(f"   Transaction {i+1}: {'FRAUD' if pred['is_fraud'] else 'LEGITIMATE'} (prob: {pred['probability']:.4f})")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def test_large_batch():
    """Test performance with larger batch"""
    print("\n=== Testing Large Batch (100 transactions) ===")
    
    import numpy as np
    np.random.seed(42)
    
    transactions = []
    for i in range(100):
        txn = {
            "time": float(i * 10),
            "amount": float(np.random.exponential(100)),
            **{f"v{j}": float(np.random.randn()) for j in range(1, 29)}
        }
        transactions.append(txn)
    
    payload = {"transactions": transactions}
    
    start = time.time()
    response = requests.post(f"{API_BASE_URL}/predict_batch", json=payload)
    elapsed = (time.time() - start) * 1000
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Success! Processed {result['total_transactions']} transactions")
        print(f"   Fraud detected: {result['fraud_count']}")
        print(f"   API processing time: {result['processing_time_ms']:.2f}ms")
        print(f"   Total time (including network): {elapsed:.2f}ms")
        print(f"   Throughput: {result['total_transactions'] / (result['processing_time_ms'] / 1000):.0f} transactions/sec")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def test_health():
    """Test health endpoint"""
    print("\n=== Testing Health Endpoint ===")
    response = requests.get(f"{API_BASE_URL}/health")
    if response.status_code == 200:
        result = response.json()
        print(f"✅ API is healthy")
        print(f"   Models loaded: {result.get('models_loaded', [])}")
    else:
        print(f"❌ API not healthy: {response.status_code}")

if __name__ == "__main__":
    print("=" * 60)
    print("Fraud Detection API - Batch Prediction Tests")
    print("=" * 60)
    
    try:
        test_health()
        test_batch_simple()
        test_batch_full()
        test_large_batch()
        
        print("\n" + "=" * 60)
        print("All tests completed!")
        print("=" * 60)
        
    except requests.exceptions.ConnectionError:
        print(f"\n❌ Cannot connect to API at {API_BASE_URL}")
        print(f"   Please make sure the API is running:")
        print(f"   python api.py")
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
