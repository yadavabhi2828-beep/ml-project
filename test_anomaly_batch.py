import requests
import json
import random
import time

def test_batch_anomaly_prediction():
    url = "http://localhost:5001/predict_anomaly_batch"
    
    # Create a batch of random transactions
    batch_size = 10
    transactions = []
    
    for _ in range(batch_size):
        txn = {
            "Time": random.uniform(0, 10000),
            "Amount": random.uniform(0, 500),
            **{f"V{i}": random.gauss(0, 1) for i in range(1, 29)}
        }
        transactions.append(txn)
    
    payload = {"transactions": transactions}
    
    print(f"Sending batch of {batch_size} transactions...")
    
    try:
        start_time = time.time()
        response = requests.post(url, json=payload)
        duration = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch Anomaly Prediction Success! (Took {duration:.2f}ms)")
            print(f"Processed: {result['total_transactions']}")
            print(f"Anomalies Found: {result['fraud_count']}")
            print("First 2 predictions:")
            print(json.dumps(result['predictions'][:2], indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running?")

if __name__ == "__main__":
    test_batch_anomaly_prediction()
