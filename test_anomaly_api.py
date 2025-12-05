import requests
import json
import random

def test_anomaly_prediction():
    url = "http://localhost:5001/predict_anomaly"
    
    # Create a random transaction
    payload = {
        "Time": 100.0,
        "Amount": 50.0,
        **{f"V{i}": random.gauss(0, 1) for i in range(1, 29)}
    }
    
    try:
        response = requests.post(url, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Anomaly Prediction Success!")
            print(json.dumps(result, indent=2))
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API. Is it running?")

if __name__ == "__main__":
    test_anomaly_prediction()
