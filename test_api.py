import requests
import json
import numpy as np

# URL of the Flask API
url = 'http://127.0.0.1:5001/predict'

# Generate random sample data
# Features: Time, Amount, V1-V28
data = {
    'Time': 1000,
    'Amount': 50.0
}
for i in range(1, 29):
    data[f'V{i}'] = np.random.randn()

# Send POST request
try:
    response = requests.post(url, json=data)
    
    print(f"Status Code: {response.status_code}")
    print("Response JSON:")
    print(json.dumps(response.json(), indent=4))
except Exception as e:
    print(f"Request failed: {e}")
