import subprocess
import time
import requests
import sys
import os
import signal

def test_api():
    print("Starting API server...")
    # Start the API in a subprocess
    process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8000"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    try:
        # Wait for server to start
        print("Waiting for server to start...")
        for i in range(10):
            try:
                response = requests.get("http://127.0.0.1:8000/health")
                if response.status_code == 200:
                    print("Server is ready!")
                    break
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        else:
            print("Server failed to start within 10 seconds.")
            return False

        # Test Simple Model
        print("\nTesting /predict_simple...")
        simple_payload = {
            "time": 100.0,
            "amount": 50.0
        }
        response = requests.post("http://127.0.0.1:8000/predict_simple", json=simple_payload)
        if response.status_code == 200:
            data = response.json()
            print("Response:", data)
            if "prediction" in data and "probability" in data:
                print("✅ /predict_simple test passed")
            else:
                print("❌ /predict_simple test failed: Missing keys")
        else:
            print(f"❌ /predict_simple test failed: Status {response.status_code}")
            print(response.text)

        # Test Full Model
        print("\nTesting /predict...")
        full_payload = {
            "time": 100.0,
            "amount": 50.0,
            **{f"v{i}": 0.1 for i in range(1, 29)}
        }
        response = requests.post("http://127.0.0.1:8000/predict", json=full_payload)
        if response.status_code == 200:
            data = response.json()
            print("Response:", data)
            if "prediction" in data and "probability" in data:
                print("✅ /predict test passed")
            else:
                print("❌ /predict test failed: Missing keys")
        else:
            print(f"❌ /predict test failed: Status {response.status_code}")
            print(response.text)

    finally:
        print("\nStopping API server...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        print("Server stopped.")

if __name__ == "__main__":
    test_api()
