#!/usr/bin/env python3
"""
Comprehensive Test Suite for Fraud Detection System
Tests both API endpoints and validates responses
"""

import requests
import json
import numpy as np
import sys

API_BASE_URL = "http://127.0.0.1:5000"

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def test_full_model():
    """Test the full model endpoint with V1-V28 features"""
    print_section("Testing Full Model (/predict)")
    
    # Generate random test data
    data = {
        'Time': 1000.0,
        'Amount': 50.0
    }
    for i in range(1, 29):
        data[f'V{i}'] = float(np.random.randn())
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Full Model Test PASSED")
            print(f"   Prediction: {result['status']}")
            print(f"   Probability: {result['probability']:.4f}")
            print(f"   Raw Prediction: {result['prediction']}")
            return True
        else:
            print(f"❌ Full Model Test FAILED")
            print(f"   Error: {response.json()}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Flask API not running on port 5000")
        return False
    except Exception as e:
        print(f"❌ Test Failed: {str(e)}")
        return False

def test_simple_model():
    """Test the simple model endpoint with only Time and Amount"""
    print_section("Testing Simple Model (/predict_simple)")
    
    data = {
        'Time': 1000.0,
        'Amount': 50.0
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/predict_simple", json=data)
        
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Simple Model Test PASSED")
            print(f"   Prediction: {result['status']}")
            print(f"   Probability: {result['probability']:.4f}")
            print(f"   Raw Prediction: {result['prediction']}")
            return True
        else:
            print(f"❌ Simple Model Test FAILED")
            print(f"   Error: {response.json()}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Flask API not running on port 5000")
        return False
    except Exception as e:
        print(f"❌ Test Failed: {str(e)}")
        return False

def test_edge_cases():
    """Test edge cases and error handling"""
    print_section("Testing Edge Cases")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Missing required fields for full model
    print("\n1. Testing missing V features...")
    total_tests += 1
    try:
        response = requests.post(f"{API_BASE_URL}/predict", json={'Time': 100, 'Amount': 50})
        if response.status_code == 400:
            print("   ✅ Correctly rejected incomplete data")
            tests_passed += 1
        else:
            print(f"   ❌ Expected 400, got {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Missing required fields for simple model
    print("\n2. Testing missing Time/Amount for simple model...")
    total_tests += 1
    try:
        response = requests.post(f"{API_BASE_URL}/predict_simple", json={'Time': 100})
        if response.status_code == 400:
            print("   ✅ Correctly rejected incomplete data")
            tests_passed += 1
        else:
            print(f"   ❌ Expected 400, got {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Large transaction amount
    print("\n3. Testing large transaction amount...")
    total_tests += 1
    try:
        data = {'Time': 1000, 'Amount': 10000.0}
        for i in range(1, 29):
            data[f'V{i}'] = 0.0
        response = requests.post(f"{API_BASE_URL}/predict", json=data)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Processed large amount: {result['status']}")
            tests_passed += 1
        else:
            print(f"   ❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Zero amount
    print("\n4. Testing zero amount...")
    total_tests += 1
    try:
        response = requests.post(f"{API_BASE_URL}/predict_simple", json={'Time': 100, 'Amount': 0.0})
        if response.status_code == 200:
            result = response.json()
            print(f"   ✅ Processed zero amount: {result['status']}")
            tests_passed += 1
        else:
            print(f"   ❌ Failed with status {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print(f"\nEdge Cases: {tests_passed}/{total_tests} passed")
    return tests_passed == total_tests

def test_multiple_predictions():
    """Test multiple predictions to ensure consistency"""
    print_section("Testing Multiple Predictions")
    
    # Same input should give same output
    data = {
        'Time': 500.0,
        'Amount': 100.0
    }
    for i in range(1, 29):
        data[f'V{i}'] = 0.5
    
    try:
        results = []
        for i in range(3):
            response = requests.post(f"{API_BASE_URL}/predict", json=data)
            if response.status_code == 200:
                results.append(response.json())
        
        if len(results) == 3:
            # Check consistency
            if all(r['prediction'] == results[0]['prediction'] for r in results):
                print("✅ Predictions are consistent")
                print(f"   All predictions: {results[0]['status']}")
                return True
            else:
                print("❌ Predictions are inconsistent!")
                return False
        else:
            print("❌ Failed to get all predictions")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    """Run all tests"""
    print_section("FRAUD DETECTION SYSTEM - COMPREHENSIVE TEST SUITE")
    
    results = {
        'Full Model': test_full_model(),
        'Simple Model': test_simple_model(),
        'Edge Cases': test_edge_cases(),
        'Consistency': test_multiple_predictions()
    }
    
    print_section("TEST SUMMARY")
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:20s}: {status}")
    
    print(f"\nOverall: {passed}/{total} test suites passed")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! System is working correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please review the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
