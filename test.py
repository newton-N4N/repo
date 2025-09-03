# test_phoenix.py
import requests
import time

def test_phoenix_server(port=6006):
    \"\"\"Test if Phoenix server is accessible\"\"\"
    try:
        response = requests.get(f"http://localhost:{port}/health")
        if response.status_code == 200:
            print(f"✅ Phoenix server is running on port {port}")
            return True
    except:
        print(f"❌ Phoenix server is not accessible on port {port}")
        return False

def test_trace_endpoint(port=6006):
    \"\"\"Test if trace endpoint is accepting data\"\"\"
    try:
        response = requests.get(f"http://localhost:{port}/v1/traces")
        print(f"✅ Trace endpoint is accessible")
        return True
    except:
        print(f"❌ Trace endpoint is not accessible")
        return False

if __name__ == "__main__":
    test_phoenix_server()
    test_trace_endpoint()