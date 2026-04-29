import requests
import argparse
import sys

def test_service(file_path):
    url = "http://localhost:8000/diarize"
    
    print(f"Testing service with file: {file_path}")
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
        
        if response.status_code == 200:
            result = response.json()
            print("\nSuccess! Diarization Results:")
            for seg in result['segments']:
                print(f"[{seg['start']:7.2f}s - {seg['end']:7.2f}s] {seg['speaker']}")
        else:
            print(f"\nError: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("\nError: Could not connect to the service. Is app.py running?")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Path to audio file to test")
    args = parser.parse_args()
    test_service(args.file)
