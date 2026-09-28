import requests
import json
import sys

def test_api():
    url = "https://data.mendeley.com/public-api/datasets/3mbnb82mxd/versions/1"
    headers = {"Accept": "application/json"}
    try:
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            if "files" in data:
                print(f"Found {len(data['files'])} files!")
                for i, f in enumerate(data['files'][:5]):
                    print(f"File {i}: {f.get('filename')} - {f.get('content_details', {}).get('download_url')}")
            else:
                print("No files key in JSON")
                print(json.dumps(data, indent=2)[:500])
        else:
            print(f"Failed: {r.status_code} - {r.text[:200]}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_api()
