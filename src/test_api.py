import requests, pandas as pd, json

URLS = {"predict": "http://127.0.0.1:8000/predict", "basic": "http://127.0.0.1:8000/predict_basic", "improved": "http://127.0.0.1:8000/predict_improved"}

def load_test_data():
    try:
        return pd.read_csv("data/future_unseen_examples.csv")
    except FileNotFoundError:
        print("Error: future_unseen_examples.csv not found.")
        exit()

def prepare_payload(row):
    payload = row.to_dict()
    for k in ['bedrooms', 'sqft_living', 'sqft_lot', 'waterfront', 'view', 'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated', 'sqft_living15', 'sqft_lot15']:
        payload[k] = int(payload[k])
    payload['zipcode'] = str(int(payload['zipcode']))
    return payload

def create_basic_payload(payload):
    return {k: v for k, v in payload.items() if k in ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 'zipcode']}

def test_endpoint(url, data):
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return f"Error: {e}"

def main():
    test_data = load_test_data()
    for i, row in test_data.head().iterrows():
        payload = prepare_payload(row)
        print(f"Payload: {json.dumps(payload, indent=2)}")

        for name, url in URLS.items():
            data = payload if name != "basic" else create_basic_payload(payload)
            result = test_endpoint(url, data)
            print(f"{name}: {result}")
        print("---")

if __name__ == "__main__":
    main()