import requests, pandas as pd, json

URLS = {"predict": "http://127.0.0.1:8000/predict", "basic": "http://127.0.0.1:8000/predict_basic", "improved": "http://127.0.0.1:8000/predict_improved"}

def load_test_data():
    try:
        return pd.read_csv("data/future_unseen_examples.csv")
    except FileNotFoundError:
        print("Error: future_unseen_examples.csv not found.")
        exit()