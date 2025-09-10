import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle, json, os

def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "..", "model", "model.pkl")
    features_path = os.path.join(os.path.dirname(__file__), "..", "model", "model_features.json")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(features_path, "r") as f:
        model_features = json.load(f)
    return model, model_features

def load_data():
    sales_path = os.path.join(os.path.dirname(__file__), "data", "kc_house_data.csv")
    demographics_path = os.path.join(os.path.dirname(__file__), "data", "zipcode_demographics.csv")
    sales_data = pd.read_csv(sales_path, dtype={'zipcode': str})
    demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})
    return sales_data, demographics

def prepare_data(sales_data, demographics, model_features):
    merged = pd.merge(sales_data, demographics, on='zipcode', how='left')
    y = merged.pop('price')
    X = merged[model_features]
    return X, y

def evaluate(model, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    return r2, mae

def print_results(r2, mae):
    print(f"--- Model Evaluation ---\nR2: {r2:.4f}\nMAE: ${mae:,.2f}\nExplains {r2:.1%} variance, avg error ${mae:,.2f}")

def main():
    try:
        model, model_features = load_model()
        sales_data, demographics = load_data()
        X, y = prepare_data(sales_data, demographics, model_features)
        r2, mae = evaluate(model, X, y)
        print_results(r2, mae)
    except FileNotFoundError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
