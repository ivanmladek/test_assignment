import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import pickle, json, pathlib, os

SALES_PATH = os.path.join(os.path.dirname(__file__), "data", "kc_house_data.csv")
DEMOGRAPHICS_PATH = os.path.join(os.path.dirname(__file__), "data", "zipcode_demographics.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

def load_data():
    sales_data = pd.read_csv(SALES_PATH, dtype={'zipcode': str})
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})
    return sales_data, demographics

def feature_engineer(data):
    data['date'] = pd.to_datetime(data['date'])
    data['sale_year'] = data['date'].dt.year
    data['sale_month'] = data['date'].dt.month
    return data.drop(columns=['date', 'id'])

def prepare_data(sales_data, demographics):
    sales_data = feature_engineer(sales_data)
    merged = pd.merge(sales_data, demographics, on='zipcode', how='left')
    y = merged.pop('price')
    X = merged.drop(columns=['zipcode']).apply(pd.to_numeric, errors='coerce').fillna(0)
    return X, y

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42).fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"--- Improved Model ---\nR2: {r2:.4f}\nMAE: ${mae:,.2f}")
    return r2, mae

def save_artifacts(model, features):
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)
    pickle.dump(model, open(f"{OUTPUT_DIR}/model_improved.pkl", 'wb'))
    json.dump(list(features.columns), open(f"{OUTPUT_DIR}/model_features_improved.json", 'w'))

def main():
    sales_data, demographics = load_data()
    X, y = prepare_data(sales_data, demographics)
    model, X_train, X_test, y_train, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test)
    save_artifacts(model, X_train)

if __name__ == "__main__":
    main()
