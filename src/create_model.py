import json, pathlib, pickle, os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

SALES_PATH = os.path.join(os.path.dirname(__file__), "data", "kc_house_data.csv")
DEMOGRAPHICS_PATH = os.path.join(os.path.dirname(__file__), "data", "zipcode_demographics.csv")
SALES_COLUMN_SELECTION = ['price', 'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'sqft_above', 'sqft_basement', 'zipcode']
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "model")

def load_data():
    data = pd.read_csv(SALES_PATH, usecols=SALES_COLUMN_SELECTION, dtype={'zipcode': str})
    demographics = pd.read_csv(DEMOGRAPHICS_PATH, dtype={'zipcode': str})
    merged = data.merge(demographics, on="zipcode").drop(columns="zipcode")
    y = merged.pop('price')
    return merged, y

def train_model(x, y):
    x_train, _, y_train, _ = train_test_split(x, y, random_state=42)
    model = make_pipeline(RobustScaler(), KNeighborsRegressor()).fit(x_train, y_train)
    return model, x_train

def save_artifacts(model, features):
    pathlib.Path(OUTPUT_DIR).mkdir(exist_ok=True)
    pickle.dump(model, open(f"{OUTPUT_DIR}/model.pkl", 'wb'))
    json.dump(list(features.columns), open(f"{OUTPUT_DIR}/model_features.json", 'w'))

def main():
    x, y = load_data()
    model, x_train = train_model(x, y)
    save_artifacts(model, x_train)

if __name__ == "__main__":
    main()
