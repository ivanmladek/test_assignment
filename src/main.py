from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import json
from typing import List, Optional

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models and demographics data at startup
try:
    with open("../model/model.pkl", "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    model = None

try:
    with open("../model/model_features.json", "r") as f:
        model_features = json.load(f)
except FileNotFoundError:
    model_features = None

try:
    with open("../model/model_improved.pkl", "rb") as f:
        improved_model = pickle.load(f)
except FileNotFoundError:
    improved_model = None

try:
    with open("../model/model_features_improved.json", "r") as f:
        improved_model_features = json.load(f)
except FileNotFoundError:
    improved_model_features = None

try:
    demographics = pd.read_csv("data/zipcode_demographics.csv", dtype={'zipcode': str})
except FileNotFoundError:
    demographics = None

class HouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: str
    lat: float
    long: float
    sqft_living15: int
    sqft_lot15: int
    sale_year: Optional[int] = None
    sale_month: Optional[int] = None

class BasicHouseFeatures(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    sqft_above: int
    sqft_basement: int
    zipcode: str

@app.on_event("startup")
async def startup_event():
    if model is None:
        raise RuntimeError("Model 'model.pkl' not found.")
    if model_features is None:
        raise RuntimeError("Model features 'model_features.json' not found.")
    if improved_model is None:
        raise RuntimeError("Improved model 'model_improved.pkl' not found.")
    if improved_model_features is None:
        raise RuntimeError("Improved model features 'model_features_improved.json' not found.")
    if demographics is None:
        raise RuntimeError("Demographics data 'zipcode_demographics.csv' not found.")

@app.post("/predict")
def predict(features: HouseFeatures):
    print("Predict endpoint called")
    print(f"Input zipcode: {features.zipcode}")
    print(f"Demographics zipcodes sample: {demographics['zipcode'].head().tolist()}")
    print(f"98118 in demographics: {'98118' in demographics['zipcode'].values}")
    print(f"98042 in demographics: {'98042' in demographics['zipcode'].values}")
    # Convert input to DataFrame
    input_dict = features.dict()
    # Set defaults for sale_year and sale_month if not provided
    if input_dict['sale_year'] is None:
        input_dict['sale_year'] = 2023  # Default year
    if input_dict['sale_month'] is None:
        input_dict['sale_month'] = 6  # Default month
    input_df = pd.DataFrame([input_dict])
    print(f"Input df: {input_df}")
    print(f"Input df zipcode dtype: {input_df['zipcode'].dtype}")
    print(f"Demographics zipcode dtype: {demographics['zipcode'].dtype}")

    # Merge with demographics data
    merged_df = pd.merge(input_df, demographics, on='zipcode', how='left')
    print(f"Merged df shape: {merged_df.shape}")
    print(f"Merged df nulls: {merged_df.isnull().sum()}")
    print(f"Merged df: {merged_df}")

    # Check if zipcode was found (only check demographics columns for nulls)
    demographics_cols = [col for col in merged_df.columns if col != 'sale_year' and col != 'sale_month']
    if merged_df[demographics_cols].isnull().values.any():
        print(f"Debug: merged_df has null values in demographics columns")
        print(f"Debug: input_df columns: {input_df.columns.tolist()}")
        print(f"Debug: demographics columns: {demographics.columns.tolist()}")
        raise HTTPException(status_code=404, detail=f"Demographics not found for zipcode {features.zipcode}")

    # Drop zipcode and select/reorder features
    merged_df = merged_df.drop(columns=['zipcode'])

    try:
        final_features = merged_df[model_features]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in input or demographics data: {e}")

    # Make prediction
    prediction = model.predict(final_features)

    return {"prediction": prediction[0]}

@app.post("/predict_basic")
def predict_basic(features: BasicHouseFeatures):
    # Convert input to DataFrame
    input_df = pd.DataFrame([features.dict()])

    # Merge with demographics data
    merged_df = pd.merge(input_df, demographics, on='zipcode', how='left')

    # Check if zipcode was found
    if merged_df.isnull().values.any():
        raise HTTPException(status_code=404, detail=f"Demographics not found for zipcode {features.zipcode}")

    # Drop zipcode and select/reorder features
    merged_df = merged_df.drop(columns=['zipcode'])

    try:
        final_features = merged_df[model_features]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in input or demographics data: {e}")

    # Make prediction using basic model
    prediction = model.predict(final_features)

    return {"prediction": prediction[0], "model": "basic"}

@app.post("/predict_improved")
def predict_improved(features: HouseFeatures):
    # Convert input to DataFrame
    input_dict = features.dict()
    # Set defaults for sale_year and sale_month if not provided
    if input_dict['sale_year'] is None:
        input_dict['sale_year'] = 2023  # Default year
    if input_dict['sale_month'] is None:
        input_dict['sale_month'] = 6  # Default month
    input_df = pd.DataFrame([input_dict])

    # Merge with demographics data
    merged_df = pd.merge(input_df, demographics, on='zipcode', how='left')

    # Check if zipcode was found
    if merged_df.isnull().values.any():
        raise HTTPException(status_code=404, detail=f"Demographics not found for zipcode {features.zipcode}")

    # Drop zipcode and select/reorder features
    merged_df = merged_df.drop(columns=['zipcode'])

    try:
        final_features = merged_df[improved_model_features]
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing feature in input or demographics data: {e}")

    # Make prediction using improved model
    prediction = improved_model.predict(final_features)

    return {"prediction": prediction[0], "model": "improved"}

@app.get("/")
def read_root():
    return {"message": "Welcome to the House Price Predictor API"}