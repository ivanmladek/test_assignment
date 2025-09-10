import unittest
import pandas as pd
import pickle
import json
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error

class TestHousePriceModels(unittest.TestCase):
    """Test suite for house price prediction models"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests"""
        # Load models and features
        with open("../model/model.pkl", "rb") as f:
            cls.basic_model = pickle.load(f)
        with open("../model/model_improved.pkl", "rb") as f:
            cls.improved_model = pickle.load(f)

        with open("../model/model_features.json", "r") as f:
            cls.basic_features = json.load(f)
        with open("../model/model_features_improved.json", "r") as f:
            cls.improved_features = json.load(f)

        # Load test data
        cls.sales_data = pd.read_csv("data/kc_house_data.csv", dtype={'zipcode': str})
        cls.demographics = pd.read_csv("data/zipcode_demographics.csv", dtype={'zipcode': str})