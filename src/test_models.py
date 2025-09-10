import unittest
import pandas as pd
import pickle
import json
import numpy as np
import os
from sklearn.metrics import r2_score, mean_absolute_error

class TestHousePriceModels(unittest.TestCase):
    """Test suite for house price prediction models"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests"""
        # Load models and features
        script_dir = os.path.dirname(__file__)
        basic_model_path = os.path.join(script_dir, "..", "model", "model.pkl")
        improved_model_path = os.path.join(script_dir, "..", "model", "model_improved.pkl")
        basic_features_path = os.path.join(script_dir, "..", "model", "model_features.json")
        improved_features_path = os.path.join(script_dir, "..", "model", "model_features_improved.json")

        try:
            with open(basic_model_path, "rb") as f:
                cls.basic_model = pickle.load(f)
        except FileNotFoundError:
            cls.basic_model = None

        try:
            with open(improved_model_path, "rb") as f:
                cls.improved_model = pickle.load(f)
        except FileNotFoundError:
            cls.improved_model = None

        try:
            with open(basic_features_path, "r") as f:
                cls.basic_features = json.load(f)
        except FileNotFoundError:
            cls.basic_features = None

        try:
            with open(improved_features_path, "r") as f:
                cls.improved_features = json.load(f)
        except FileNotFoundError:
            cls.improved_features = None

        # Load test data
        sales_path = os.path.join(os.path.dirname(__file__), "data", "kc_house_data.csv")
        demographics_path = os.path.join(os.path.dirname(__file__), "data", "zipcode_demographics.csv")
        cls.sales_data = pd.read_csv(sales_path, dtype={'zipcode': str})
        cls.demographics = pd.read_csv(demographics_path, dtype={'zipcode': str})

        # Prepare test data for predictions
        cls.test_sample = cls.sales_data.head(1).copy()
        cls.test_sample = cls.test_sample.merge(cls.demographics, on='zipcode', how='left')
        cls.test_sample = cls.test_sample.drop(columns=['zipcode', 'price'])
        cls.test_sample = cls.test_sample.select_dtypes(include=[np.number]).fillna(0)

        # Add sale_year and sale_month for improved model if needed
        if cls.improved_features and 'sale_year' in cls.improved_features:
            cls.test_sample['sale_year'] = 2023
        if cls.improved_features and 'sale_month' in cls.improved_features:
            cls.test_sample['sale_month'] = 6

    def test_basic_model_loaded(self):
        """Test that basic model is loaded correctly"""
        if self.basic_model is None:
            self.skipTest("Basic model not found - skipping test")
        self.assertIsNotNone(self.basic_model, "Basic model should be loaded")
        self.assertIsInstance(self.basic_features, list, "Basic features should be a list")

    def test_improved_model_loaded(self):
        """Test that improved model is loaded correctly"""
        if self.improved_model is None:
            self.skipTest("Improved model not found - skipping test")
        self.assertIsNotNone(self.improved_model, "Improved model should be loaded")
        self.assertIsInstance(self.improved_features, list, "Improved features should be a list")

    def test_basic_model_prediction(self):
        """Test that basic model can make predictions"""
        if self.basic_model is None or self.basic_features is None:
            self.skipTest("Basic model or features not found - skipping test")
        # Select features for basic model
        features = self.test_sample[self.basic_features]
        prediction = self.basic_model.predict(features)
        self.assertIsInstance(prediction, np.ndarray, "Prediction should be numpy array")
        self.assertEqual(len(prediction), 1, "Should predict one value")
        self.assertGreater(prediction[0], 0, "Prediction should be positive")

    def test_improved_model_prediction(self):
        """Test that improved model can make predictions"""
        if self.improved_model is None or self.improved_features is None:
            self.skipTest("Improved model or features not found - skipping test")
        # Select features for improved model
        features = self.test_sample[self.improved_features]
        prediction = self.improved_model.predict(features)
        self.assertIsInstance(prediction, np.ndarray, "Prediction should be numpy array")
        self.assertEqual(len(prediction), 1, "Should predict one value")
        self.assertGreater(prediction[0], 0, "Prediction should be positive")

    def test_data_loaded(self):
        """Test that test data is loaded correctly"""
        self.assertIsNotNone(self.sales_data, "Sales data should be loaded")
        self.assertIsNotNone(self.demographics, "Demographics data should be loaded")
        self.assertGreater(len(self.sales_data), 0, "Sales data should not be empty")
        self.assertGreater(len(self.demographics), 0, "Demographics data should not be empty")

    def test_models_have_different_features(self):
        """Test that basic and improved models have different feature sets"""
        if self.basic_features is None or self.improved_features is None:
            self.skipTest("Model features not found - skipping test")
        self.assertNotEqual(self.basic_features, self.improved_features,
                           "Basic and improved models should have different features")

if __name__ == '__main__':
    # Run tests when script is executed directly
    unittest.main(verbosity=2, exit=False)