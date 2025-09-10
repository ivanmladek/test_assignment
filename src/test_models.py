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
        with open("model/model.pkl", "rb") as f:
            cls.basic_model = pickle.load(f)
        with open("model/model_improved.pkl", "rb") as f:
            cls.improved_model = pickle.load(f)

        with open("model/model_features.json", "r") as f:
            cls.basic_features = json.load(f)
        with open("model/model_features_improved.json", "r") as f:
            cls.improved_features = json.load(f)

        # Load test data
        cls.sales_data = pd.read_csv("data/kc_house_data.csv", dtype={'zipcode': str})
        cls.demographics = pd.read_csv("data/zipcode_demographics.csv", dtype={'zipcode': str})

    def test_data_loading(self):
        """Test that data loads correctly"""
        self.assertGreater(len(self.sales_data), 0, "Sales data should not be empty")
        self.assertGreater(len(self.demographics), 0, "Demographics data should not be empty")
        self.assertEqual(self.sales_data['zipcode'].dtype, 'object', "Zipcode should be string type")
        self.assertEqual(self.demographics['zipcode'].dtype, 'object', "Zipcode should be string type")

    def test_zipcode_coverage(self):
        """Test that zipcodes in sales data exist in demographics"""
        sales_zipcodes = set(self.sales_data['zipcode'].unique())
        demo_zipcodes = set(self.demographics['zipcode'].unique())

        # All sales zipcodes should exist in demographics
        missing_zipcodes = sales_zipcodes - demo_zipcodes
        self.assertEqual(len(missing_zipcodes), 0,
                        f"Missing zipcodes in demographics: {missing_zipcodes}")

    def test_data_merge(self):
        """Test that data merges correctly without null values"""
        merged = pd.merge(self.sales_data, self.demographics, on='zipcode', how='left')
        null_count = merged.isnull().sum().sum()
        self.assertEqual(null_count, 0, f"Merge should not produce null values, found {null_count}")

    def test_basic_model_features(self):
        """Test that basic model has expected features"""
        expected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                           'sqft_above', 'sqft_basement']
        for feature in expected_features:
            self.assertIn(feature, self.basic_features,
                         f"Expected feature '{feature}' not found in basic model features")

        # Check that demographics features are included
        demo_features = ['ppltn_qty', 'medn_hshld_incm_amt', 'hous_val_amt']
        for feature in demo_features:
            self.assertIn(feature, self.basic_features,
                         f"Expected demographics feature '{feature}' not found in basic model features")

    def test_improved_model_features(self):
        """Test that improved model has expected features"""
        expected_features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
                           'waterfront', 'view', 'condition', 'grade', 'yr_built', 'yr_renovated',
                           'lat', 'long', 'sqft_living15', 'sqft_lot15', 'sale_year', 'sale_month']
        for feature in expected_features:
            self.assertIn(feature, self.improved_features,
                         f"Expected feature '{feature}' not found in improved model features")

    def test_basic_model_prediction(self):
        """Test that basic model can make predictions"""
        # Prepare test data
        merged = pd.merge(self.sales_data, self.demographics, on='zipcode', how='left')
        y = merged.pop('price')
        X = merged[self.basic_features]

        # Test prediction
        predictions = self.basic_model.predict(X.head(5))
        self.assertEqual(len(predictions), 5, "Should return 5 predictions")
        self.assertTrue(all(pred > 0 for pred in predictions), "All predictions should be positive")

    def test_improved_model_prediction(self):
        """Test that improved model can make predictions"""
        # Prepare test data
        sales_copy = self.sales_data.copy()
        sales_copy['date'] = pd.to_datetime(sales_copy['date'])
        sales_copy['sale_year'] = sales_copy['date'].dt.year
        sales_copy['sale_month'] = sales_copy['date'].dt.month
        sales_copy = sales_copy.drop(columns=['date', 'id'])

        merged = pd.merge(sales_copy, self.demographics, on='zipcode', how='left')
        y = merged.pop('price')
        X = merged.drop(columns=['zipcode']).apply(pd.to_numeric, errors='coerce').fillna(0)

        # Test prediction
        predictions = self.improved_model.predict(X.head(5))
        self.assertEqual(len(predictions), 5, "Should return 5 predictions")
        self.assertTrue(all(pred > 0 for pred in predictions), "All predictions should be positive")

    def test_specific_zipcode_98042(self):
        """Test predictions for zipcode 98042"""
        self._test_zipcode_prediction('98042')

    def test_specific_zipcode_98115(self):
        """Test predictions for zipcode 98115"""
        self._test_zipcode_prediction('98115')

    def test_specific_zipcode_98118(self):
        """Test predictions for zipcode 98118"""
        self._test_zipcode_prediction('98118')

    def _test_zipcode_prediction(self, zipcode):
        """Helper method to test prediction for a specific zipcode"""
        # Get data for this zipcode
        zip_sales = self.sales_data[self.sales_data['zipcode'] == zipcode]
        self.assertGreater(len(zip_sales), 0, f"No sales data found for zipcode {zipcode}")

        # Test basic model
        basic_data = zip_sales[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                               'floors', 'sqft_above', 'sqft_basement', 'zipcode']]
        basic_merged = pd.merge(basic_data, self.demographics, on='zipcode', how='left')
        basic_merged = basic_merged.drop(columns=['zipcode'])
        basic_sample = basic_merged.iloc[0:1]

        basic_pred = self.basic_model.predict(basic_sample)
        self.assertGreater(basic_pred[0], 0, f"Basic model prediction should be positive for {zipcode}")

        # Test improved model
        improved_data = zip_sales.copy()
        improved_data['date'] = pd.to_datetime(improved_data['date'])
        improved_data['sale_year'] = improved_data['date'].dt.year
        improved_data['sale_month'] = improved_data['date'].dt.month
        improved_data = improved_data.drop(columns=['date', 'id'])

        improved_merged = pd.merge(improved_data, self.demographics, on='zipcode', how='left')
        if 'price' in improved_merged.columns:
            improved_merged = improved_merged.drop(columns=['price'])
        improved_merged = improved_merged.drop(columns=['zipcode'])
        improved_merged = improved_merged.apply(pd.to_numeric, errors='coerce').fillna(0)
        improved_sample = improved_merged.iloc[0:1]

        improved_pred = self.improved_model.predict(improved_sample)
        self.assertGreater(improved_pred[0], 0, f"Improved model prediction should be positive for {zipcode}")

if __name__ == '__main__':
    unittest.main(verbosity=2)