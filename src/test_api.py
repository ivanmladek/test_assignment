import unittest
import requests
import pandas as pd
import json

class TestHousePriceAPI(unittest.TestCase):
    """Test suite for house price prediction API"""

    BASE_URL = "http://127.0.0.1:8000"
    ENDPOINTS = {
        "predict": f"{BASE_URL}/predict",
        "basic": f"{BASE_URL}/predict_basic",
        "improved": f"{BASE_URL}/predict_improved"
    }

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests"""
        # Load test data
        cls.test_data = pd.read_csv("data/future_unseen_examples.csv")

    def test_api_endpoints_exist(self):
        """Test that all API endpoints are accessible"""
        for name, url in self.ENDPOINTS.items():
            with self.subTest(endpoint=name):
                try:
                    response = requests.get(self.BASE_URL + "/")
                    self.assertEqual(response.status_code, 200,
                                   f"Root endpoint should be accessible for {name}")
                except requests.exceptions.ConnectionError:
                    self.fail(f"Cannot connect to API server for {name} endpoint")

    def test_predict_endpoint(self):
        """Test the main predict endpoint"""
        # Get first row of test data
        row = self.test_data.iloc[0]
        payload = self._prepare_payload(row)

        response = requests.post(self.ENDPOINTS["predict"], json=payload)
        self.assertEqual(response.status_code, 200,
                        f"Predict endpoint should return 200, got {response.status_code}")

        result = response.json()
        self.assertIn("prediction", result, "Response should contain prediction")
        self.assertIsInstance(result["prediction"], (int, float),
                            "Prediction should be numeric")
        self.assertGreater(result["prediction"], 0, "Prediction should be positive")

    def test_predict_basic_endpoint(self):
        """Test the basic predict endpoint"""
        # Get first row of test data
        row = self.test_data.iloc[0]
        payload = self._prepare_payload(row)
        basic_payload = self._create_basic_payload(payload)

        response = requests.post(self.ENDPOINTS["basic"], json=basic_payload)
        self.assertEqual(response.status_code, 200,
                        f"Basic predict endpoint should return 200, got {response.status_code}")

        result = response.json()
        self.assertIn("prediction", result, "Response should contain prediction")
        self.assertIn("model", result, "Response should contain model type")
        self.assertEqual(result["model"], "basic", "Model type should be 'basic'")
        self.assertIsInstance(result["prediction"], (int, float),
                            "Prediction should be numeric")
        self.assertGreater(result["prediction"], 0, "Prediction should be positive")

    def test_predict_improved_endpoint(self):
        """Test the improved predict endpoint"""
        # Get first row of test data
        row = self.test_data.iloc[0]
        payload = self._prepare_payload(row)

        response = requests.post(self.ENDPOINTS["improved"], json=payload)
        self.assertEqual(response.status_code, 200,
                        f"Improved predict endpoint should return 200, got {response.status_code}")

        result = response.json()
        self.assertIn("prediction", result, "Response should contain prediction")
        self.assertIn("model", result, "Response should contain model type")
        self.assertEqual(result["model"], "improved", "Model type should be 'improved'")
        self.assertIsInstance(result["prediction"], (int, float),
                            "Prediction should be numeric")
        self.assertGreater(result["prediction"], 0, "Prediction should be positive")

    def test_multiple_predictions(self):
        """Test predictions for multiple data points"""
        for i in range(min(3, len(self.test_data))):  # Test first 3 rows
            with self.subTest(row=i):
                row = self.test_data.iloc[i]
                payload = self._prepare_payload(row)

                response = requests.post(self.ENDPOINTS["predict"], json=payload)
                self.assertEqual(response.status_code, 200,
                               f"Prediction {i} should return 200, got {response.status_code}")

                result = response.json()
                self.assertIn("prediction", result, f"Response {i} should contain prediction")
                self.assertGreater(result["prediction"], 0, f"Prediction {i} should be positive")

    def test_invalid_zipcode(self):
        """Test error handling for invalid zipcode"""
        payload = {
            "bedrooms": 3,
            "bathrooms": 2.5,
            "sqft_living": 2220,
            "sqft_lot": 7350,
            "floors": 2,
            "waterfront": 0,
            "view": 0,
            "condition": 3,
            "grade": 8,
            "yr_built": 1945,
            "yr_renovated": 0,
            "zipcode": "99999",  # Invalid zipcode
            "lat": 47.6764,
            "long": -122.293
        }

        response = requests.post(self.ENDPOINTS["predict"], json=payload)
        self.assertEqual(response.status_code, 404,
                        f"Invalid zipcode should return 404, got {response.status_code}")

        result = response.json()
        self.assertIn("detail", result, "Error response should contain detail")

    def _prepare_payload(self, row):
        """Helper method to prepare payload from data row"""
        payload = row.to_dict()
        # Convert numeric fields to int/float as expected by API
        int_fields = ['bedrooms', 'sqft_living', 'sqft_lot', 'waterfront', 'view',
                     'condition', 'grade', 'sqft_above', 'sqft_basement', 'yr_built',
                     'yr_renovated', 'sqft_living15', 'sqft_lot15']
        for field in int_fields:
            if field in payload:
                payload[field] = int(payload[field])

        # Ensure zipcode is string
        if 'zipcode' in payload:
            payload['zipcode'] = str(int(payload['zipcode']))

        return payload

    def _create_basic_payload(self, payload):
        """Helper method to create basic payload with required fields only"""
        basic_fields = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                       'floors', 'sqft_above', 'sqft_basement', 'zipcode']
        return {k: v for k, v in payload.items() if k in basic_fields}

if __name__ == '__main__':
    unittest.main(verbosity=2)