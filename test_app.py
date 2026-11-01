
import unittest
from unittest.mock import patch

import app as prediction_app


class TestPredictionAPI(unittest.TestCase):

    def setUp(self):
        prediction_app.app.config["TESTING"] = True
        self.client = prediction_app.app.test_client()

    def test_home_endpoint(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "ok")

    def test_missing_json(self):
        response = self.client.post("/predict", json={})
        self.assertEqual(response.status_code, 400)

    def test_missing_fields(self):
        response = self.client.post(
            "/predict",
            json={"attendance": 90}
        )
        self.assertEqual(response.status_code, 400)

    @patch("app.load_model")
    def test_prediction_success(self, mock_load_model):
        mock_load_model.return_value.predict.return_value = [1]

        response = self.client.post(
            "/predict",
            json={
                "attendance": 90,
                "internal_marks": 85,
                "assignment_marks": 80,
                "previous_score": 75,
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["prediction"], "PASS")


if __name__ == "__main__":
    unittest.main()
