# flake8: noqa: E402

import os
import sys

sys.path.insert(
    0,
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..")
    ),
)

from fastapi.testclient import TestClient
from main import app


client = TestClient(app)


def test_get():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to Census Income Prediction API"
    }


def test_predict_low_income():
    response = client.post(
        "/predict",
        json={
            "age": 25,
            "workclass": "Private",
            "fnlgt": 226802,
            "education": "11th",
            "education_num": 7,
            "marital_status": "Never-married",
            "occupation": "Machine-op-inspct",
            "relationship": "Own-child",
            "race": "Black",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]


def test_predict_high_income():
    response = client.post(
        "/predict",
        json={
            "age": 52,
            "workclass": "Self-emp-not-inc",
            "fnlgt": 209642,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 15024,
            "capital_loss": 0,
            "hours_per_week": 60,
            "native_country": "United-States"
        }
    )

    assert response.status_code == 200
    assert response.json()["prediction"] in ["<=50K", ">50K"]