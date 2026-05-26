# Put the code for your API here.
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
from ml.data import process_data
import os
app = FastAPI()

# Load model artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "..", "model", "model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "..", "model", "encoder.pkl"))
lb = joblib.load(os.path.join(BASE_DIR, "..", "model", "lb.pkl"))

# Root endpoint


@app.get("/")
def welcome():
    return {"message": "Welcome to Census Income Prediction API"}


# Pydantic model
class CensusData(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int
    capital_loss: int
    hours_per_week: int
    native_country: str

    class Config:
        json_schema_extra = {
            "example": {
                "age": 37,
                "workclass": "Private",
                "fnlgt": 284582,
                "education": "Bachelors",
                "education_num": 13,
                "marital_status": "Married-civ-spouse",
                "occupation": "Exec-managerial",
                "relationship": "Husband",
                "race": "White",
                "sex": "Male",
                "capital_gain": 0,
                "capital_loss": 0,
                "hours_per_week": 40,
                "native_country": "United-States"
            }
        }


# Prediction endpoint
@app.post("/predict")
def predict(data: CensusData):

    input_data = pd.DataFrame(
        {
            "age": [data.age],
            "workclass": [data.workclass],
            "fnlgt": [data.fnlgt],
            "education": [data.education],
            "education-num": [data.education_num],
            "marital-status": [data.marital_status],
            "occupation": [data.occupation],
            "relationship": [data.relationship],
            "race": [data.race],
            "sex": [data.sex],
            "capital-gain": [data.capital_gain],
            "capital-loss": [data.capital_loss],
            "hours-per-week": [data.hours_per_week],
            "native-country": [data.native_country],
        }
    )

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = process_data(
        input_data,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    prediction = model.predict(X)[0]

    result = ">50K" if prediction == 1 else "<=50K"

    return {"prediction": result}
