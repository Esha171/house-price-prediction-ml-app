from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load the model
model = joblib.load("D:/ml_app_assignment/ml/model.pkl")

class HouseFeatures(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    stories: int
    mainroad: int
    guestroom: int
    basement: int
    hotwaterheating: int
    airconditioning: int
    parking: int
    prefarea: int
    furnishingstatus: int

@app.post("/predict")
def predict_price(data: HouseFeatures):
    features = np.array([[data.area, data.bedrooms, data.bathrooms, data.stories,
                          data.mainroad, data.guestroom, data.basement, data.hotwaterheating,
                          data.airconditioning, data.parking, data.prefarea, data.furnishingstatus]])
    prediction = model.predict(features)
    return {"predicted_price": prediction[0]}

