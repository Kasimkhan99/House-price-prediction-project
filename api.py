from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
import pandas as pd


# Load model
model = joblib.load("house_price_model.pkl")

app = FastAPI()

@app.get('/')
def home():
    return {'message':'House prediction API'}
@app.get('/health')
def health_check():
    return {
        'status':'OK'
    }

@app.post("/predict")
def predict(data: dict):
    """
    Example JSON input:
    {
        "area": 1500, 
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "parking": 1,
        "year_built": 2010,
        "city": "Delhi"
    }
    """
    # Derived features
    house_age = 2025 - data["year_built"]
    area_per_bedroom = data["area"] / data["bedrooms"] if data["bedrooms"] > 0 else data["area"]

    input_df = pd.DataFrame([{
        "area": data["area"],
        "bedrooms": data["bedrooms"],
        "bathrooms": data["bathrooms"],
        "stories": data["stories"],
        "parking": data["parking"],
        "year_built": data["year_built"],
        "city": data["city"],
        "house_age": house_age,
        "area_per_bedroom": area_per_bedroom
    }])

    prediction = model.predict(input_df)[0]
    return JSONResponse(status_code=200,content={"predicted_price": float(prediction)})
