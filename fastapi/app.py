from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import joblib
import mlflow
import os

app = FastAPI(title="Crop Prediction App")
templates = Jinja2Templates(directory="/app/templates")

model_path = "/app/crop_model.pkl"
model = joblib.load(model_path)

mlflow_uri = os.getenv("MLFLOW_TRACKING_URI","http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)
mlflow.set_experiment("Crop_Prediction")

class CropRequest(BaseModel):
    N:int
    P:int
    K:int
    temperature:float
    humidity:float
    ph:float
    rainfall:float

@app.get("/", response_class=HTMLResponse)
def home(request:Request):
    return templates.TemplateResponse("index.html", {"request":request})

@app.post("/predict")
def predict(request:CropRequest):
    df = pd.DataFrame([request.dict()])
    prediction = model.predict(df)
    with mlflow.start_run():
        mlflow.log_params(request.dict())
        mlflow.set_tag("predicted_crop", str(prediction[0]))
    return {"predicted_crop":prediction[0]}