import pickle
from sklearn.model_selection import train_test_split
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Heart Disease Prediction API")

# Load the trained model
with open("heart.pkl", "rb") as f:
    model = pickle.load(f)

class Patient(BaseModel): 
    age: int
    sex: int 
    cp: int
    trestbps: int 
    chol: int  
    fbs: int  
    restecg: int 
    thalach: int  
    exang: int
    oldpeak: float
    slope:int  
    ca: int 
    thal:int  


@app.get("/a")
def read_root():
    return {"message": "Welcome to Heart Prediction API"}

@app.post("/predict")
def predict(patient: Patient):
    data = patient.dict()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"Sick": int(prediction)}
