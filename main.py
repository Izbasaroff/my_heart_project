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

