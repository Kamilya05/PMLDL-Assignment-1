from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

app = FastAPI(title="Titanic Survival Prediction API")

class PassengerData(BaseModel):
    sex: str  # "male" or "female"
    age: float
    fare: float
    sibsp: int  # number of siblings/spouses
    parch: int  # number of parents/children

try:
    with open('/app/models/titanic_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('/app/models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("Model and encoder loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    label_encoder = None

@app.get("/")
def read_root():
    return {"message": "Titanic Survival Prediction API"}

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.post("/predict")
def predict_survival(passenger: PassengerData):
    if model is None or label_encoder is None:
        raise HTTPException(status_code=500, detail=f"Model not loaded")
    
    try:
        sex_encoded = label_encoder.transform([passenger.sex])[0]
        
        features = np.array([[sex_encoded, passenger.age, passenger.fare, 
                            passenger.sibsp, passenger.parch]])
        
        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0][1]
        
        return {
            "survival_prediction": int(prediction),
            "survival_probability": float(probability),
            "survival_status": "Survived" if prediction == 1 else "Did not survive",
            "confidence": float(probability) if prediction == 1 else float(1 - probability)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)