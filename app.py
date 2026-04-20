from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(title="Student Burnout Risk Predictor", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained model and encoder
print("Loading model and encoder...")
with open('burnout_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)
print("Model loaded successfully!")

# Define input schema
class StudentData(BaseModel):
    studytime: int
    failures: int
    absences: int

# Define response schema
class PredictionResponse(BaseModel):
    studytime: int
    failures: int
    absences: int
    burnout_risk: str
    confidence: float

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Student Burnout Risk Predictor API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - POST request to predict burnout risk",
            "health": "/health - Check API health",
            "info": "/info - Get model information"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy", "message": "API is running"}

# Model info endpoint
@app.get("/info")
def model_info():
    return {
        "model": "LogisticRegression",
        "scaler": "StandardScaler",
        "features": ["studytime", "failures", "absences"],
        "classes": label_encoder.classes_.tolist(),
        "training_accuracy": "Check train.py output"
    }

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict_burnout(data: StudentData):
    # Prepare input
    X = np.array([[data.studytime, data.failures, data.absences]])
    
    # Make prediction
    prediction_encoded = pipeline.predict(X)[0]
    burnout_risk = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get prediction probability
    probability = pipeline.predict_proba(X)[0]
    confidence = float(np.max(probability))
    
    return PredictionResponse(
        studytime=data.studytime,
        failures=data.failures,
        absences=data.absences,
        burnout_risk=burnout_risk,
        confidence=confidence
    )

# Batch prediction endpoint (optional)
@app.post("/predict-batch")
def predict_batch(data_list: list[StudentData]):
    results = []
    for data in data_list:
        X = np.array([[data.studytime, data.failures, data.absences]])
        prediction_encoded = pipeline.predict(X)[0]
        burnout_risk = label_encoder.inverse_transform([prediction_encoded])[0]
        probability = pipeline.predict_proba(X)[0]
        confidence = float(np.max(probability))
        
        results.append({
            "studytime": data.studytime,
            "failures": data.failures,
            "absences": data.absences,
            "burnout_risk": burnout_risk,
            "confidence": confidence
        })
    
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
