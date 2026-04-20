"""
Simple test script to verify the model works correctly
Run this AFTER running: python train.py
"""

import pickle
import numpy as np

print("="*60)
print("Testing Student Burnout Risk Predictor Model")
print("="*60)

# Load model and encoder
print("\nLoading model and encoder...")
try:
    with open('burnout_model.pkl', 'rb') as f:
        pipeline = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    print("✓ Model loaded successfully!")
except FileNotFoundError:
    print("✗ Error: Model files not found!")
    print("Please run 'python train.py' first to train the model.")
    exit(1)

# Test cases
test_cases = [
    {"name": "High Burnout", "studytime": 4, "failures": 1, "absences": 3},
    {"name": "Medium Burnout", "studytime": 2, "failures": 2, "absences": 8},
    {"name": "Low Burnout", "studytime": 1, "failures": 0, "absences": 2},
]

print(f"\nBurnout Classes: {list(label_encoder.classes_)}")
print("\n" + "="*60)
print("Test Results:")
print("="*60)

for i, test in enumerate(test_cases, 1):
    print(f"\nTest {i}: {test['name']}")
    print(f"  Input: studytime={test['studytime']}, failures={test['failures']}, absences={test['absences']}")
    
    # Prepare input
    X = np.array([[test['studytime'], test['failures'], test['absences']]])
    
    # Make prediction
    prediction_encoded = pipeline.predict(X)[0]
    burnout_risk = label_encoder.inverse_transform([prediction_encoded])[0]
    
    # Get probability
    probability = pipeline.predict_proba(X)[0]
    confidence = float(np.max(probability))
    
    print(f"  → Prediction: {burnout_risk}")
    print(f"  → Confidence: {confidence:.2%}")
    print(f"  → Probabilities: {dict(zip(label_encoder.classes_, probability.round(3)))}")

print("\n" + "="*60)
print("✓ Model Testing Complete!")
print("="*60)
