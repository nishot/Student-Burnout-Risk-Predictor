# Student Burnout Risk Predictor 🎓

A simple machine learning project to predict student burnout risk based on study habits.

## Project Structure

```
├── train.py              # Main ML training script
├── app.py                # FastAPI backend
├── student-mat.csv       # Sample dataset
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## How It Works

### 1. Data Loading
- Loads `student-mat.csv` with `;` separator

### 2. Data Cleaning
- Removes rows with missing values
- Selects columns: `studytime`, `failures`, `absences`, `G3`

### 3. Feature Engineering
Creates a `burnout` column with three categories:
- **High**: High study time (≥3) AND Low grades (≤5)
- **Medium**: More than 1 failure
- **Low**: Otherwise

### 4. Model Training
- **Features**: studytime, failures, absences
- **Scaler**: StandardScaler
- **Model**: LogisticRegression
- **Split**: 80-20 train-test

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Train the Model

```bash
python train.py
```

This will:
- Load and clean the data
- Create burnout labels
- Train the model
- Print accuracy
- Save `burnout_model.pkl` and `label_encoder.pkl`

### Step 2: Run FastAPI Backend

```bash
python app.py
```

The API will start at `http://localhost:8000`

## API Endpoints

### 1. Root Endpoint
**GET** `/`
- Returns API information

### 2. Health Check
**GET** `/health`
- Checks if API is running

### 3. Model Info
**GET** `/info`
- Returns model details and classes

### 4. Single Prediction
**POST** `/predict`

Request:
```json
{
  "studytime": 3,
  "failures": 1,
  "absences": 5
}
```

Response:
```json
{
  "studytime": 3,
  "failures": 1,
  "absences": 5,
  "burnout_risk": "High",
  "confidence": 0.85
}
```

### 5. Batch Prediction
**POST** `/predict-batch`

Request:
```json
[
  {"studytime": 3, "failures": 1, "absences": 5},
  {"studytime": 1, "failures": 3, "absences": 12}
]
```

Response:
```json
[
  {
    "studytime": 3,
    "failures": 1,
    "absences": 5,
    "burnout_risk": "High",
    "confidence": 0.85
  },
  {
    "studytime": 1,
    "failures": 3,
    "absences": 12,
    "burnout_risk": "Medium",
    "confidence": 0.92
  }
]
```

## API Documentation

Once the API is running, visit:
- **Interactive Docs**: `http://localhost:8000/docs`
- **Alternative Docs**: `http://localhost:8000/redoc`

## Files Generated After Training

- `burnout_model.pkl` - Trained pipeline (scaler + model)
- `label_encoder.pkl` - Encoder for burnout classes

## Example Flow

```bash
# 1. Train model
python train.py

# 2. Start API in another terminal
python app.py

# 3. Make predictions via API
# Visit http://localhost:8000/docs and try it out!
```

## Burnout Categories

| Category | Description |
|----------|-------------|
| High | High study time (≥3 hours/week) + Low grades (≤5) |
| Medium | More than 1 previous failure |
| Low | Normal study pattern, good grades |

## Input Features

| Feature | Description | Range |
|---------|-------------|-------|
| studytime | Weekly study time (1-4) | 1-4 hours |
| failures | Number of past failures (0-4) | 0-4 |
| absences | Number of absences (0-93) | 0-93 days |

## Model Information

- **Algorithm**: Logistic Regression
- **Preprocessing**: StandardScaler
- **Framework**: Scikit-learn
- **Backend**: FastAPI
- **Classes**: High, Medium, Low

## Notes

- This is a beginner-friendly project
- Dataset is a sample with 20 records
- Replace with your own dataset in the same format
- Model accuracy depends on training data quality

Enjoy! 🚀
