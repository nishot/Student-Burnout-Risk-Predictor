import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Step 1: Load the dataset
print("Step 1: Loading dataset...")
df = pd.read_csv('student-mat.csv', sep=';')
print(f"Dataset shape: {df.shape}")
print(df.head())

# Step 2: Data Cleaning - Drop missing values
print("\nStep 2: Data cleaning...")
df = df.dropna()
print(f"Shape after dropping missing values: {df.shape}")

# Step 3: Select required columns
print("\nStep 3: Selecting required columns...")
df = df[['studytime', 'failures', 'absences', 'G3']]
print(f"Selected columns: {df.columns.tolist()}")

# Step 4: Feature Engineering - Create burnout column
print("\nStep 4: Feature Engineering - Creating burnout column...")
def create_burnout_label(row):
    # High burnout: high studytime and low G3
    if row['studytime'] >= 3 and row['G3'] <= 5:
        return 'High'
    # Medium burnout: failures > 1
    elif row['failures'] > 1:
        return 'Medium'
    # Low burnout: otherwise
    else:
        return 'Low'

df['burnout'] = df.apply(create_burnout_label, axis=1)
print(f"Burnout distribution:\n{df['burnout'].value_counts()}")

# Step 5: Feature Selection - Select features for training
print("\nStep 5: Feature Selection...")
X = df[['studytime', 'failures', 'absences']]
y = df['burnout']
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Step 6: Encoding - Encode target variable (burnout)
print("\nStep 6: Encoding target variable...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
print(f"Classes: {label_encoder.classes_}")
print(f"Encoded target: {y_encoded[:5]}")

# Step 7: Train-Test Split
print("\nStep 7: Train-Test Split (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
print(f"Training set size: {X_train.shape[0]}")
print(f"Testing set size: {X_test.shape[0]}")

# Step 8: Create Pipeline
print("\nStep 8: Creating pipeline...")
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=200, random_state=42))
])

# Step 9: Train Model
print("\nStep 9: Training model...")
pipeline.fit(X_train, y_train)
print("Model training completed!")

# Step 10: Evaluate
print("\nStep 10: Evaluation...")
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Step 11: Save the model and encoder
print("\nStep 11: Saving model and encoder...")
with open('burnout_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)
print("Model and encoder saved successfully!")

print("\n" + "="*50)
print("Training Complete!")
print("="*50)
