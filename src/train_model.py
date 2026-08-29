import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import sys
import os

# Check if dataset exists
if not os.path.exists("data/gesture_dataset.csv"):
    print("Error: Dataset not found. Please run app.py first to collect gesture data.")
    sys.exit(1)

try:
    # Load dataset
    df = pd.read_csv("data/gesture_dataset.csv")
    
    if df.empty:
        print("Error: Dataset is empty. Please collect gesture data first.")
        sys.exit(1)
    
    print(f"Loaded {len(df)} gesture samples")
except Exception as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)

try:
    # Features and labels
    X = df.iloc[:, :-1]   # first 63 columns
    y = df.iloc[:, -1]    # last column (A, B, C)
    
    if len(X) < 10:
        print("Error: Need at least 10 samples to train. Please collect more data.")
        sys.exit(1)
    
    print(f"Found {len(X.columns)} features")
    print(f"Gesture classes: {y.unique()}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n✓ Model Accuracy: {accuracy:.2%}")
    
    # Save model
    if not os.path.exists("model"):
        os.makedirs("model")
    
    joblib.dump(model, "model/gesture_model.pkl")
    print("✓ Model saved successfully to model/gesture_model.pkl")
    print("\nNext: Run run_predict.bat to test the model!")
    
except Exception as e:
    print(f"Error training model: {e}")
    sys.exit(1)