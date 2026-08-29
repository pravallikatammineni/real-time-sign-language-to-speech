"""
Gesture Recognition Module
Takes 63 hand coordinates and figures out what gesture it is.
Uses a trained Random Forest classifier to do the heavy lifting.
"""

import joblib
import numpy as np
import os


class GestureRecognizer:
    """
    Recognizes gestures from hand landmarks using a trained ML model.
    
    This loads a trained model file and uses it to predict which gesture
    you're making based on your hand position. Simple and effective.
    """

    def __init__(self, model_path="model/gesture_model.pkl"):
        """
        Load up the trained model.
        
        Args:
            model_path: Where the model file is. Needs to exist, so make sure
                       you've run train_model.py first.
            
        Raises:
            FileNotFoundError: If model_path doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run train_model.py first.")
        
        self.model = joblib.load(model_path)
        self.model_path = model_path

    def predict(self, landmark_list):
        """
        Predict what gesture you're making.
        
        Takes 63 hand coordinates and runs them through the model.
        Returns whatever label the model was trained on (A, B, C, etc.).
        
        Args:
            landmark_list: List of 63 numbers (21 points × 3 coordinates)
            
        Returns:
            str: The predicted gesture label (or empty string if invalid input)
        """
        if len(landmark_list) != 63:
            return ""

        data = np.array(landmark_list).reshape(1, -1)
        prediction = self.model.predict(data)[0]
        return str(prediction)

    def predict_proba(self, landmark_list):
        """
        Get confidence scores for each gesture class.
        
        Instead of just "A or B or C", this tells you how confident
        the model is in each option. Useful for filtering out weak predictions.
        
        Args:
            landmark_list: List of 63 hand coordinates
            
        Returns:
            dict: Like {'A': 0.95, 'B': 0.04, 'C': 0.01} (example)
        """
        if len(landmark_list) != 63:
            return {}

        data = np.array(landmark_list).reshape(1, -1)
        
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(data)[0]
            classes = self.model.classes_
            return {str(c): float(p) for c, p in zip(classes, proba)}
        
        return {}

    def get_model_info(self):
        """
        Get information about the loaded model.
        
        Returns:
            dict: Model information
        """
        return {
            "model_type": type(self.model).__name__,
            "model_path": self.model_path,
            "classes": list(self.model.classes_) if hasattr(self.model, 'classes_') else [],
        }
