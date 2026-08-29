"""
Gesture Recognition Module
Provides utilities for gesture recognition and model handling
"""

import joblib
import numpy as np
import os


class GestureRecognizer:
    """
    Recognizes gestures from hand landmarks using a trained ML model.
    """

    def __init__(self, model_path="model/gesture_model.pkl"):
        """
        Initialize the gesture recognizer with a trained model.
        
        Args:
            model_path: Path to the trained model file
            
        Raises:
            FileNotFoundError: If model file doesn't exist
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = joblib.load(model_path)
        self.model_path = model_path

    def predict(self, landmark_list):
        """
        Predict gesture from landmark features.
        
        Args:
            landmark_list: List of 63 normalized landmark coordinates
            
        Returns:
            str: Predicted gesture label (e.g., 'A', 'B', 'C')
        """
        if len(landmark_list) != 63:
            return ""

        data = np.array(landmark_list).reshape(1, -1)
        prediction = self.model.predict(data)[0]
        return str(prediction)

    def predict_proba(self, landmark_list):
        """
        Get prediction probabilities for all classes.
        
        Args:
            landmark_list: List of 63 normalized landmark coordinates
            
        Returns:
            dict: Class probabilities
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
