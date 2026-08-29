"""
Hand Detection Module
Provides utilities for detecting hand landmarks using MediaPipe
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """
    Detects hand landmarks from video frames using MediaPipe.
    """

    def __init__(self, max_num_hands=1, min_detection_confidence=0.7):
        """
        Initialize the hand detector.
        
        Args:
            max_num_hands: Maximum number of hands to detect
            min_detection_confidence: Minimum confidence for detection
        """
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def detect(self, frame):
        """
        Detect hands in a frame.
        
        Args:
            frame: Input video frame (BGR format)
            
        Returns:
            tuple: (processed_frame, hand_landmarks, results)
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )

        return frame, results.multi_hand_landmarks, results

    def extract_landmarks(self, hand_landmarks):
        """
        Extract landmark coordinates from hand.
        
        Args:
            hand_landmarks: MediaPipe hand landmarks
            
        Returns:
            list: Normalized landmark coordinates (63 features)
        """
        landmark_list = []
        wrist = hand_landmarks.landmark[0]

        for lm in hand_landmarks.landmark:
            landmark_list.append(lm.x - wrist.x)
            landmark_list.append(lm.y - wrist.y)
            landmark_list.append(lm.z - wrist.z)

        return landmark_list

    def close(self):
        """Close the hand detector resources."""
        self.hands.close()
