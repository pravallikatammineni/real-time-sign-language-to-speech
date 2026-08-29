"""
Hand Detection Module
Wraps MediaPipe's hand detection to make it cleaner to use.
Finds 21 landmark points on each hand detected in a frame.
"""

import cv2
import mediapipe as mp
import numpy as np


class HandDetector:
    """
    Detects hand landmarks from video frames using MediaPipe.
    
    This is basically a wrapper around MediaPipe that makes it easier to work with.
    It handles all the setup boilerplate and gives you clean methods to detect hands
    and extract their landmark coordinates.
    """

    def __init__(self, max_num_hands=1, min_detection_confidence=0.7):
        """
        Set up the hand detector.
        
        Args:
            max_num_hands: How many hands to find in each frame (default: 1)
            min_detection_confidence: How confident MediaPipe needs to be (0-1).
                                      Lower = easier to find hands but might be wrong,
                                      Higher = only detects when sure. 0.7 is good middle ground.
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
        Find hands in a video frame.
        
        Takes a single frame and finds all the hands in it, returning the original
        frame with hand landmarks drawn on it plus the raw landmark data.
        
        Args:
            frame: A video frame in BGR format (that's what OpenCV gives you)
            
        Returns:
            tuple: (modified_frame_with_hands_drawn, list_of_hand_landmarks, raw_results)
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
        Pull out the coordinates from a detected hand.
        
        MediaPipe gives you 21 points per hand (fingers, palm, wrist, etc.).
        Each point has x, y, z coordinates. This extracts all of them and normalizes
        them so hand size/distance doesn't matter as much.
        
        Args:
            hand_landmarks: The hand landmarks object from MediaPipe
            
        Returns:
            list: 63 numbers (21 points × 3 coordinates) - this is what the ML model wants
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
