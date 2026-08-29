import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import sys
import os

# Check if model exists
if not os.path.exists("model/gesture_model.pkl"):
    print("Error: Model not found. Please run train_model.py first.")
    sys.exit(1)

try:
    # Load model
    model = joblib.load("model/gesture_model.pkl")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

try:
    # Initialize speech engine
    engine = pyttsx3.init()
except Exception as e:
    print(f"Error initializing text-to-speech: {e}")
    sys.exit(1)

last_prediction = ""

try:
    # Import MediaPipe solutions
    from mediapipe import solutions
    mp_hands = solutions.hands
    mp_draw = solutions.drawing_utils
except ImportError as e:
    print(f"Error importing MediaPipe: {e}")
    print("Please run: pip install mediapipe --upgrade")
    sys.exit(1)

try:
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7
    )
except Exception as e:
    print(f"Error initializing hand detector: {e}")
    sys.exit(1)

# Open camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open camera. Check if camera is connected.")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    prediction = ""

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            landmark_list = []

            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x - wrist.x)
                landmark_list.append(lm.y - wrist.y)
                landmark_list.append(lm.z - wrist.z)

            if len(landmark_list) == 63:
                data = np.array(landmark_list).reshape(1, -1)
                prediction = model.predict(data)[0]

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    # 🔊 Speak only if gesture changes
    if prediction != "" and prediction != last_prediction:
        engine.say(prediction)
        engine.runAndWait()
        last_prediction = prediction

    # Display prediction
    cv2.putText(frame, f'Gesture: {prediction}', (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()