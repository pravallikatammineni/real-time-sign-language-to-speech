import cv2
import mediapipe as mp
import pandas as pd
import sys

try:
    # Import MediaPipe solutions
    from mediapipe import solutions
    from mediapipe.framework.formats import landmark_pb2
    mp_hands = solutions.hands
    mp_draw = solutions.drawing_utils
except ImportError as e:
    print(f"Error importing MediaPipe: {e}")
    print("Please run: pip install mediapipe --upgrade")
    sys.exit(1)

dataset = []

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

print("Press A/B/C to record gesture")
print("Press S to save dataset")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    landmark_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x - wrist.x)
                landmark_list.append(lm.y - wrist.y)
                landmark_list.append(lm.z - wrist.z)

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('a') and len(landmark_list) == 63:
        dataset.append(landmark_list + ["A"])
        print("Recorded A")

    if key == ord('b') and len(landmark_list) == 63:
        dataset.append(landmark_list + ["B"])
        print("Recorded B")

    if key == ord('c') and len(landmark_list) == 63:
        dataset.append(landmark_list + ["C"])
        print("Recorded C")

    if key == ord('s'):
        df = pd.DataFrame(dataset)
        df.to_csv("data/gesture_dataset.csv", index=False)
        print("Dataset saved!")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()