import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

# For drawing landmarks
mp_draw = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR (OpenCV) to RGB (MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame and detect hands
    results = hands.process(rgb_frame)

    # If a hand is detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmark_list = []

            # Wrist landmark (reference point)
            wrist = hand_landmarks.landmark[0]

            for lm in hand_landmarks.landmark:
                landmark_list.append(lm.x - wrist.x)
                landmark_list.append(lm.y - wrist.y)
                landmark_list.append(lm.z - wrist.z)

            print("Total values:", len(landmark_list))

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
