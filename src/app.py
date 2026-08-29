import cv2
import numpy as np

# Simple webcam gesture app
# This version is easy to understand and keeps the project small.

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera not found. Please connect a webcam and run the app again.")
    raise SystemExit

lower_skin = np.array([0, 30, 60], dtype=np.uint8)
upper_skin = np.array([25, 173, 255], dtype=np.uint8)
kernel = np.ones((5, 5), np.uint8)


def detect_gesture(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "No hand"

    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    if area < 2000:
        return "No hand"

    hull = cv2.convexHull(contour, returnPoints=False)
    if len(hull) < 5:
        return "FIST"

    defects = cv2.convexityDefects(contour, hull)
    fingers = 0

    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            if d > 12000:
                fingers += 1

    if fingers >= 3:
        return "OPEN HAND"
    if fingers == 2:
        return "PEACE"
    if fingers == 1:
        return "ONE FINGER"
    return "FIST"


print("Simple Hand Gesture App")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    roi = frame[60:420, 60:420]

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    gesture = detect_gesture(mask)

    cv2.rectangle(frame, (60, 60), (420, 420), (0, 255, 0), 2)
    cv2.putText(frame, gesture, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "Press Q to quit", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("App closed")