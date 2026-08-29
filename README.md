# Real-Time Sign Language to Speech 🤟🔊

Hey! This project started as an idea to make communication more accessible. It translates hand gestures into spoken words using your webcam in real-time. Pretty cool stuff if I do say so myself.

The tech stack is straightforward: MediaPipe handles hand detection, scikit-learn trains the gesture classifier, and pyttsx3 does the speaking. Nothing overly complicated, just practical tools working together.

## What You Can Do With This

- **Detect Hands Live**: Grabs hand position and landmarks from your webcam using MediaPipe
- **Recognize Gestures**: A trained Random Forest model figures out which gesture you're making
- **Speak It Out**: Automatically converts recognized gestures to voice output
- **Train With Your Data**: Collect your own gesture samples and build a custom model
- **Easy to Extend**: Code is organized so you can add more gestures or swap in different ML models
- **Works Everywhere**: Windows, Mac, Linux—it'll run anywhere Python does

## How It Actually Works

The workflow is pretty straightforward:

1. **You record some gestures** (`app.py`) - Point your webcam at your hand and press A/B/C to record samples
2. **Train the model** (`train_model.py`) - Feed those samples into a Random Forest classifier
3. **Start predicting** (`predict.py`) - Make gestures and watch it recognize them and speak out loud

That's it. No black magic here.

## What You'll Need

- Python 3.8+ (I'm using 3.10, works great)
- A webcam (obviously)
- Speakers or headphones (for the speech output)
- 4GB RAM minimum (more is better for smooth processing)
- About 500MB of disk space for all the dependencies

## Getting Started

### Clone It
```bash
git clone https://github.com/yourusername/real-time-sign-language-to-speech.git
cd real-time-sign-language-to-speech
```

### Set Up Your Environment
I always use a virtual environment to keep things clean. Here's what I do:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

This grabs everything you need. It might take a couple minutes—OpenCV and MediaPipe are pretty hefty.

### Run It

**First, collect some gesture data:**
```bash
python src/app.py
```
A window pops up showing your webcam. Press A, B, or C to record different gestures. I usually record 30-50 samples per gesture to get decent accuracy. Hit S to save your dataset, Q to exit.

**Then, train the model:**
```bash
python src/train_model.py
```
Feeds your gesture data into a Random Forest classifier. It spits out accuracy metrics so you can see how well it's learning.

**Finally, try it out:**
```bash
python src/predict.py
```
This is the fun part. Make gestures at your camera and hear them spoken back. Press Q to quit.

For more details and troubleshooting, check out [COMMANDS.md](COMMANDS.md) for the full guide.

## 📁 Project Structure

```
real-time-sign-language-to-speech/
│
├── src/
│   ├── app.py                    # 📹 Data collection from webcam
│   ├── train_model.py            # 🤖 Model training script
│   ├── predict.py                # 🎤 Real-time prediction & speech
│   ├── hand_detection.py         # 🖐️ Hand detection utilities
│   ├── gesture_recognition.py    # 🔍 Gesture recognition utilities
│   └── text_to_speech.py         # 🔊 Text-to-speech utilities
│
├── data/
│   └── gesture_dataset.csv       # 📊 Collected training data
│
├── model/
│   └── gesture_model.pkl         # 🧠 Trained ML model
│
├── requirements.txt              # 📦 Python dependencies
├── README.md                     # 📖 Project documentation
├── COMMANDS.md                   # ⌨️ Command reference guide
└── .gitignore                    # 🚫 Git ignore rules
```

## 🔧 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.13.0.92 | Video capture and image processing |
| mediapipe | 1.0.1 | Hand landmark detection |
| pandas | 3.0.5 | Data handling and CSV operations |
| scikit-learn | 1.7.2 | Machine learning (Random Forest) |
| numpy | 2.2.6 | Numerical computations |
| pyttsx3 | 2.90 | Text-to-speech |
| joblib | 1.5.3 | Model serialization |

## The Code, Explained Simply

### hand_detection.py
Handles everything about finding your hands in the video. It uses MediaPipe, which honestly does most of the heavy lifting. I wrapped it in a `HandDetector` class to make it cleaner to use.

**What it does:**
- Grabs frames from your webcam
- Finds hand landmarks (all 21 points of your hand)
- Normalizes coordinates so they're consistent regardless of hand size or distance

**Quick usage:**
```python
from hand_detection import HandDetector

detector = HandDetector()
frame, landmarks, results = detector.detect(video_frame)
if landmarks:
    landmark_list = detector.extract_landmarks(landmarks[0])
    print(f"Got {len(landmark_list)} coordinates")
```

### gesture_recognition.py
This is where the magic happens. Takes those 63 hand coordinates and figures out which gesture it is using a trained Random Forest model. Pretty straightforward.

**The class:**
- Loads a trained model from disk
- Takes landmark coordinates and predicts the gesture
- Can give you confidence scores if you want them

**Example:**
```python
from gesture_recognition import GestureRecognizer

recognizer = GestureRecognizer("model/gesture_model.pkl")
gesture = recognizer.predict(landmark_list)  # Returns "A", "B", "C", etc.
confidence = recognizer.predict_proba(landmark_list)  # Get scores for each class
```

### text_to_speech.py
Converts text (or in our case, gesture labels) to actual speech. Uses pyttsx3, which works offline—no internet required.

**Features:**
- Speak text out loud
- Adjust speed and volume
- Switch between different voices if your system has them
- Tracks what was last spoken so you don't repeat yourself constantly

**Simple example:**
```python
from text_to_speech import TextToSpeech

tts = TextToSpeech(rate=150)  # Rate is words per minute
tts.speak("A")  # Speaks out the letter
tts.set_volume(0.8)  # Set volume between 0-1
```

## Putting It All Together

Here's a real example of how all the pieces work together. This is basically what `predict.py` does:

```python
import cv2
from hand_detection import HandDetector
from gesture_recognition import GestureRecognizer
from text_to_speech import TextToSpeech

# Set things up
detector = HandDetector()
recognizer = GestureRecognizer("model/gesture_model.pkl")
tts = TextToSpeech()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Find hands in the frame
    frame, landmarks, _ = detector.detect(frame)
    
    # If we see a hand, try to recognize it
    if landmarks:
        landmark_list = detector.extract_landmarks(landmarks[0])
        gesture = recognizer.predict(landmark_list)
        
        # If recognized, speak it
        if gesture:
            tts.speak(gesture)
            # Show it on screen
            cv2.putText(frame, f'Gesture: {gesture}', (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

That's really all there is to it.

## Tips & Tricks I've Learned

### Getting Better Accuracy
- More data is everything. I started with 20 samples per gesture and got pretty poor accuracy. After bumping it up to 50+ per gesture, things improved dramatically.
- Good lighting helps a ton. If your room is dark, MediaPipe struggles to find hands reliably.
- Record gestures at different distances from the camera. If you only train on close-up hand shots, it won't recognize gestures made at arm's length.
- Be consistent with your gesture definitions. Make sure each gesture variation is distinct enough that even you can tell them apart.

### Making It Faster
- The hand detection is the slowest part. If you're running on older hardware, try lowering the detection confidence threshold (make it easier to find hands, faster detection).
- You can reduce camera resolution in the capture loop if you don't need high quality.
- MediaPipe will use GPU if available, which is noticeably faster.

### Customizing the Audio
- I find a rate of 150 WPM (words per minute) sounds natural. Some people prefer faster or slower—experiment.
- You can set different voices with `set_voice()` if your system has multiple voices installed.
- Adjust volume with `set_volume()` if the default is too loud or quiet.

### Adding More Gestures
- Not limited to A, B, C. You can add D, E, F, whatever. Just record them in `app.py` and retrain.
- The model will automatically learn however many classes you give it.

## When Things Go Wrong (Troubleshooting)

### Camera not working
First, check if your system even sees the camera:
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera works!' if cap.isOpened() else 'Camera not found')"
```
If that fails, you might have a different camera ID (try 1, 2, etc. instead of 0).

### Import errors
Just reinstall everything:
```bash
pip install --upgrade -r requirements.txt
```
Sometimes packages get corrupted or there's a version conflict.

### "Model not found" error
You forgot to train it first:
```bash
python src/train_model.py
```
The model file has to exist before you can use it for predictions.

### No sound
- Check your system volume (seems obvious, but I've been there)
- Make sure speakers/headphones are actually connected
- Quick test: `python -c "import pyttsx3; e = pyttsx3.init(); e.say('test'); e.runAndWait()"`

### Bad accuracy
More likely than not, you just need more training data. 20 samples per gesture is usually not enough. Try 50+.

### Everything's slow
MediaPipe can be resource-heavy. If you're on an older machine, try:
- Closing other programs
- Lowering camera resolution
- Reducing `max_num_hands` to 1 (already set, but keep it there)

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs and issues
- Suggest new features
- Submit pull requests
- Improve documentation

### Areas for Improvement
- Support for more gesture classes
- Real-time model improvement during usage
- GUI interface
- Multi-hand gesture recognition
- Sign language phrase recognition

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## Credits & Thanks

This project stands on the shoulders of giants:
- **MediaPipe** - Seriously, hand detection would be a nightmare without it. Their pre-trained models are incredible.
- **OpenCV** - The backbone for all the video processing. It just works.
- **scikit-learn** - Simple, elegant machine learning. Random Forest was perfect for this.
- **pyttsx3** - Text-to-speech that doesn't need internet. Love it.

## Questions or Ideas?

If you find a bug, have a suggestion, or just want to chat about the project:
- Open an issue on GitHub
- Check [COMMANDS.md](COMMANDS.md) if you're stuck
- Read the docstrings in the code—they're pretty detailed

## What's Next?

Some ideas I've been thinking about:
- Support for continuous sign language (instead of single gestures)
- Real-time model refinement as you use it
- A simple web interface
- Mobile app version
- More gesture classes
- Better gesture smoothing to reduce jitter

Feel free to fork and build on this. That's what it's here for.

---

Made with coffee ☕ and hopefully some good vibes
