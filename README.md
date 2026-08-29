# Real-Time Sign Language to Speech 🤟🔊

A real-time AI system that translates sign language gestures into speech using computer vision and machine learning. This project uses hand detection with MediaPipe, gesture recognition with scikit-learn, and text-to-speech with pyttsx3.

## 🌟 Features

- **Real-Time Hand Detection**: Uses MediaPipe to detect hand landmarks from webcam feed
- **Gesture Recognition**: Machine learning model (Random Forest) classifies hand gestures
- **Text-to-Speech Conversion**: Automatically speaks out recognized gestures
- **Easy Data Collection**: Simple interface to record and label training data
- **Modular Architecture**: Well-organized code with reusable components
- **Cross-Platform Support**: Works on Windows, macOS, and Linux

## 🎯 How It Works

1. **Data Collection** (`app.py`): Records hand landmarks for different gestures (A, B, C)
2. **Model Training** (`train_model.py`): Trains a Random Forest classifier on collected data
3. **Real-Time Prediction** (`predict.py`): Detects hands, predicts gestures, and speaks them out

## 📋 Prerequisites

- Python 3.8 or higher
- Webcam or camera
- Microphone or speakers (for audio output)
- 4GB RAM minimum
- 500MB storage for dependencies

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/real-time-sign-language-to-speech.git
cd real-time-sign-language-to-speech
```

### 2. Set Up Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Project

**Step 1: Collect gesture data**
```bash
python src/app.py
```
- Press **A**, **B**, **C** to record gestures
- Press **S** to save dataset
- Press **Q** to quit

**Step 2: Train the model**
```bash
python src/train_model.py
```

**Step 3: Run live gesture recognition**
```bash
python src/predict.py
```

For detailed commands and troubleshooting, see [COMMANDS.md](COMMANDS.md)

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

## 📚 Module Documentation

### `hand_detection.py`
Provides hand detection and landmark extraction utilities.

**Key Class:**
- `HandDetector`: Detects hand landmarks from video frames

**Example:**
```python
from hand_detection import HandDetector

detector = HandDetector()
frame, landmarks, results = detector.detect(video_frame)
if landmarks:
    landmark_list = detector.extract_landmarks(landmarks[0])
```

### `gesture_recognition.py`
Provides gesture recognition and model handling utilities.

**Key Class:**
- `GestureRecognizer`: Predicts gesture from hand landmarks

**Example:**
```python
from gesture_recognition import GestureRecognizer

recognizer = GestureRecognizer("model/gesture_model.pkl")
gesture = recognizer.predict(landmark_list)
probabilities = recognizer.predict_proba(landmark_list)
```

### `text_to_speech.py`
Provides text-to-speech conversion utilities.

**Key Class:**
- `TextToSpeech`: Converts text to speech

**Example:**
```python
from text_to_speech import TextToSpeech

tts = TextToSpeech(rate=150)
tts.speak("Hello")
tts.set_rate(200)
tts.close()
```

## 🎓 Workflow Example

```python
import cv2
from hand_detection import HandDetector
from gesture_recognition import GestureRecognizer
from text_to_speech import TextToSpeech

# Initialize components
detector = HandDetector()
recognizer = GestureRecognizer("model/gesture_model.pkl")
tts = TextToSpeech()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect hands
    frame, landmarks, _ = detector.detect(frame)
    
    # Recognize gesture
    if landmarks:
        landmark_list = detector.extract_landmarks(landmarks[0])
        gesture = recognizer.predict(landmark_list)
        
        # Speak the gesture
        if gesture:
            tts.speak(gesture)
    
    # Display
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
tts.close()
```

## 🎮 Usage Tips

### Improving Accuracy
- Collect 50-100 samples per gesture class
- Record gestures under consistent lighting
- Vary hand positions, angles, and distances
- Use different hand sizes (close and far from camera)

### Performance Optimization
- Reduce detection confidence threshold for faster processing
- Lower camera resolution
- Use GPU acceleration if available

### Customization
- Add more gesture classes (D, E, F, etc.)
- Try different ML models (SVM, Neural Networks)
- Adjust TTS voice and speed
- Implement gesture smoothing for stability

## 🐛 Troubleshooting

### Webcam Issues
```bash
# Test if camera is accessible
python -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened())"
```

### Import Errors
```bash
# Reinstall all dependencies
pip install --upgrade -r requirements.txt
```

### Model Not Found
- Ensure you've trained the model (run `train_model.py`)
- Check that `model/gesture_model.pkl` exists

### No Audio Output
- Check system volume and speaker connections
- Verify pyttsx3 installation: `python -c "import pyttsx3; pyttsx3.init().say('test'); pyttsx3.init().runAndWait()"`

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

## 🙏 Acknowledgments

- [MediaPipe](https://mediapipe.dev/) - Hand detection framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [pyttsx3](https://pyttsx3.readthedocs.io/) - Text-to-speech library

## 📞 Contact & Support

For questions, issues, or suggestions:
- Open an GitHub issue
- Check [COMMANDS.md](COMMANDS.md) for detailed instructions
- Review module docstrings for API details

## 🚀 Future Enhancements

- [ ] Support for continuous sign language sentences
- [ ] Multi-hand gesture recognition
- [ ] Web interface for easier interaction
- [ ] Real-time model training and adaptation
- [ ] Support for different sign languages
- [ ] GPU acceleration
- [ ] Mobile app deployment
- [ ] Advanced gesture smoothing and filtering

---

**Made with ❤️ for accessibility and AI**
