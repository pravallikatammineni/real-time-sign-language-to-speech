# Project Commands

Quick reference guide for running the Real-Time Sign Language to Speech project.

## Prerequisites

- Python 3.8+
- Webcam/Camera for real-time hand detection
- Microphone or speakers for text-to-speech output

## Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/real-time-sign-language-to-speech.git
cd real-time-sign-language-to-speech
```

### 2. Create Virtual Environment (Optional but Recommended)

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

## Running the Project

### Step 1: Collect Gesture Data (Data Collection Mode)
```bash
python src/app.py
```

**Instructions:**
- A webcam window will open showing hand detection
- Press **A**, **B**, or **C** to record gestures with corresponding labels
- Record multiple samples of each gesture for better accuracy
- Press **S** to save the collected dataset to `data/gesture_dataset.csv`
- Press **Q** to quit

**Tips:**
- Ensure good lighting for better hand detection
- Keep your hand clearly visible in the frame
- Record at least 20-30 samples per gesture for good model training

### Step 2: Train the Model
```bash
python src/train_model.py
```

**Output:**
- Trains a Random Forest Classifier on your dataset
- Displays model accuracy on test set
- Saves the trained model to `model/gesture_model.pkl`

**Note:**
- Requires `data/gesture_dataset.csv` to exist
- Requires at least some data samples (from Step 1)

### Step 3: Run Real-Time Gesture Recognition & Speech
```bash
python src/predict.py
```

**Features:**
- Real-time hand gesture detection
- Converts recognized gestures to speech
- Displays the current gesture prediction on screen
- Press **Q** to quit

**Output:**
- Text-to-speech output for each recognized gesture
- Live video feed with gesture labels

## Full Workflow Example

```bash
# 1. Activate virtual environment
venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # macOS/Linux

# 2. Collect gesture data
python src/app.py
# (Record A, B, C gestures, then press S to save, Q to quit)

# 3. Train the model
python src/train_model.py
# (Watch for accuracy metrics)

# 4. Run live prediction
python src/predict.py
# (Make gestures to see predictions and hear speech)
```

## Module Documentation

### `src/app.py`
- **Purpose**: Data collection from webcam
- **Records**: Hand landmarks for gestures labeled A, B, or C
- **Output**: `data/gesture_dataset.csv`

### `src/train_model.py`
- **Purpose**: Train gesture recognition model
- **Input**: `data/gesture_dataset.csv`
- **Output**: `model/gesture_model.pkl`
- **Algorithm**: Random Forest Classifier

### `src/predict.py`
- **Purpose**: Real-time gesture recognition with speech output
- **Input**: Webcam feed
- **Output**: Text-to-speech and on-screen display
- **Requirements**: Trained model (`model/gesture_model.pkl`)

### `src/hand_detection.py`
- **Purpose**: Hand detection utilities
- **Main Class**: `HandDetector`
- **Features**: Hand landmark extraction, frame visualization

### `src/gesture_recognition.py`
- **Purpose**: Gesture recognition utilities
- **Main Class**: `GestureRecognizer`
- **Features**: Model prediction, probability estimation

### `src/text_to_speech.py`
- **Purpose**: Text-to-speech utilities
- **Main Class**: `TextToSpeech`
- **Features**: Voice control, rate adjustment, voice selection

## Troubleshooting

### Webcam not detected
```bash
# Check available cameras (usually 0 is the default)
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Module import errors
```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Model not found
- Ensure you've completed Step 2 (training)
- Check that `model/gesture_model.pkl` exists

### No sound output
- Check system volume settings
- Verify speakers/headphones are connected
- Test pyttsx3 independently:
  ```bash
  python -c "import pyttsx3; pyttsx3.init().say('Hello'); pyttsx3.init().runAndWait()"
  ```

## System Requirements

| Component | Requirement |
|-----------|------------|
| Python | 3.8+ |
| RAM | 4GB+ |
| CPU | Dual-core minimum |
| GPU | Optional (faster processing) |
| Webcam | Required |
| Storage | 500MB+ for dependencies |

## Project Structure

```
real-time-sign-language-to-speech/
├── src/
│   ├── app.py                    # Data collection
│   ├── train_model.py            # Model training
│   ├── predict.py                # Live prediction
│   ├── hand_detection.py         # Hand detection module
│   ├── gesture_recognition.py    # Gesture recognition module
│   └── text_to_speech.py         # Text-to-speech module
├── data/
│   └── gesture_dataset.csv       # Collected gesture data
├── model/
│   └── gesture_model.pkl         # Trained model
├── requirements.txt              # Project dependencies
├── README.md                     # Project documentation
└── COMMANDS.md                   # This file
```

## Performance Tips

1. **Improve Accuracy**:
   - Collect more training samples (100+ per gesture)
   - Record under consistent lighting
   - Vary hand positions and angles

2. **Faster Processing**:
   - Lower camera resolution in app.py
   - Use fewer max_num_hands in hand detection
   - Reduce min_detection_confidence

3. **Better Speech**:
   - Adjust rate in TextToSpeech (default: 150 WPM)
   - Select different voices with `set_voice()`
   - Adjust volume with `set_volume()`

## Contributing

Feel free to improve the project by:
- Adding more gesture classes
- Implementing different ML models
- Improving hand detection accuracy
- Adding more language support for TTS

## License

This project is open source. See LICENSE file for details.
