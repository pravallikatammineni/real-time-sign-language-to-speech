# How to Run This Thing

Here's everything you need to know to get this project up and running. It's not complicated, I promise.

## Before You Start

- Python 3.8 or higher (I'm on 3.10)
- A webcam
- Speakers or headphones (so you can hear the output)

## Setup

### Get the Code
```bash
git clone https://github.com/yourusername/real-time-sign-language-to-speech.git
cd real-time-sign-language-to-speech
```

### Create a Virtual Environment
I do this to keep my system clean and avoid package conflicts.

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Everything
```bash
pip install -r requirements.txt
```

Yeah, it's a bunch of packages. It'll take a minute or two. OpenCV is big.

## Running the Project

### Step 1: Record Some Gestures
```bash
python src/app.py
```

This opens a window with your webcam feed. Now:
- **Press A** - Record a gesture you'll label "A"
- **Press B** - Record a gesture you'll label "B"  
- **Press C** - Record a gesture you'll label "C"
- **Press S** - Save everything to the CSV file
- **Press Q** - Quit

**Real talk:** Record at least 30 samples per gesture if you want decent accuracy. I usually do 50. Light matters too—if your room is dark, MediaPipe will struggle to find your hands.

The data gets saved to `data/gesture_dataset.csv`. You can see it's just a bunch of numbers—those are the 21 hand landmarks * 3 coordinates each = 63 numbers per gesture.

### Step 2: Train the Model
```bash
python src/train_model.py
```

Takes your gesture data and trains a Random Forest classifier. The output tells you:
- How accurate the model is on test data
- If it's terrible, you probably need more training data

The model gets saved to `model/gesture_model.pkl` and that's what the prediction script uses.

### Step 3: See It In Action
```bash
python src/predict.py
```

Now make the same gestures you recorded. The model recognizes them and speaks them out loud. There's also a video window showing what it's seeing.

**Press Q to stop.**

That's the whole workflow. Done.

## Quick Copy-Paste Workflow

If you just want to get going without thinking:

```bash
# Activate your virtual environment
venv\Scripts\activate  # Windows
# OR
source venv/bin/activate  # Mac/Linux

# Record some gestures (make A, B, C hand signs, press A/B/C to record them)
python src/app.py

# Train the model on what you recorded
python src/train_model.py

# See the magic happen (make your gestures at the camera)
python src/predict.py
```

That's literally it.

## What Each File Does

### app.py
Starts a webcam window and lets you record hand gestures. That's it. Simple.
- **Input:** Your hand (via webcam)
- **Output:** CSV file with gesture data

### train_model.py
Takes your gesture data and trains a machine learning model to recognize them.
- **Input:** `data/gesture_dataset.csv`
- **Output:** `model/gesture_model.pkl` and accuracy metrics

### predict.py
The money shot. Uses your trained model to recognize gestures in real-time and speak them.
- **Input:** Webcam feed + trained model
- **Output:** On-screen labels + spoken words

### hand_detection.py
A helper module that wraps up all the MediaPipe hand detection stuff. You probably won't call this directly, but it's there.

### gesture_recognition.py
Another helper module that handles loading the model and making predictions. Also there in the background.

### text_to_speech.py
Handles all the speech stuff. Wraps pyttsx3 in a cleaner interface.

## Troubleshooting

### It says "No module named 'X'"
```bash
pip install -r requirements.txt
```
Just reinstall everything. Fixes 90% of these issues.

### Webcam won't open
Test it first:
```bash
python -c "import cv2; cap = cv2.VideoCapture(0); print('Works!' if cap.isOpened() else 'Nope')"
```
If that fails, your camera might be on index 1 or 2 instead of 0. Try changing `VideoCapture(0)` to `VideoCapture(1)` in the code.

### "Model not found" error
You forgot to train it. Run:
```bash
python src/train_model.py
```
Your gesture data has to exist first (from running app.py).

### No sound coming out
- Check your volume (seriously)
- Make sure speakers/headphones are connected
- Try this to test: `python -c "import pyttsx3; e = pyttsx3.init(); e.say('test'); e.runAndWait()"`

### Accuracy is terrible
Probably not enough training data. 20 samples is basically useless. Try 50+ per gesture. Also matters:
- Good lighting
- Consistent gesture definitions
- Recording from different distances

### Everything's running slow
MediaPipe is CPU-heavy. Try:
- Closing other programs
- Reducing detection confidence (make it easier, faster)
- Using GPU if you have one
- Lowering camera resolution

## Random Useful Stuff

### Make It Faster
- More training data = better accuracy but slower training
- Fewer samples = trains faster, predicts less accurately
- Adjust `min_detection_confidence` in hand_detection.py if it's too slow

### Tweak the Voice
- Change speech rate: `tts = TextToSpeech(rate=200)` (higher = faster)
- Adjust volume: `tts.set_volume(0.5)` (0 = silent, 1 = max)
- Different voices: depends on your system

### Add More Gestures
Not limited to A, B, C. In app.py, you can add more key presses (D, E, F, etc.). Just make sure to record samples and retrain the model.

## Project Layout

```
real-time-sign-language-to-speech/
├── src/
│   ├── app.py                    # Record gestures
│   ├── train_model.py            # Train classifier
│   ├── predict.py                # Use the model
│   ├── hand_detection.py         # Hand tracking stuff
│   ├── gesture_recognition.py    # Model stuff
│   └── text_to_speech.py         # Audio stuff
├── data/
│   └── gesture_dataset.csv       # Your gesture data
├── model/
│   └── gesture_model.pkl         # The trained model
├── requirements.txt              # What to pip install
└── README.md                     # This whole project explained
```

## Final Thoughts

This isn't magic. It's just:
1. Detecting hands (MediaPipe)
2. Converting hand position to numbers (landmarks)
3. Running those numbers through a trained model (scikit-learn)
4. Speaking the result (pyttsx3)

Everything is straightforward. If something breaks, Google the error message. Stack Overflow has answers for basically everything.
