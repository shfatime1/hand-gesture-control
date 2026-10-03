# Hand Gesture Control

Control a demo with your hand gestures. Two versions:

- **Web demo**: runs in your browser, no installation, no sign-up.
- **Desktop app**: really controls your computer (system volume, slides, mouse).

**Live web demo:** https://shfatime1.github.io/hand-gesture-control/

## What it does

- Recognizes hand gestures in real time from your webcam.
- **Media / Slides mode:** gestures change the volume and flip through slides.
- **Mouse mode:** your index fingertip moves a pointer and a thumb-index pinch acts as a click.
- An **activation gesture** (hold an open palm for 1.5 s) prevents accidental commands. Control switches off automatically after 8 s of inactivity.

## Gestures

| Gesture | Media / Slides mode |
|---|---|
| Open palm (hold 1.5 s) | Activate control |
| Thumb up | Volume up |
| Thumb down | Volume down |
| Victory | Next slide |
| Pointing up | Previous slide |
| OK | Detected by a custom rule (see below) |

In **Mouse mode**: move your index finger to move the pointer, touch thumb and index finger together to click.

## Web demo

Open the live link, press **Start**, allow the camera and turn your speakers on. A built-in music loop plays so you can hear the volume change, and a small presentation shows the slide control. Inside a browser page, actions only affect the page itself, because browsers cannot control your operating system. The camera feed is processed locally and never uploaded.

## Desktop app

The desktop app uses the same gesture model but sends real key presses and mouse movements with `pyautogui`.

**Requirements:** Python 3.9+ with Tkinter (included with the standard Python installers on Windows and macOS; on Linux install `python3-tk`), and a webcam.

```bash
git clone https://github.com/shfatime1/hand-gesture-control.git
cd hand-gesture-control
pip install -r requirements.txt
python gui_app.py
```

The model file (`gesture_recognizer.task`, about 8 MB) is downloaded automatically on the first run.

How to use:

1. Press **Start** (choose another camera number if the default does not open).
2. Press the **Control** button so it turns green (control ON).
3. Hold an open palm for 1.5 s until the status says **ACTIVE**.
4. Show a gesture. Volume keys, left/right arrow keys (slides) or the mouse are controlled.

Notes:

- The gesture to key mapping and timings are stored in `config.json`, which is created on the first run. Edit it to change what each gesture does, for example `"Open_Palm": ["space", 1.5, "Pause"]`.
- **Safety:** move your real mouse to the top-left corner of the screen and control switches off.
- On macOS the system volume keys sent by `pyautogui` may not work, while slide control (arrow keys) does.

## How it works

1. The webcam frame is passed to the pre-trained **MediaPipe Gesture Recognizer**, which finds the hand and its 21 landmarks and classifies the gesture.
2. A majority vote over the last 7 frames removes jitter.
3. The OK sign is not part of the built-in model, so it is detected with a simple geometric rule on the landmarks: thumb tip and index tip close together, the other three fingers extended.
4. In mouse mode, the index fingertip position is smoothed and mapped to the screen, and the pinch distance relative to the palm size triggers a click.

No model training was needed; the model is provided by MediaPipe.

## Tech

- Web: JavaScript, HTML, CSS (single `index.html`), MediaPipe Tasks Vision, Web Audio API
- Desktop: Python, MediaPipe, OpenCV, Tkinter, Pillow, PyAutoGUI

## Limitations

- Works best with good lighting and the hand about 40-60 cm from the camera.
- Similar gestures such as open palm and "stop" can occasionally be confused.

## Credits

Gesture recognition by [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache 2.0).
