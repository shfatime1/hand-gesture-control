"""
Hand Gesture Control - desktop app (Tkinter interface)

Uses the pre-trained MediaPipe Gesture Recognizer (no training needed).
OpenCV is only used for the camera stream; the interface is built with Tkinter.

Features:
  * Activation gesture: commands are ignored until you hold an open palm for 1.5 s
    (prevents accidental commands). Control switches off again after a period of inactivity.
  * Modes: "Media / Slides" (gesture -> key press) and "Mouse"
    (index fingertip = pointer, thumb + index pinch = click).
  * config.json: edit the gesture-to-key mapping and timings without touching the code.

Install:  pip install opencv-python mediapipe numpy pillow pyautogui
Run:      python gui_app.py
Model:    gesture_recognizer.task is downloaded automatically on the first run
Safety:   move the mouse to the top-left corner of the screen -> pyautogui stops control.
"""
import json
import os
import time
import urllib.request
from collections import Counter, deque
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

try:
    import pyautogui          # presses keys / moves the mouse for us
    pyautogui.PAUSE = 0       # no delay after each call (keeps the pointer smooth)
    FAILSAFE_ERR = pyautogui.FailSafeException
except ImportError:
    pyautogui = None
    FAILSAFE_ERR = ()        # empty tuple: "except" catches nothing

MODEL = "gesture_recognizer.task"
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
             "gesture_recognizer/float16/1/gesture_recognizer.task")
CONFIG_FILE = "config.json"
VIDEO_W, VIDEO_H = 640, 480

# colors
BG, PANEL, FG = "#1e1f26", "#272936", "#e8e8ee"
ACCENT, MUTED, DANGER, AMBER = "#4cc38a", "#8b8fa3", "#e5534b", "#e0a82e"
BTN = "#3a3d4f"

MODE_MEDIA, MODE_MOUSE = "Media / Slides", "Mouse"

NAMES = {
    "Closed_Fist": "Closed fist",
    "Open_Palm": "Open palm",
    "Pointing_Up": "Pointing up",
    "Thumb_Down": "Thumb down",
    "Thumb_Up": "Thumb up",
    "Victory": "Victory",
    "ILoveYou": "I love you",
    "OK": "OK",
    "None": "-",
}

DEFAULT_CONFIG = {
    "arm_hold_sec": 1.5,       # how long the open palm must be held to activate
    "arm_timeout_sec": 8,      # deactivate after this many seconds without activity
    "actions": {               # gesture: [key, cooldown before repeating (s), description]
        "Thumb_Up":    ["volumeup",   0.4, "Volume up"],
        "Thumb_Down":  ["volumedown", 0.4, "Volume down"],
        "Victory":     ["right",      1.5, "Next slide"],
        "Pointing_Up": ["left",       1.5, "Previous slide"],
    },
}


def load_config():
    """Read config.json if it exists, otherwise create it with the defaults."""
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, encoding="utf-8") as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except Exception as e:
            print("Could not read config.json, using defaults:", e)
    else:
        try:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(DEFAULT_CONFIG, f, ensure_ascii=False, indent=2)
        except OSError:
            pass
    return DEFAULT_CONFIG


CONFIG = load_config()
ACTIONS = {g: tuple(v) for g, v in CONFIG["actions"].items()}
ARM_HOLD = float(CONFIG["arm_hold_sec"])
ARM_TIMEOUT = float(CONFIG["arm_timeout_sec"])

# mouse mode parameters
MOUSE_MARGIN = 0.15     # unused border of the camera frame (makes screen edges easy to reach)
MOUSE_SMOOTH = 0.35     # 0-1: higher = faster, lower = smoother
PINCH_RATIO = 0.25      # thumb-index distance / hand size
CLICK_COOLDOWN = 0.6

CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),
               (10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),
               (17,18),(18,19),(19,20),(0,17)]


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in landmarks]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def _pt(landmarks, i, shape):
    h, w = shape[:2]
    return np.array([landmarks[i].x * w, landmarks[i].y * h])


def pinch_ratio(landmarks, shape):
    """Distance between thumb tip (4) and index tip (8) divided by the hand size."""
    palm = np.linalg.norm(_pt(landmarks, 0, shape) - _pt(landmarks, 9, shape))
    return np.linalg.norm(_pt(landmarks, 4, shape) - _pt(landmarks, 8, shape)) / max(palm, 1e-6)


def is_ok(landmarks, shape):
    """OK rule: thumb and index tips are close, the other 3 fingers are extended."""
    if pinch_ratio(landmarks, shape) > 0.35:
        return False
    wrist = _pt(landmarks, 0, shape)
    for tip, pip in ((12, 10), (16, 14), (20, 18)):
        if np.linalg.norm(_pt(landmarks, tip, shape) - wrist) < \
           np.linalg.norm(_pt(landmarks, pip, shape) - wrist):
            return False
    return True


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Hand Gesture Control")
        self.configure(bg=BG)
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.cap = None
        self.recognizer = None
        self.running = False
        self.after_id = None
        self.votes = deque(maxlen=7)       # majority vote over recent frames reduces jitter
        self.last_stable = "None"
        self.last_action = {}
        self.last_frame = None
        self.last_ts = 0
        self.start_time = 0
        self.prev = time.time()
        self.fps = 0.0

        # activation and mouse state
        self.armed = False
        self.armed_until = 0.0
        self.palm_since = None
        self.mouse_pos = None
        self.pinching = False
        self.last_click = 0.0
        if pyautogui:
            self.screen_w, self.screen_h = pyautogui.size()
        else:
            self.screen_w, self.screen_h = self.winfo_screenwidth(), self.winfo_screenheight()

        self.actions_var = tk.BooleanVar(value=False)
        self.safe_var = tk.BooleanVar(value=True)
        self.mode_var = tk.StringVar(value=MODE_MEDIA)
        self.cam_var = tk.StringVar(value="0")
        self.gesture_var = tk.StringVar(value="-")
        self.conf_var = tk.StringVar(value="0%")
        self.arm_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready. Press “Start”.")

        self.build_ui()
        self.refresh_arm_label()

    # ------------------------------------------------------------------ interface
    def build_ui(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Conf.Horizontal.TProgressbar", troughcolor=BTN,
                        background=ACCENT, bordercolor=PANEL, lightcolor=ACCENT,
                        darkcolor=ACCENT)

        left = tk.Frame(self, bg=BG)
        left.grid(row=0, column=0, padx=(16, 8), pady=16, sticky="n")
        right = tk.Frame(self, bg=PANEL)
        right.grid(row=0, column=1, padx=(8, 16), pady=16, sticky="n")

        self.blank = ImageTk.PhotoImage(Image.new("RGB", (VIDEO_W, VIDEO_H), "black"))
        self.video = tk.Label(left, image=self.blank, bg="black", fg=MUTED,
                              text="Camera is off.\nPress “Start”.",
                              compound="center", font=("Segoe UI", 14))
        self.video.pack()

        pad = dict(padx=18, anchor="w")
        tk.Label(right, text="Hand Gesture Control", bg=PANEL, fg=FG,
                 font=("Segoe UI", 15, "bold")).pack(pady=(14, 0), **pad)
        tk.Label(right, text="MediaPipe Gesture Recognizer", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(pady=(0, 8), **pad)

        tk.Label(right, text="Detected gesture", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(**pad)
        tk.Label(right, textvariable=self.gesture_var, bg=PANEL, fg=ACCENT,
                 font=("Segoe UI", 24, "bold")).pack(pady=(0, 2), **pad)
        self.bar = ttk.Progressbar(right, style="Conf.Horizontal.TProgressbar",
                                   length=300, maximum=100)
        self.bar.pack(padx=18, pady=(0, 2))
        tk.Label(right, textvariable=self.conf_var, bg=PANEL, fg=FG,
                 font=("Segoe UI", 10)).pack(pady=(0, 8), **pad)

        # mode
        row = tk.Frame(right, bg=PANEL)
        row.pack(pady=(0, 6), **pad)
        tk.Label(row, text="Mode:", bg=PANEL, fg=FG,
                 font=("Segoe UI", 10)).pack(side="left")
        mode_box = ttk.Combobox(row, textvariable=self.mode_var, width=16,
                                values=[MODE_MEDIA, MODE_MOUSE], state="readonly")
        mode_box.pack(side="left", padx=6)
        mode_box.bind("<<ComboboxSelected>>", self.on_mode_change)

        # control and safety buttons
        self.chk = tk.Button(right, command=self.toggle_actions, bd=0, width=30,
                             font=("Segoe UI", 10, "bold"), cursor="hand2")
        self.chk.pack(pady=(0, 4), padx=18)
        self.safe_btn = tk.Button(right, command=self.toggle_safe, bd=0, width=30,
                                  font=("Segoe UI", 9), cursor="hand2")
        self.safe_btn.pack(pady=(0, 4), padx=18)
        self.refresh_actions_btn()
        self.refresh_safe_btn()

        self.arm_label = tk.Label(right, textvariable=self.arm_var, bg=PANEL, fg=MUTED,
                                  font=("Segoe UI", 10, "bold"))
        self.arm_label.pack(pady=(2, 6), **pad)

        # gesture -> action map
        for g, (_, _, desc) in ACTIONS.items():
            tk.Label(right, text=f"{NAMES.get(g, g)}  →  {desc}", bg=PANEL, fg=MUTED,
                     font=("Segoe UI", 9)).pack(padx=30, anchor="w")
        tk.Label(right, text="Mouse: index fingertip = pointer, pinch = click",
                 bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack(padx=30, anchor="w")

        # history
        tk.Label(right, text="Recent gestures", bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9)).pack(pady=(8, 2), **pad)
        self.history = tk.Listbox(right, height=5, bg=BG, fg=FG, bd=0,
                                  highlightthickness=0, font=("Consolas", 10),
                                  selectbackground=BG)
        self.history.pack(padx=18, fill="x")

        # camera selection and buttons
        row2 = tk.Frame(right, bg=PANEL)
        row2.pack(pady=(8, 4), **pad)
        tk.Label(row2, text="Camera:", bg=PANEL, fg=FG,
                 font=("Segoe UI", 10)).pack(side="left")
        ttk.Combobox(row2, textvariable=self.cam_var, values=["0", "1", "2"],
                     width=3, state="readonly").pack(side="left", padx=6)

        self.btn_start = tk.Button(right, text="Start", command=self.toggle,
                                   bg=ACCENT, fg="black", bd=0, width=30,
                                   font=("Segoe UI", 11, "bold"), cursor="hand2")
        self.btn_start.pack(pady=(2, 4))
        tk.Button(right, text="Save screenshot", command=self.screenshot,
                  bg=BTN, fg=FG, bd=0, width=30,
                  font=("Segoe UI", 10), cursor="hand2").pack(pady=(0, 4))

        tk.Label(right, textvariable=self.status_var, bg=PANEL, fg=MUTED,
                 font=("Segoe UI", 9), wraplength=300,
                 justify="left").pack(pady=(4, 12), **pad)

    # ----------------------------------------------------------- buttons and state
    def toggle_actions(self):
        if pyautogui is None:
            messagebox.showinfo("pyautogui is missing",
                                "Run this in a terminal:  pip install pyautogui\nThen restart the app.")
            return
        self.actions_var.set(not self.actions_var.get())
        self.armed = False
        self.palm_since = None
        self.refresh_actions_btn()
        self.status_var.set("Control is ON." if self.actions_var.get() else "Control is OFF.")

    def refresh_actions_btn(self):
        if pyautogui is None:
            self.chk.config(text="Control: pyautogui is missing", bg=BTN, fg=MUTED)
        elif self.actions_var.get():
            self.chk.config(text="Control: ON  (turn off)", bg=ACCENT, fg="black")
        else:
            self.chk.config(text="Control: OFF  (turn on)", bg=BTN, fg=FG)

    def toggle_safe(self):
        self.safe_var.set(not self.safe_var.get())
        self.armed = False
        self.palm_since = None
        self.refresh_safe_btn()

    def refresh_safe_btn(self):
        if self.safe_var.get():
            self.safe_btn.config(text=f"Activation gesture: ON (hold palm {ARM_HOLD:g} s)",
                                 bg=BTN, fg=FG)
        else:
            self.safe_btn.config(text="Activation gesture: OFF (always active)",
                                 bg=AMBER, fg="black")

    def on_mode_change(self, _event=None):
        self.mouse_pos = None
        self.pinching = False
        self.armed = False
        self.status_var.set(f"Mode: {self.mode_var.get()}")

    def refresh_arm_label(self, now=None):
        now = now or time.time()
        if not self.actions_var.get():
            self.arm_var.set("● Control is off")
            self.arm_label.config(fg=MUTED)
        elif not self.safe_var.get():
            self.arm_var.set("● Always active (activation gesture is off)")
            self.arm_label.config(fg=AMBER)
        elif self.armed:
            left_s = max(0, self.armed_until - now)
            self.arm_var.set(f"● ACTIVE  ({left_s:.0f} s)")
            self.arm_label.config(fg=ACCENT)
        else:
            held = (now - self.palm_since) if self.palm_since else 0
            self.arm_var.set(f"● Waiting: hold an open palm for {ARM_HOLD:g} s  ({held:.1f})")
            self.arm_label.config(fg=AMBER)

    # ---------------------------------------------------------------------- camera
    def toggle(self):
        self.stop() if self.running else self.start()

    def ensure_model(self):
        """If the model file is missing, download it once from Google's official server (~8 MB)."""
        if os.path.exists(MODEL):
            return True
        self.status_var.set("Downloading the model (first run only, ~8 MB)...")
        self.update()
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL)
            return True
        except Exception as e:
            if os.path.exists(MODEL):
                os.remove(MODEL)                        # do not keep a partial file
            messagebox.showerror("Model download failed",
                                 f"Automatic download failed:\n{e}\n\nDownload it manually from this "
                                 f"link and put it next to gui_app.py:\n{MODEL_URL}")
            self.status_var.set("Model not found.")
            return False

    def start(self):
        if not self.ensure_model():
            return
        cam = int(self.cam_var.get())
        cap = cv2.VideoCapture(cam)
        if not cap.isOpened():
            messagebox.showerror("Camera", f"Camera {cam} could not be opened. Try another number.")
            return
        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)
        self.cap = cap
        self.running = True
        self.start_time = time.time()
        self.last_ts = 0
        self.prev = time.time()
        self.btn_start.config(text="Stop", bg=DANGER, fg="white")
        self.status_var.set("Camera is on.")
        self.loop()

    def stop(self):
        self.running = False
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.recognizer:
            self.recognizer.close()
            self.recognizer = None
        self.votes.clear()
        self.last_stable = "None"
        self.armed = False
        self.palm_since = None
        self.mouse_pos = None
        self.video.configure(image=self.blank,
                             text="Camera is off.\nPress “Start”.")
        self.gesture_var.set("-")
        self.conf_var.set("0%")
        self.bar["value"] = 0
        self.refresh_arm_label()
        self.btn_start.config(text="Start", bg=ACCENT, fg="black")
        self.status_var.set("Stopped.")

    def loop(self):
        if not self.running:
            return
        ok, frame = self.cap.read()
        if ok:
            self.process(frame)
        self.after_id = self.after(10, self.loop)

    # ----------------------------------------------------------------- recognition
    def process(self, frame):
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=np.ascontiguousarray(rgb))
        ts = max(int((time.time() - self.start_time) * 1000), self.last_ts + 1)
        self.last_ts = ts
        result = self.recognizer.recognize_for_video(mp_img, ts)

        name, score, lms = "None", 0.0, None
        if result.gestures and result.hand_landmarks:
            top = result.gestures[0][0]
            name, score = top.category_name, top.score
            lms = result.hand_landmarks[0]
            draw_hand(frame, lms)
            if is_ok(lms, frame.shape):            # OK is not in the built-in model
                name, score = "OK", 0.9
        self.votes.append(name if score > 0.5 else "None")

        stable = Counter(self.votes).most_common(1)[0][0]
        if stable != self.last_stable and stable != "None":
            self.add_history(NAMES.get(stable, stable))
        self.last_stable = stable

        now = time.time()
        try:
            if self.actions_var.get() and pyautogui:
                self.update_arming(stable, now)
                if self.armed:
                    if self.mode_var.get() == MODE_MOUSE:
                        self.run_mouse(lms, frame.shape, now)
                    else:
                        self.run_action(stable, now)
        except FAILSAFE_ERR:
            self.actions_var.set(False)
            self.armed = False
            self.refresh_actions_btn()
            self.status_var.set("Safety stop: the mouse reached the corner, control switched off.")
        self.refresh_arm_label(now)

        self.fps = 0.9 * self.fps + 0.1 * (1 / max(now - self.prev, 1e-6))
        self.prev = now
        self.gesture_var.set(NAMES.get(stable, stable))
        shown = score * 100 if stable != "None" else 0
        self.bar["value"] = shown
        self.conf_var.set(f"{shown:.0f}%   |   FPS: {self.fps:.0f}")

        self.last_frame = frame.copy()
        img = Image.fromarray(cv2.cvtColor(
            cv2.resize(frame, (VIDEO_W, VIDEO_H)), cv2.COLOR_BGR2RGB))
        self._photo = ImageTk.PhotoImage(img)       # keep a reference or Tk drops the image
        self.video.configure(image=self._photo, text="")

    # -------------------------------------------------------------------- commands
    def update_arming(self, stable, now):
        """Activates after holding an open palm for ARM_HOLD s; deactivates after ARM_TIMEOUT s idle."""
        if not self.safe_var.get():
            self.armed = True
            return
        if stable == "Open_Palm":
            if self.palm_since is None:
                self.palm_since = now
            elif not self.armed and now - self.palm_since >= ARM_HOLD:
                self.armed = True
                self.armed_until = now + ARM_TIMEOUT
                self.add_history("Activated")
        else:
            self.palm_since = None
        if self.armed and now > self.armed_until:
            self.armed = False
            self.add_history("Deactivated (timeout)")

    def run_action(self, stable, now):
        if stable not in ACTIONS:
            return
        key, cooldown, desc = ACTIONS[stable]
        if now - self.last_action.get(stable, 0) > cooldown:
            pyautogui.press(key)
            self.last_action[stable] = now
            self.armed_until = now + ARM_TIMEOUT      # activity: refresh the timer
            self.add_history(">> " + desc)

    def run_mouse(self, lms, shape, now):
        """The index fingertip moves the pointer, touching the thumb to it clicks."""
        if lms is None:
            self.pinching = False
            return
        self.armed_until = now + ARM_TIMEOUT
        pinching = pinch_ratio(lms, shape) < PINCH_RATIO

        if not pinching:                               # the pointer freezes while pinching
            tip = lms[8]
            span = 1 - 2 * MOUSE_MARGIN
            nx = min(max((tip.x - MOUSE_MARGIN) / span, 0), 1)
            ny = min(max((tip.y - MOUSE_MARGIN) / span, 0), 1)
            tx = 5 + nx * (self.screen_w - 10)          # 5 px border so we never hit (0,0) failsafe
            ty = 5 + ny * (self.screen_h - 10)
            if self.mouse_pos is None:
                self.mouse_pos = (tx, ty)
            else:
                cx, cy = self.mouse_pos
                self.mouse_pos = (cx + (tx - cx) * MOUSE_SMOOTH,
                                  cy + (ty - cy) * MOUSE_SMOOTH)
            pyautogui.moveTo(int(self.mouse_pos[0]), int(self.mouse_pos[1]))

        if pinching and not self.pinching and now - self.last_click > CLICK_COOLDOWN:
            pyautogui.click()
            self.last_click = now
            self.add_history(">> Click")
        self.pinching = pinching

    def add_history(self, text):
        self.history.insert(0, f"{time.strftime('%H:%M:%S')}  {text}")
        if self.history.size() > 30:
            self.history.delete(30, "end")

    # ----------------------------------------------------------------------- other
    def screenshot(self):
        if self.last_frame is None:
            self.status_var.set("Start the camera first.")
            return
        os.makedirs("screenshots", exist_ok=True)
        path = os.path.join("screenshots", time.strftime("gesture_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, self.last_frame)
        self.status_var.set(f"Saved: {path}")

    def on_close(self):
        self.stop()
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
