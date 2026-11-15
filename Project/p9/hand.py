import os
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import threading
import urllib.request
import numpy as np
import queue
from collections import deque
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
GESTURE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL_PATH = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
GESTURE_MODEL_PATH = os.path.join(SCRIPT_DIR, "gesture_recognizer.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
FINGER_NAMES = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
FINGER_SHORT = ["T", "I", "M", "R", "P"]

DISPLAY_W, DISPLAY_H = 640, 480

BG_TOP = "#0a0a1a"
BG_BOTTOM = "#15102a"
CARD_BG = "#1a1a2e"
CARD_BORDER = "#2d2d4a"
ACCENT = "#7c3aed"
ACCENT2 = "#a855f7"
CYAN = "#06b6d4"
GREEN = "#22c55e"
RED = "#ef4444"
AMBER = "#f59e0b"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
DIM = "#475569"

LM_COLOR = (168, 85, 247)
CONN_COLOR = (6, 182, 212)


def ensure_model(url, path):
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    urllib.request.urlretrieve(url, path)
    return path


def make_gradient(w, h, c1, c2):
    a1 = np.array([int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)])
    a2 = np.array([int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)])
    t = np.linspace(0, 1, h).reshape(-1, 1)
    grad = (a1 + (a2 - a1) * t).astype(np.uint8)
    grad = np.broadcast_to(grad[:, None, :], (h, w, 3)).copy()
    return Image.fromarray(grad)


def rounded_card(parent, w, h, bg=CARD_BG, border=CARD_BORDER, radius=14):
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=bg, outline=border, width=1)
    photo = ImageTk.PhotoImage(img)
    lbl = tk.Label(parent, image=photo, bg=parent["bg"])
    lbl.image = photo
    return lbl


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], CONN_COLOR, 3, cv2.LINE_AA)
    for i, p in enumerate(pts):
        cv2.circle(frame, p, 7, LM_COLOR, -1, cv2.LINE_AA)
        cv2.circle(frame, p, 7, (255, 255, 255), 1, cv2.LINE_AA)


def draw_overlay(frame, gesture, posture, handedness):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 52), (10, 10, 26), -1)
    cv2.rectangle(overlay, (0, h - 40), (w, h), (10, 10, 26), -1)
    alpha = 0.55
    frame[:] = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    if gesture and gesture != "None":
        cv2.putText(frame, gesture, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (168, 85, 247), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, posture, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (6, 182, 212), 2, cv2.LINE_AA)
    sub = f"{handedness} hand" if handedness != "—" else "No hand detected"
    cv2.putText(frame, sub, (16, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (148, 163, 184), 1, cv2.LINE_AA)


def classify_posture(fingers_up):
    up = fingers_up
    if all(up):
        return "Open Hand"
    if not any(up):
        return "Fist"
    if up[1] and not up[2] and not up[3] and not up[4] and not up[0]:
        return "Pointing"
    if up[1] and up[2] and not up[3] and not up[4]:
        return "Peace / Victory"
    if up[0] and up[4] and not up[1] and not up[2] and not up[3]:
        return "Call Me"
    if up[0] and not up[1] and not up[2] and not up[3] and up[4]:
        return "Rock / Horns"
    if up[0] and up[1] and not up[2] and not up[3] and not up[4]:
        return "Gun"
    if up[4] and not up[1] and not up[2] and not up[3] and not up[0]:
        return "Pinky Out"
    if up[1] and up[2] and up[3] and not up[4] and not up[0]:
        return "Three Fingers"
    if up[1] and up[2] and up[3] and up[4] and not up[0]:
        return "Four Fingers"
    if up[0] and up[1] and up[2] and not up[3] and not up[4]:
        return "Spider-Man"
    if up[0] and not up[1] and not up[2] and not up[3] and not up[4]:
        return "Thumbs Up" if fingers_up[0] else "Fist"
    return "Unknown"


def get_fingers_up(landmarks, handedness_label):
    fingers_up = []
    for i in range(5):
        tip = landmarks[FINGER_TIPS[i]]
        pip = landmarks[FINGER_PIPS[i]]
        if i == 0:
            if handedness_label == "Right":
                fingers_up.append(tip.x < pip.x)
            else:
                fingers_up.append(tip.x > pip.x)
        else:
            fingers_up.append(tip.y < pip.y)
    return [int(f) for f in fingers_up]


class StyledButton(tk.Canvas):
    def __init__(self, parent, text, command, bg_color=ACCENT, fg="white",
                 width=96, height=38, font=("Helvetica", 12, "bold"), state="normal"):
        super().__init__(parent, width=width, height=height, bg=parent["bg"],
                         highlightthickness=0, cursor="hand2")
        self._text = text
        self._command = command
        self._bg = bg_color
        self._fg = fg
        self._font = font
        self._bw = width
        self._bh = height
        self._disabled = state == "disabled"
        self._draw()
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        self.bind("<Enter>", self._enter)
        self.bind("<Leave>", self._leave)

    def _round(self, x0, y0, x1, y1, r, **kw):
        pts = [x0+r, y0, x1-r, y0, x1, y0, x1, y0+r, x1, y1-r, x1-r, y1,
               x0+r, y1, x0, y1, x0, y1-r, x0, y0+r, x0, y0]
        return self.create_polygon(pts, smooth=True, **kw)

    def _draw(self, bg=None):
        self.delete("all")
        c = bg or self._bg
        if self._disabled:
            c = "#2a2a3e"
        self._round(2, 2, self._bw - 2, self._bh - 2, 10, fill=c, outline="")
        self.create_text(self._bw // 2, self._bh // 2, text=self._text,
                         fill=self._fg if not self._disabled else "#5a5a6e",
                         font=self._font)

    def _press(self, e):
        if not self._disabled:
            self._draw(self._shade(self._bg, -0.25))

    def _release(self, e):
        if not self._disabled:
            self._draw()
            if self._command:
                self._command()

    def _enter(self, e):
        if not self._disabled:
            self._draw(self._shade(self._bg, 0.15))

    def _leave(self, e):
        if not self._disabled:
            self._draw()

    @staticmethod
    def _shade(hex_c, f):
        h = hex_c.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = max(0, min(255, int(r + 255 * f)))
        g = max(0, min(255, int(g + 255 * f)))
        b = max(0, min(255, int(b + 255 * f)))
        return f"#{r:02x}{g:02x}{b:02x}"

    def set_state(self, state):
        self._disabled = state == "disabled"
        self._draw()


class FingerIndicator(tk.Canvas):
    def __init__(self, parent, width=270, height=80):
        super().__init__(parent, width=width, height=height, bg=CARD_BG,
                         highlightthickness=0)
        self._bw = width
        self._bh = height
        self.update_state([0, 0, 0, 0, 0])

    def update_state(self, fingers_up):
        self.delete("all")
        n = 5
        gap = 12
        pill_w = (self._bw - gap * (n + 1)) // n
        pill_h = 46
        y0 = 8
        for i in range(n):
            x0 = gap + i * (pill_w + gap)
            up = fingers_up[i]
            color = GREEN if up else DIM
            self._round(x0, y0, x0 + pill_w, y0 + pill_h, 8, fill=color, outline="")
            self.create_text(x0 + pill_w // 2, y0 + pill_h // 2,
                             text=FINGER_SHORT[i], fill="white" if up else "#6a6a7e",
                             font=("Helvetica", 16, "bold"))
            self.create_text(x0 + pill_w // 2, y0 + pill_h + 12,
                             text=FINGER_NAMES[i], fill=MUTED,
                             font=("Helvetica", 8))

    def _round(self, x0, y0, x1, y1, r, **kw):
        pts = [x0+r, y0, x1-r, y0, x1, y0, x1, y0+r, x1, y1-r, x1-r, y1,
               x0+r, y1, x0, y1, x0, y1-r, x0, y0+r, x0, y0]
        self.create_polygon(pts, smooth=True, **kw)


class ConfidenceBar(tk.Canvas):
    def __init__(self, parent, width=250, height=12):
        super().__init__(parent, width=width, height=height, bg=CARD_BG,
                         highlightthickness=0)
        self._bw = width
        self._bh = height
        self.set_value(0)

    def set_value(self, val):
        self.delete("all")
        self._round(0, 0, self._bw - 1, self._bh - 1, 6, fill="#0d0d1a", outline=CARD_BORDER)
        fill_w = int((self._bw - 4) * max(0, min(1, val)))
        if fill_w > 0:
            self._round(2, 2, 2 + fill_w, self._bh - 3, 5, fill=ACCENT2, outline="")

    def _round(self, x0, y0, x1, y1, r, **kw):
        pts = [x0+r, y0, x1-r, y0, x1, y0, x1, y0+r, x1, y1-r, x1-r, y1,
               x0+r, y1, x0, y1, x0, y1-r, x0, y0+r, x0, y0]
        self.create_polygon(pts, smooth=True, **kw)


class HandPostureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Hand Posture Detection")
        self.root.configure(bg=BG_TOP)
        self.root.geometry("1000x640")
        self.root.resizable(False, False)

        self.running = False
        self.cap = None
        self.gestureRecognizer = None
        self.mirror = True
        self.max_hands = 1
        self.det_conf = 0.7
        self.track_conf = 0.6
        self.timestamp_ms = 0
        self.gesture_history = deque(maxlen=6)
        self.last_gesture = ""
        self._queue = queue.Queue(maxsize=2)
        self._polling = False

        self._build_ui()

    def _build_ui(self):
        self._bg_img = make_gradient(1000, 640, BG_TOP, BG_BOTTOM)
        self._bg_photo = ImageTk.PhotoImage(self._bg_img)
        self.bg_label = tk.Label(self.root, image=self._bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        header = tk.Frame(self.root, bg=BG_TOP)
        header.pack(fill=tk.X, padx=18, pady=(16, 0))
        tk.Label(header, text="✋  AI Hand Posture Detection", bg=BG_TOP, fg=TEXT,
                 font=("Helvetica", 22, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="MediaPipe Tasks  ·  Gesture AI", bg=BG_TOP, fg=ACCENT2,
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(12, 0), pady=(6, 0))

        btn_frame = tk.Frame(header, bg=BG_TOP)
        btn_frame.pack(side=tk.RIGHT)
        self.btn_start = StyledButton(btn_frame, "▶  Start", self.start, bg_color=GREEN, width=100)
        self.btn_start.pack(side=tk.LEFT, padx=6)
        self.btn_stop = StyledButton(btn_frame, "■  Stop", self.stop, bg_color=RED, width=100,
                                     state="disabled")
        self.btn_stop.pack(side=tk.LEFT, padx=6)

        controls = tk.Frame(self.root, bg=BG_TOP)
        controls.pack(fill=tk.X, padx=18, pady=(10, 4))
        tk.Label(controls, text="Hands:", bg=BG_TOP, fg=MUTED,
                 font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.spin_hands = tk.Spinbox(controls, from_=1, to=4, width=3, font=("Helvetica", 11),
                                     bg=CARD_BG, fg=TEXT, relief="flat", justify="center")
        self.spin_hands.pack(side=tk.LEFT, padx=(6, 16))
        self.spin_hands.delete(0, "end")
        self.spin_hands.insert(0, "1")

        self.var_mirror = tk.BooleanVar(value=True)
        cb = tk.Checkbutton(controls, text="Mirror", variable=self.var_mirror,
                            bg=BG_TOP, fg=TEXT, selectcolor=CARD_BG, activebackground=BG_TOP,
                            activeforeground=TEXT, font=("Helvetica", 11), highlightthickness=0)
        cb.pack(side=tk.LEFT, padx=8)

        tk.Label(controls, text="Confidence:", bg=BG_TOP, fg=MUTED,
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(8, 4))
        self.scale_det = tk.Scale(controls, from_=0.3, to=0.9, orient=tk.HORIZONTAL,
                                  length=110, bg=BG_TOP, fg=TEXT, troughcolor=CARD_BG,
                                  highlightthickness=0, font=("Helvetica", 9),
                                  activebackground=ACCENT)
        self.scale_det.set(0.7)
        self.scale_det.pack(side=tk.LEFT)

        body = tk.Frame(self.root, bg=BG_TOP)
        body.pack(fill=tk.BOTH, expand=True, padx=18, pady=6)

        video_wrap = tk.Frame(body, bg="#000000", width=DISPLAY_W, height=DISPLAY_H,
                              highlightbackground=CARD_BORDER, highlightthickness=2)
        video_wrap.pack(side=tk.LEFT, padx=(0, 14))
        video_wrap.pack_propagate(False)
        self.video_label = tk.Label(video_wrap, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self._placeholder = ImageTk.PhotoImage(
            Image.new("RGB", (DISPLAY_W, DISPLAY_H), (15, 16, 42)))
        self.video_label.config(image=self._placeholder)

        info = tk.Frame(body, bg=BG_TOP, width=310)
        info.pack(side=tk.LEFT, fill=tk.Y)
        info.pack_propagate(False)

        self._section(info, "🎯  AI GESTURE")
        self.gesture_var = tk.StringVar(value="—")
        tk.Label(info, textvariable=self.gesture_var, bg=BG_TOP, fg=ACCENT2,
                 font=("Helvetica", 19, "bold"), wraplength=290,
                 justify="left").pack(anchor=tk.W, pady=(0, 4))
        self.conf_bar = ConfidenceBar(info, width=270)
        self.conf_bar.pack(anchor=tk.W, pady=(0, 2))
        self.conf_var = tk.StringVar(value="")
        tk.Label(info, textvariable=self.conf_var, bg=BG_TOP, fg=MUTED,
                 font=("Helvetica", 9)).pack(anchor=tk.W)

        self._sep(info)
        self._section(info, "🤚  CUSTOM POSTURE")
        self.posture_var = tk.StringVar(value="—")
        tk.Label(info, textvariable=self.posture_var, bg=BG_TOP, fg=CYAN,
                 font=("Helvetica", 16, "bold"), wraplength=290,
                 justify="left").pack(anchor=tk.W, pady=(0, 4))

        self._sep(info)
        self._section(info, "🖐  FINGER STATUS")
        self.finger_widget = FingerIndicator(info, width=270, height=80)
        self.finger_widget.pack(anchor=tk.W, pady=(2, 4))

        self._sep(info)
        self._section(info, "📊  STATS")
        stats = tk.Frame(info, bg=BG_TOP)
        stats.pack(fill=tk.X, pady=(2, 0))
        self.hand_var = tk.StringVar(value="Hand: —")
        self.fps_var = tk.StringVar(value="FPS: —")
        self.lm_var = tk.StringVar(value="Landmarks: 0")
        tk.Label(stats, textvariable=self.hand_var, bg=BG_TOP, fg=TEXT,
                 font=("Helvetica", 10)).pack(anchor=tk.W)
        tk.Label(stats, textvariable=self.fps_var, bg=BG_TOP, fg=GREEN,
                 font=("Helvetica", 10, "bold")).pack(anchor=tk.W)
        tk.Label(stats, textvariable=self.lm_var, bg=BG_TOP, fg=MUTED,
                 font=("Helvetica", 10)).pack(anchor=tk.W)

        self._sep(info)
        self._section(info, "📜  HISTORY")
        self.history_var = tk.StringVar(value="—")
        tk.Label(info, textvariable=self.history_var, bg=BG_TOP, fg=MUTED,
                 font=("Helvetica", 9), wraplength=290, justify="left",
                 height=5).pack(anchor=tk.W, pady=(2, 0))

        footer = tk.Frame(self.root, bg="#0d0d1a", height=32)
        footer.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_var = tk.StringVar(value="  ● Ready — press Start to begin")
        tk.Label(footer, textvariable=self.status_var, bg="#0d0d1a", fg=MUTED,
                 font=("Helvetica", 10), anchor="w").pack(fill=tk.X, side=tk.LEFT, pady=6)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _section(self, parent, title):
        tk.Label(parent, text=title, bg=BG_TOP, fg=ACCENT,
                 font=("Helvetica", 10, "bold")).pack(anchor=tk.W, pady=(10, 2))

    def _sep(self, parent):
        tk.Frame(parent, bg=CARD_BORDER, height=1).pack(fill=tk.X, pady=4)

    def start(self):
        if self.running:
            return
        try:
            self.max_hands = int(self.spin_hands.get())
        except ValueError:
            self.max_hands = 1
        self.mirror = self.var_mirror.get()
        self.det_conf = float(self.scale_det.get())
        self.track_conf = 0.6

        self.status_var.set("  ● Loading AI models...")
        self.root.update_idletasks()
        try:
            ensure_model(HAND_MODEL_URL, HAND_MODEL_PATH)
            gpath = ensure_model(GESTURE_MODEL_URL, GESTURE_MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download model:\n{e}")
            return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Error", "Could not open webcam.\n\n"
                                 "macOS: System Settings > Privacy & Security > Camera\n"
                                 "Enable access for Terminal / Python.")
            self.cap = None
            return
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        base_options = python.BaseOptions(model_asset_path=gpath)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.det_conf,
            min_hand_presence_confidence=self.track_conf,
            min_tracking_confidence=self.track_conf,
        )
        self.gestureRecognizer = vision.GestureRecognizer.create_from_options(options)
        self.timestamp_ms = 0

        self.running = True
        self.btn_start.set_state("disabled")
        self.btn_stop.set_state("normal")
        self.status_var.set("  ● Running — show your hand ✋")
        self._polling = True
        self._poll_queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        self._polling = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.gestureRecognizer is not None:
            self.gestureRecognizer.close()
            self.gestureRecognizer = None
        self.btn_start.set_state("normal")
        self.btn_stop.set_state("disabled")
        self.status_var.set("  ● Stopped.")
        self.gesture_var.set("—")
        self.posture_var.set("—")
        self.hand_var.set("Hand: —")
        self.fps_var.set("FPS: —")
        self.lm_var.set("Landmarks: 0")
        self.conf_var.set("")
        self.conf_bar.set_value(0)
        self.finger_widget.update_state([0, 0, 0, 0, 0])
        self.video_label.config(image=self._placeholder)

    def _loop(self):
        prev = cv2.getTickCount()
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                break
            if self.mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.timestamp_ms += 1
            try:
                result = self.gestureRecognizer.recognize_for_video(mp_image, self.timestamp_ms)
            except Exception:
                break

            gesture_text = "—"
            posture_text = "—"
            hand_text = "—"
            fingers_up = [0, 0, 0, 0, 0]
            conf = 0.0
            lm_count = 0

            if result.hand_landmarks and result.handedness:
                for idx, landmarks in enumerate(result.hand_landmarks):
                    draw_hand(frame, landmarks)
                    lm_count = len(landmarks)
                    handed = result.handedness[idx][0].category_name
                    hand_text = handed
                    fingers_up = get_fingers_up(landmarks, handed)
                    posture_text = classify_posture(fingers_up)
                    if result.gestures and idx < len(result.gestures):
                        g = result.gestures[idx][0]
                        gesture_text = g.category_name.replace("_", " ")
                        conf = g.score
                    self._add_history(gesture_text if gesture_text != "None" else posture_text)

            draw_overlay(frame, gesture_text if gesture_text != "—" else "",
                         posture_text, hand_text)

            now = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (now - prev)
            prev = now

            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            try:
                self._queue.put_nowait({
                    "frame": display,
                    "gesture": gesture_text if gesture_text != "None" else "—",
                    "posture": posture_text,
                    "hand": hand_text,
                    "fps": fps,
                    "lm": lm_count,
                    "conf": conf,
                    "fingers": fingers_up,
                    "history": list(self.gesture_history),
                })
            except queue.Full:
                pass

        if self.running:
            self.stop()

    def _poll_queue(self):
        if not self._polling:
            return
        try:
            while True:
                d = self._queue.get_nowait()
                img = Image.fromarray(d["frame"])
                photo = ImageTk.PhotoImage(image=img)
                self.video_label.img = photo
                self.video_label.config(image=photo)
                self.gesture_var.set(d["gesture"])
                self.posture_var.set(d["posture"])
                self.hand_var.set(f"Hand: {d['hand']}")
                self.fps_var.set(f"FPS: {d['fps']:.1f}")
                self.lm_var.set(f"Landmarks: {d['lm']}")
                self.conf_var.set(f"{d['conf'] * 100:.0f}% confidence" if d["conf"] > 0 else "")
                self.conf_bar.set_value(d["conf"])
                self.finger_widget.update_state(d["fingers"])
                self.history_var.set("\n".join(d["history"]))
        except queue.Empty:
            pass
        self.root.after(30, self._poll_queue)

    def _add_history(self, gesture):
        if not gesture or gesture == "—" or gesture == self.last_gesture:
            return
        self.last_gesture = gesture
        ts = datetime.now().strftime("%H:%M:%S")
        self.gesture_history.appendleft(f"[{ts}]  {gesture}")

    def _show_frame(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(image=img)
        self.video_label.img = img
        self.video_label.config(image=img)

    def on_close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = HandPostureApp(root)
    root.mainloop()
