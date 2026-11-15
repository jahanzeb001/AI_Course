import os
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import threading
import urllib.request
import numpy as np
import queue
from collections import deque
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "holistic_landmarker.task")
POSE_MODEL_PATH = os.path.join(SCRIPT_DIR, "pose_landmarker_full.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

POSE_GROUPS = [
    ([(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10)], (168, 85, 247)),
    ([(11,12),(11,23),(12,24),(23,24)], (6, 182, 212)),
    ([(11,13),(13,15),(15,17),(15,19),(15,21),(17,19)], (34, 197, 94)),
    ([(12,14),(14,16),(16,18),(16,20),(16,22),(18,20)], (245, 158, 11)),
    ([(23,25),(25,27),(27,29),(27,31),(29,31)], (34, 197, 94)),
    ([(24,26),(26,28),(28,30),(28,32),(30,32)], (245, 158, 11)),
]

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269,
        267, 0, 37, 39, 40, 185, 61]
LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159,
            160, 161, 246, 33]
RIGHT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385,
             386, 387, 388, 466, 263]
LEFT_BROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 70]
RIGHT_BROW = [336, 296, 334, 293, 300, 285, 295, 282, 283, 276, 336]

FACE_FEATURES = [
    (FACE_OVAL, (124, 58, 237), 2),
    (LEFT_EYE, (6, 182, 212), 2),
    (RIGHT_EYE, (6, 182, 212), 2),
    (LEFT_BROW, (245, 158, 11), 2),
    (RIGHT_BROW, (245, 158, 11), 2),
    (LIPS, (236, 72, 153), 2),
]

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
PINK = "#ec4899"
TEXT = "#e2e8f0"
MUTED = "#94a3b8"
DIM = "#475569"

FACE_C = (168, 85, 247)
LHAND_C = (34, 197, 94)
RHAND_C = (245, 158, 11)
POSE_C = (6, 182, 212)


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def ensure_pose_model():
    if os.path.exists(POSE_MODEL_PATH) and os.path.getsize(POSE_MODEL_PATH) > 0:
        return POSE_MODEL_PATH
    urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    return POSE_MODEL_PATH


def make_gradient(w, h, c1, c2):
    a1 = np.array([int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)])
    a2 = np.array([int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)])
    t = np.linspace(0, 1, h).reshape(-1, 1)
    grad = (a1 + (a2 - a1) * t).astype(np.uint8)
    grad = np.broadcast_to(grad[:, None, :], (h, w, 3)).copy()
    return Image.fromarray(grad)


def draw_hand(frame, landmarks, color):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, color, -1, cv2.LINE_AA)


def draw_face(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for lm in landmarks:
        cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 1, (50, 50, 80), -1)
    for indices, color, thick in FACE_FEATURES:
        poly = [pts[i] for i in indices]
        cv2.polylines(frame, [np.array(poly, dtype=np.int32)], False, color, thick, cv2.LINE_AA)


def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for connections, color in POSE_GROUPS:
        for a, b in connections:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], color, 3, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 4, POSE_C, -1, cv2.LINE_AA)


def draw_overlay(frame, parts_detected, total_lm):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 26), -1)
    cv2.rectangle(overlay, (0, h - 38), (w, h), (10, 10, 26), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    label = "Full Body Detected ✅" if parts_detected >= 3 else "Partial Detection"
    cv2.putText(frame, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                (168, 85, 247) if parts_detected >= 3 else (148, 163, 184),
                2, cv2.LINE_AA)
    cv2.putText(frame, f"Landmarks: {total_lm}", (w - 180, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (6, 182, 212), 2, cv2.LINE_AA)


def classify_emotion(bs):
    if not bs:
        return "—"
    smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    jaw = bs.get("jawOpen", 0)
    eye_wide = (bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 2
    blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
    if blink > 0.6:
        return "Blinking 😑"
    if smile > 0.4 and jaw < 0.2:
        return "Happy 😊"
    if smile > 0.2:
        return "Smiling 🙂"
    if jaw > 0.3 and eye_wide > 0.15:
        return "Surprised 😮"
    if frown > 0.3:
        return "Sad 😢"
    if jaw > 0.25:
        return "Mouth Open 😮"
    return "Neutral 😐"


def classify_posture(lm):
    if not lm or len(lm) < 29:
        return "—"
    nose = lm[0]
    ls, rs = lm[11], lm[12]
    lw, rw = lm[15], lm[16]
    lh, rh = lm[23], lm[24]
    lk, rk = lm[25], lm[26]

    def v(p):
        return getattr(p, "visibility", 1.0)

    if v(lw) > 0.3 and v(rw) > 0.3:
        if lw.y < nose.y and rw.y < nose.y:
            return "Hands Up 🙌"
    if v(lw) > 0.3 and v(rw) > 0.3 and v(ls) > 0.3 and v(rs) > 0.3:
        if abs(lw.y - ls.y) < 0.06 and abs(rw.y - rs.y) < 0.06:
            if lw.x < ls.x and rw.x > rs.x:
                return "T-Pose 🤖"
    if v(lh) > 0.3 and v(rh) > 0.3 and v(lk) > 0.3 and v(rk) > 0.3:
        if abs(lh.y - lk.y) < 0.08 and abs(rh.y - rk.y) < 0.08:
            return "Squatting 🏋️"
    if v(ls) > 0.3 and v(rs) > 0.3 and v(lh) > 0.3 and v(rh) > 0.3:
        mid_s = (ls.x + rs.x) / 2
        mid_h = (lh.x + rh.x) / 2
        if abs(mid_s - mid_h) > 0.08:
            return "Leaning ↔️"
    if v(ls) > 0.3 and v(rs) > 0.3 and v(lh) > 0.3 and v(rh) > 0.3:
        return "Standing 🧍"
    return "Partial"


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


class StatusPills(tk.Canvas):
    def __init__(self, parent, labels, width=270, height=70):
        super().__init__(parent, width=width, height=height, bg=BG_TOP,
                         highlightthickness=0)
        self._bw = width
        self._bh = height
        self._labels = labels
        self.update_state([0] * len(labels))

    def update_state(self, states):
        self.delete("all")
        n = len(self._labels)
        cols = 2
        gap = 8
        pill_w = (self._bw - gap * (cols + 1)) // cols
        pill_h = 26
        for i, label in enumerate(self._labels):
            col = i % cols
            row = i // cols
            x0 = gap + col * (pill_w + gap)
            y0 = gap + row * (pill_h + gap)
            up = states[i] if i < len(states) else 0
            color = GREEN if up else DIM
            self._round(x0, y0, x0 + pill_w, y0 + pill_h, 6, fill=color, outline="")
            mark = "✓" if up else "✗"
            self.create_text(x0 + 14, y0 + pill_h // 2, text=mark,
                             fill="white", font=("Helvetica", 10, "bold"))
            self.create_text(x0 + pill_w // 2 + 8, y0 + pill_h // 2,
                             text=label, fill="white" if up else "#6a6a7e",
                             font=("Helvetica", 9, "bold"))

    def _round(self, x0, y0, x1, y1, r, **kw):
        pts = [x0+r, y0, x1-r, y0, x1, y0, x1, y0+r, x1, y1-r, x1-r, y1,
               x0+r, y1, x0, y1, x0, y1-r, x0, y0+r, x0, y0]
        self.create_polygon(pts, smooth=True, **kw)


class FullBodyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Full-Body Detection")
        self.root.configure(bg=BG_TOP)
        self.root.geometry("1000x700")
        self.root.resizable(False, False)

        self.running = False
        self.cap = None
        self.holisticLandmarker = None
        self.poseLandmarker = None
        self.mirror = True
        self.det_conf = 0.5
        self.track_conf = 0.5
        self.timestamp_ms = 0
        self.history = deque(maxlen=6)
        self.last_entry = ""
        self._queue = queue.Queue(maxsize=2)
        self._polling = False
        self.video_path = None
        self.source_mode = "webcam"
        self.num_poses = 1
        self.video_total_frames = 0

        self._build_ui()

    def _build_ui(self):
        self._bg_img = make_gradient(1000, 700, BG_TOP, BG_BOTTOM)
        self._bg_photo = ImageTk.PhotoImage(self._bg_img)
        self.bg_label = tk.Label(self.root, image=self._bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        header = tk.Frame(self.root, bg=BG_TOP)
        header.pack(fill=tk.X, padx=18, pady=(16, 0))
        tk.Label(header, text="🧍  AI Full-Body Detection", bg=BG_TOP, fg=TEXT,
                 font=("Helvetica", 22, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="Holistic  ·  Face + Hands + Pose + Expressions",
                 bg=BG_TOP, fg=ACCENT2,
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

        tk.Label(controls, text="Source:", bg=BG_TOP, fg=MUTED,
                 font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.var_source = tk.StringVar(value="webcam")
        tk.Radiobutton(controls, text="Webcam", variable=self.var_source,
                       value="webcam", bg=BG_TOP, fg=TEXT, selectcolor=CARD_BG,
                       activebackground=BG_TOP, activeforeground=TEXT,
                       font=("Helvetica", 10), highlightthickness=0,
                       command=self._on_source_change).pack(side=tk.LEFT, padx=(4, 8))
        tk.Radiobutton(controls, text="Video File", variable=self.var_source,
                       value="video", bg=BG_TOP, fg=TEXT, selectcolor=CARD_BG,
                       activebackground=BG_TOP, activeforeground=TEXT,
                       font=("Helvetica", 10), highlightthickness=0,
                       command=self._on_source_change).pack(side=tk.LEFT, padx=(0, 8))

        self.btn_upload = StyledButton(controls, "📁 Upload", self.upload_video,
                                       bg_color=ACCENT, width=90, height=30,
                                       font=("Helvetica", 10, "bold"), state="disabled")
        self.btn_upload.pack(side=tk.LEFT, padx=4)

        self.video_label_var = tk.StringVar(value="")
        tk.Label(controls, textvariable=self.video_label_var, bg=BG_TOP, fg=AMBER,
                 font=("Helvetica", 8), width=22, anchor="w").pack(side=tk.LEFT, padx=4)

        controls2 = tk.Frame(self.root, bg=BG_TOP)
        controls2.pack(fill=tk.X, padx=18, pady=(2, 4))

        tk.Label(controls2, text="Max Persons:", bg=BG_TOP, fg=MUTED,
                 font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.spin_poses = tk.Spinbox(controls2, from_=1, to=4, width=3, font=("Helvetica", 11),
                                     bg=CARD_BG, fg=TEXT, relief="flat", justify="center")
        self.spin_poses.pack(side=tk.LEFT, padx=(6, 16))
        self.spin_poses.delete(0, "end")
        self.spin_poses.insert(0, "1")

        self.var_mirror = tk.BooleanVar(value=True)
        tk.Checkbutton(controls2, text="Mirror", variable=self.var_mirror,
                       bg=BG_TOP, fg=TEXT, selectcolor=CARD_BG, activebackground=BG_TOP,
                       activeforeground=TEXT, font=("Helvetica", 11),
                       highlightthickness=0).pack(side=tk.LEFT, padx=8)

        tk.Label(controls2, text="Confidence:", bg=BG_TOP, fg=MUTED,
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(8, 4))
        self.scale_det = tk.Scale(controls2, from_=0.3, to=0.9, orient=tk.HORIZONTAL,
                                  length=110, bg=BG_TOP, fg=TEXT, troughcolor=CARD_BG,
                                  highlightthickness=0, font=("Helvetica", 9),
                                  activebackground=ACCENT)
        self.scale_det.set(0.5)
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

        self._section(info, "🧩  BODY PARTS DETECTED")
        self.status_widget = StatusPills(info, ["Face", "Left Hand", "Right Hand", "Body Pose"],
                                         width=270, height=70)
        self.status_widget.pack(anchor=tk.W, pady=(2, 4))

        self._sep(info)
        self._section(info, "😀  EXPRESSION")
        self.emotion_var = tk.StringVar(value="—")
        tk.Label(info, textvariable=self.emotion_var, bg=BG_TOP, fg=ACCENT2,
                 font=("Helvetica", 18, "bold"), wraplength=290,
                 justify="left").pack(anchor=tk.W, pady=(0, 4))

        self._sep(info)
        self._section(info, "🧍  POSTURE")
        self.posture_var = tk.StringVar(value="—")
        tk.Label(info, textvariable=self.posture_var, bg=BG_TOP, fg=CYAN,
                 font=("Helvetica", 16, "bold"), wraplength=290,
                 justify="left").pack(anchor=tk.W, pady=(0, 4))

        self._sep(info)
        self._section(info, "✋  HANDS")
        self.hands_var = tk.StringVar(value="—")
        tk.Label(info, textvariable=self.hands_var, bg=BG_TOP, fg=AMBER,
                 font=("Helvetica", 11), wraplength=290,
                 justify="left").pack(anchor=tk.W, pady=(0, 4))

        self._sep(info)
        self._section(info, "📈  STATS")
        stats = tk.Frame(info, bg=BG_TOP)
        stats.pack(fill=tk.X, pady=(2, 0))
        self.fps_var = tk.StringVar(value="FPS: —")
        self.lm_var = tk.StringVar(value="Landmarks: 0")
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

    def _on_source_change(self):
        self.source_mode = self.var_source.get()
        if self.source_mode == "video":
            self.btn_upload.set_state("normal")
        else:
            self.btn_upload.set_state("disabled")

    def upload_video(self):
        path = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                       ("All Files", "*.*")])
        if path:
            self.video_path = path
            name = os.path.basename(path)
            if len(name) > 22:
                name = name[:19] + "..."
            self.video_label_var.set(name)

    def start(self):
        if self.running:
            return
        self.mirror = self.var_mirror.get()
        self.det_conf = float(self.scale_det.get())
        self.track_conf = 0.5
        self.source_mode = self.var_source.get()
        try:
            self.num_poses = int(self.spin_poses.get())
        except ValueError:
            self.num_poses = 1

        if self.source_mode == "video" and not self.video_path:
            messagebox.showwarning("Warning", "Please upload a video file first.")
            return

        self.status_var.set("  ● Loading AI model...")
        self.root.update_idletasks()
        try:
            model_path = ensure_model()
            if self.num_poses > 1 or self.source_mode == "video":
                pose_model_path = ensure_pose_model()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to download model:\n{e}")
            return

        self.status_var.set("  ● Opening source...")
        self.root.update_idletasks()
        self.cap = None

        if self.source_mode == "video":
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", f"Could not open video file:\n{self.video_path}")
                self.cap = None
                return
            self.video_total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        else:
            for attempt in range(5):
                self.cap = cv2.VideoCapture(0)
                if self.cap.isOpened():
                    ret, _ = self.cap.read()
                    if ret:
                        break
                self.cap.release()
                self.cap = None
                threading.Event().wait(0.5)
            if self.cap is None or not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open webcam.\n\n"
                                     "macOS: System Settings > Privacy & Security > Camera\n"
                                     "Enable access for Terminal / Python.\n\n"
                                     "If stuck: close other apps using the camera.")
                self.cap = None
                return
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            self.video_total_frames = 0

        self.status_var.set("  ● Initializing AI model...")
        self.root.update_idletasks()
        try:
            if self.num_poses > 1 or self.source_mode == "video":
                base_options = python.BaseOptions(model_asset_path=pose_model_path)
                options = vision.PoseLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    num_poses=self.num_poses,
                    min_pose_detection_confidence=self.det_conf,
                    min_pose_presence_confidence=self.track_conf,
                    min_tracking_confidence=self.track_conf,
                )
                self.poseLandmarker = vision.PoseLandmarker.create_from_options(options)
            else:
                base_options = python.BaseOptions(model_asset_path=model_path)
                options = vision.HolisticLandmarkerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.VIDEO,
                    min_face_detection_confidence=self.det_conf,
                    min_face_presence_confidence=self.track_conf,
                    min_hand_landmarks_confidence=self.det_conf,
                    min_pose_detection_confidence=self.det_conf,
                    min_pose_presence_confidence=self.track_conf,
                )
                self.holisticLandmarker = vision.HolisticLandmarker.create_from_options(options)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize model:\n{e}")
            self.cap.release()
            self.cap = None
            return
        self.timestamp_ms = 0

        self.running = True
        self.btn_start.set_state("disabled")
        self.btn_stop.set_state("normal")
        src_label = "video" if self.source_mode == "video" else "camera"
        self.status_var.set(f"  ● Running — detecting from {src_label} 🧍")
        self._polling = True
        self._poll_queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        self._polling = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.holisticLandmarker is not None:
            self.holisticLandmarker.close()
            self.holisticLandmarker = None
        if self.poseLandmarker is not None:
            self.poseLandmarker.close()
            self.poseLandmarker = None
        self.btn_start.set_state("normal")
        self.btn_stop.set_state("disabled")
        self.status_var.set("  ● Stopped.")
        self.emotion_var.set("—")
        self.posture_var.set("—")
        self.hands_var.set("—")
        self.fps_var.set("FPS: —")
        self.lm_var.set("Landmarks: 0")
        self.history_var.set("—")
        self.status_widget.update_state([0, 0, 0, 0])
        self.video_label.config(image=self._placeholder)

    def _loop(self):
        prev = cv2.getTickCount()
        while self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if not ret:
                if self.source_mode == "video":
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            if self.mirror:
                frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if self.source_mode == "video":
                self.timestamp_ms = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
            else:
                self.timestamp_ms += 33
            if self.timestamp_ms == 0:
                self.timestamp_ms = 1

            total_lm = 0
            parts = [0, 0, 0, 0]
            emotion = "—"
            posture = "—"
            hands_text = "—"
            bs_dict = {}
            pose_count = 0

            if self.poseLandmarker is not None:
                try:
                    result = self.poseLandmarker.detect_for_video(mp_image, self.timestamp_ms)
                except Exception:
                    break
                if result.pose_landmarks:
                    pose_count = len(result.pose_landmarks)
                    colors = [(34,197,94), (245,158,11), (6,182,212), (236,72,153)]
                    for idx, landmarks in enumerate(result.pose_landmarks):
                        draw_pose(frame, landmarks)
                        total_lm += len(landmarks)
                    parts[3] = 1
                    if result.pose_landmarks:
                        posture = classify_posture(result.pose_landmarks[0])
                    posture = f"{posture} ({pose_count} persons)"
            else:
                try:
                    result = self.holisticLandmarker.detect_for_video(mp_image, self.timestamp_ms)
                except Exception:
                    break

                if result.face_landmarks:
                    draw_face(frame, result.face_landmarks)
                    total_lm += len(result.face_landmarks)
                    parts[0] = 1

                if result.left_hand_landmarks:
                    draw_hand(frame, result.left_hand_landmarks, LHAND_C)
                    total_lm += len(result.left_hand_landmarks)
                    parts[1] = 1

                if result.right_hand_landmarks:
                    draw_hand(frame, result.right_hand_landmarks, RHAND_C)
                    total_lm += len(result.right_hand_landmarks)
                    parts[2] = 1

                if result.pose_landmarks:
                    draw_pose(frame, result.pose_landmarks)
                    total_lm += len(result.pose_landmarks)
                    parts[3] = 1
                    posture = classify_posture(result.pose_landmarks)

                if result.face_blendshapes:
                    for bs in result.face_blendshapes:
                        bs_dict[bs.category_name] = bs.score
                    emotion = classify_emotion(bs_dict)

                if parts[1] or parts[2]:
                    hands_text = f"{'Left ✓' if parts[1] else 'Left ✗'}  ·  {'Right ✓' if parts[2] else 'Right ✗'}"

            entry = f"{emotion} | {posture}"
            self._add_history(entry)

            draw_overlay(frame, sum(parts), total_lm)

            now = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (now - prev)
            prev = now

            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            try:
                self._queue.put_nowait({
                    "frame": display,
                    "parts": parts,
                    "emotion": emotion,
                    "posture": posture,
                    "hands": hands_text,
                    "fps": fps,
                    "lm": total_lm,
                    "history": list(self.history),
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
                self.status_widget.update_state(d["parts"])
                self.emotion_var.set(d["emotion"])
                self.posture_var.set(d["posture"])
                self.hands_var.set(d["hands"])
                self.fps_var.set(f"FPS: {d['fps']:.1f}")
                self.lm_var.set(f"Landmarks: {d['lm']}")
                self.history_var.set("\n".join(d["history"]))
        except queue.Empty:
            pass
        self.root.after(30, self._poll_queue)

    def _add_history(self, entry):
        if not entry or entry == self.last_entry:
            return
        self.last_entry = entry
        ts = datetime.now().strftime("%H:%M:%S")
        self.history.appendleft(f"[{ts}]  {entry}")

    def on_close(self):
        self.stop()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = FullBodyApp(root)
    root.mainloop()
