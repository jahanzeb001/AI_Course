import os
import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
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
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "face_landmarker.task")

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
NOSE = [1, 2, 5, 4, 6, 19, 94, 168, 8, 9, 10, 151, 175, 147, 196, 127, 35, 164]

FEATURES = [
    (FACE_OVAL, (124, 58, 237), 2),
    (LEFT_EYE, (6, 182, 212), 2),
    (RIGHT_EYE, (6, 182, 212), 2),
    (LEFT_BROW, (245, 158, 11), 2),
    (RIGHT_BROW, (245, 158, 11), 2),
    (LIPS, (236, 72, 153), 2),
    (NOSE, (34, 197, 94), 1),
]

KEY_BLENDSHAPES = [
    "eyeBlinkLeft", "eyeBlinkRight", "jawOpen",
    "mouthSmileLeft", "mouthSmileRight",
    "browInnerUp", "eyeWideLeft", "eyeWideRight",
    "mouthFrownLeft", "mouthFrownRight",
    "cheekPuff", "noseSneerLeft", "noseSneerRight",
    "mouthPucker", "tongueOut",
]
BS_DISPLAY = 6

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


def ensure_model():
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 0:
        return MODEL_PATH
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def make_gradient(w, h, c1, c2):
    a1 = np.array([int(c1[1:3], 16), int(c1[3:5], 16), int(c1[5:7], 16)])
    a2 = np.array([int(c2[1:3], 16), int(c2[3:5], 16), int(c2[5:7], 16)])
    t = np.linspace(0, 1, h).reshape(-1, 1)
    grad = (a1 + (a2 - a1) * t).astype(np.uint8)
    grad = np.broadcast_to(grad[:, None, :], (h, w, 3)).copy()
    return Image.fromarray(grad)


def draw_face(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for lm in landmarks:
        x, y = int(lm.x * w), int(lm.y * h)
        cv2.circle(frame, (x, y), 1, (60, 60, 100), -1)
    for indices, color, thick in FEATURES:
        poly = [pts[i] for i in indices]
        cv2.polylines(frame, [np.array(poly, dtype=np.int32)], False, color, thick, cv2.LINE_AA)


def draw_bbox(frame, landmarks):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    cv2.rectangle(frame, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), (124, 58, 237), 2, cv2.LINE_AA)
    cv2.putText(frame, "FACE", (x1 - 8, y1 - 14), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (168, 85, 247), 1, cv2.LINE_AA)


def draw_overlay(frame, emotion, face_count):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (10, 10, 26), -1)
    cv2.rectangle(overlay, (0, h - 38), (w, h), (10, 10, 26), -1)
    frame[:] = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)
    label = emotion if emotion != "—" else "No face detected"
    color = (168, 85, 247) if emotion != "—" else (148, 163, 184)
    cv2.putText(frame, label, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {face_count}", (w - 120, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (6, 182, 212), 2, cv2.LINE_AA)


def classify_emotion(bs):
    smile = (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2
    frown = (bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 2
    jaw = bs.get("jawOpen", 0)
    eye_wide = (bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 2
    brow_down = (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0)) / 2
    brow_up = bs.get("browInnerUp", 0)
    blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
    left_blink = bs.get("eyeBlinkLeft", 0)
    right_blink = bs.get("eyeBlinkRight", 0)

    if left_blink > 0.5 and right_blink < 0.2:
        return "Wink 😉"
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
    if brow_down > 0.3 and jaw < 0.15:
        return "Angry 😠"
    if brow_up > 0.3 and jaw > 0.1:
        return "Surprised 😮"
    if jaw > 0.25:
        return "Mouth Open 😮"
    return "Neutral 😐"


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


class BlendshapeBar(tk.Canvas):
    def __init__(self, parent, label, width=270, height=22):
        super().__init__(parent, width=width, height=height, bg=BG_TOP,
                         highlightthickness=0)
        self._bw = width
        self._bh = height
        self._label = label
        self.set_value(0)

    def set_value(self, val):
        self.delete("all")
        self.create_text(4, self._bh // 2, text=self._label, anchor="w",
                         fill=MUTED, font=("Helvetica", 8))
        bar_x = 90
        bar_w = self._bw - bar_x - 30
        self._round(bar_x, 4, bar_x + bar_w, self._bh - 4, 4, fill="#0d0d1a", outline=CARD_BORDER)
        fw = int(bar_w * max(0, min(1, val)))
        if fw > 0:
            self._round(bar_x, 4, bar_x + fw, self._bh - 4, 4, fill=ACCENT2, outline="")
        self.create_text(self._bw - 4, self._bh // 2, text=f"{val*100:.0f}%",
                         anchor="e", fill=TEXT, font=("Helvetica", 8, "bold"))

    def _round(self, x0, y0, x1, y1, r, **kw):
        pts = [x0+r, y0, x1-r, y0, x1, y0, x1, y0+r, x1, y1-r, x1-r, y1,
               x0+r, y1, x0, y1, x0, y1-r, x0, y0+r, x0, y0]
        self.create_polygon(pts, smooth=True, **kw)


class FaceDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Face Detection & Expression")
        self.root.configure(bg=BG_TOP)
        self.root.geometry("1000x660")
        self.root.resizable(False, False)

        self.running = False
        self.cap = None
        self.faceLandmarker = None
        self.mirror = True
        self.num_faces = 1
        self.det_conf = 0.5
        self.track_conf = 0.5
        self.timestamp_ms = 0
        self.blink_count = 0
        self.was_blinking = False
        self.emotion_history = deque(maxlen=6)
        self.last_emotion = ""
        self.bs_bars = {}
        self._queue = queue.Queue(maxsize=2)
        self._polling = False

        self._build_ui()

    def _build_ui(self):
        self._bg_img = make_gradient(1000, 660, BG_TOP, BG_BOTTOM)
        self._bg_photo = ImageTk.PhotoImage(self._bg_img)
        self.bg_label = tk.Label(self.root, image=self._bg_photo)
        self.bg_label.place(x=0, y=0, relwidth=1, relheight=1)

        header = tk.Frame(self.root, bg=BG_TOP)
        header.pack(fill=tk.X, padx=18, pady=(16, 0))
        tk.Label(header, text="😀  AI Face Detection & Expression", bg=BG_TOP, fg=TEXT,
                 font=("Helvetica", 22, "bold")).pack(side=tk.LEFT)
        tk.Label(header, text="MediaPipe Tasks  ·  478 Landmarks  ·  Blendshapes",
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
        tk.Label(controls, text="Faces:", bg=BG_TOP, fg=MUTED,
                 font=("Helvetica", 11)).pack(side=tk.LEFT)
        self.spin_faces = tk.Spinbox(controls, from_=1, to=4, width=3, font=("Helvetica", 11),
                                     bg=CARD_BG, fg=TEXT, relief="flat", justify="center")
        self.spin_faces.pack(side=tk.LEFT, padx=(6, 16))
        self.spin_faces.delete(0, "end")
        self.spin_faces.insert(0, "1")

        self.var_mirror = tk.BooleanVar(value=True)
        tk.Checkbutton(controls, text="Mirror", variable=self.var_mirror,
                       bg=BG_TOP, fg=TEXT, selectcolor=CARD_BG, activebackground=BG_TOP,
                       activeforeground=TEXT, font=("Helvetica", 11),
                       highlightthickness=0).pack(side=tk.LEFT, padx=8)

        tk.Label(controls, text="Confidence:", bg=BG_TOP, fg=MUTED,
                 font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(8, 4))
        self.scale_det = tk.Scale(controls, from_=0.3, to=0.9, orient=tk.HORIZONTAL,
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

        self._section(info, "🎯  EMOTION")
        self.emotion_var = tk.StringVar(value="—")
        tk.Label(info, textvariable=self.emotion_var, bg=BG_TOP, fg=ACCENT2,
                 font=("Helvetica", 20, "bold"), wraplength=290,
                 justify="left").pack(anchor=tk.W, pady=(0, 4))

        self._sep(info)
        self._section(info, "📊  EXPRESSIONS")
        bs_frame = tk.Frame(info, bg=BG_TOP)
        bs_frame.pack(fill=tk.X, pady=(2, 0))
        for name in KEY_BLENDSHAPES[:BS_DISPLAY]:
            pretty = name.replace("eye", "Eye ").replace("mouth", "Mouth ") \
                         .replace("brow", "Brow ").replace("jaw", "Jaw ") \
                         .replace("cheek", "Cheek ").replace("nose", "Nose ") \
                         .replace("tongue", "Tongue ")
            pretty = pretty[0].upper() + pretty[1:]
            bar = BlendshapeBar(bs_frame, pretty)
            bar.pack(anchor=tk.W, pady=1)
            self.bs_bars[name] = bar

        self._sep(info)
        self._section(info, "👁  BLINKS")
        self.blink_var = tk.StringVar(value="0")
        tk.Label(info, textvariable=self.blink_var, bg=BG_TOP, fg=CYAN,
                 font=("Helvetica", 24, "bold")).pack(anchor=tk.W, pady=(0, 4))

        self._sep(info)
        self._section(info, "📈  STATS")
        stats = tk.Frame(info, bg=BG_TOP)
        stats.pack(fill=tk.X, pady=(2, 0))
        self.face_var = tk.StringVar(value="Faces: 0")
        self.fps_var = tk.StringVar(value="FPS: —")
        self.lm_var = tk.StringVar(value="Landmarks: 0")
        tk.Label(stats, textvariable=self.face_var, bg=BG_TOP, fg=TEXT,
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
            self.num_faces = int(self.spin_faces.get())
        except ValueError:
            self.num_faces = 1
        self.mirror = self.var_mirror.get()
        self.det_conf = float(self.scale_det.get())
        self.track_conf = 0.5

        self.status_var.set("  ● Loading AI model...")
        self.root.update_idletasks()
        try:
            model_path = ensure_model()
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

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=self.num_faces,
            min_face_detection_confidence=self.det_conf,
            min_face_presence_confidence=self.track_conf,
            min_tracking_confidence=self.track_conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.faceLandmarker = vision.FaceLandmarker.create_from_options(options)
        self.timestamp_ms = 0
        self.blink_count = 0
        self.was_blinking = False

        self.running = True
        self.btn_start.set_state("disabled")
        self.btn_stop.set_state("normal")
        self.status_var.set("  ● Running — look at the camera 😀")
        self._polling = True
        self._poll_queue()
        threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.faceLandmarker is not None:
            self.faceLandmarker.close()
            self.faceLandmarker = None
        self.btn_start.set_state("normal")
        self.btn_stop.set_state("disabled")
        self._polling = False
        self.status_var.set("  ● Stopped.")
        self.emotion_var.set("—")
        self.face_var.set("Faces: 0")
        self.fps_var.set("FPS: —")
        self.lm_var.set("Landmarks: 0")
        self.blink_var.set("0")
        self.history_var.set("—")
        for bar in self.bs_bars.values():
            bar.set_value(0)
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
                result = self.faceLandmarker.detect_for_video(mp_image, self.timestamp_ms)
            except Exception:
                break

            emotion_text = "—"
            face_count = 0
            lm_count = 0
            bs_dict = {}

            if result.face_landmarks:
                face_count = len(result.face_landmarks)
                for landmarks in result.face_landmarks:
                    draw_face(frame, landmarks)
                    draw_bbox(frame, landmarks)
                    lm_count = len(landmarks)

                if result.face_blendshapes:
                    for bs in result.face_blendshapes[0]:
                        bs_dict[bs.category_name] = bs.score
                    emotion_text = classify_emotion(bs_dict)
                    self._track_blink(bs_dict)
                    self._add_history(emotion_text)

            draw_overlay(frame, emotion_text, face_count)

            now = cv2.getTickCount()
            fps = cv2.getTickFrequency() / (now - prev)
            prev = now

            display = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
            display = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)

            try:
                self._queue.put_nowait({
                    "frame": display,
                    "emotion": emotion_text,
                    "faces": face_count,
                    "fps": fps,
                    "lm": lm_count,
                    "blinks": self.blink_count,
                    "bs": bs_dict,
                    "history": list(self.emotion_history),
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
                data = self._queue.get_nowait()
                img = Image.fromarray(data["frame"])
                photo = ImageTk.PhotoImage(image=img)
                self.video_label.img = photo
                self.video_label.config(image=photo)
                self.emotion_var.set(data["emotion"])
                self.face_var.set(f"Faces: {data['faces']}")
                self.fps_var.set(f"FPS: {data['fps']:.1f}")
                self.lm_var.set(f"Landmarks: {data['lm']}")
                self.blink_var.set(str(data["blinks"]))
                for name, bar in self.bs_bars.items():
                    bar.set_value(data["bs"].get(name, 0))
                self.history_var.set("\n".join(data["history"]))
        except queue.Empty:
            pass
        self.root.after(30, self._poll_queue)

    def _track_blink(self, bs):
        blink = (bs.get("eyeBlinkLeft", 0) + bs.get("eyeBlinkRight", 0)) / 2
        if blink > 0.5 and not self.was_blinking:
            self.was_blinking = True
        elif blink < 0.3 and self.was_blinking:
            self.was_blinking = False
            self.blink_count += 1

    def _add_history(self, emotion):
        if not emotion or emotion == "—" or emotion == self.last_emotion:
            return
        self.last_emotion = emotion
        ts = datetime.now().strftime("%H:%M:%S")
        self.emotion_history.appendleft(f"[{ts}]  {emotion}")

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
    app = FaceDetectionApp(root)
    root.mainloop()