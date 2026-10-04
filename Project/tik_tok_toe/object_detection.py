import os
import cv2
import numpy as np

# ==========================================
# COCO CLASS NAMES (YOLO11)
# ==========================================
CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

# Generate unique color for each class
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(CLASSES), 3), dtype=np.uint8)

# ==========================================
# LOAD YOLO11 MODEL VIA OPENCV DNN
# ==========================================
# Uses OpenCV DNN directly - no PyTorch/AVX2 crashes, fast CPU inference
model_path = os.path.join(os.path.dirname(__file__), "yolo11n.onnx")

if not os.path.exists(model_path):
    print("Model file not found. Downloading yolo11n.onnx...")
    import urllib.request
    url = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.onnx"
    urllib.request.urlretrieve(url, model_path)
    print("Model downloaded successfully!")

print("Loading YOLO11 neural network...")
net = cv2.dnn.readNetFromONNX(model_path)
print("Model loaded successfully!")

# ==========================================
# OPEN LAPTOP CAMERA
# ==========================================
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Could not open camera.")
    exit()

print("\nStarting object detection... Press 'q' to quit.")

# ==========================================
# CAMERA LOOP
# ==========================================
CONF_THRESHOLD = 0.50
NMS_THRESHOLD = 0.40

while True:
    success, frame = camera.read()
    if not success:
        print("ERROR: Could not read camera frame.")
        break

    orig_h, orig_w = frame.shape[:2]

    # Preprocess image for YOLO (640x640, normalized to [0, 1], RGB)
    blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
    net.setInput(blob)

    # Forward pass: output shape is (1, 84, 8400)
    output = net.forward()
    predictions = output[0].T  # Transpose to (8400, 84)

    boxes = []
    confidences = []
    class_ids = []

    x_scale = orig_w / 640.0
    y_scale = orig_h / 640.0

    # Process detections
    for row in predictions:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])

        if confidence >= CONF_THRESHOLD:
            # Box center coordinates, width and height
            cx, cy, w, h = row[0:4]

            # Convert to top-left x, y, width, height for OpenCV
            x1 = int((cx - w / 2.0) * x_scale)
            y1 = int((cy - h / 2.0) * y_scale)
            box_w = int(w * x_scale)
            box_h = int(h * y_scale)

            boxes.append([x1, y1, box_w, box_h])
            confidences.append(confidence)
            class_ids.append(class_id)

    # Apply Non-Maximum Suppression to eliminate overlapping boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        for idx in indices.flatten():
            x, y, w, h = boxes[idx]
            conf = confidences[idx]
            cls_id = class_ids[idx]

            class_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"Class {cls_id}"
            color = [int(c) for c in COLORS[cls_id % len(COLORS)]]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label text
            label = f"{class_name} {conf * 100:.1f}%"

            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2
            )

            # Prevent label from drawing off-screen at the top
            label_y = max(y, text_height + 10)

            # Label background
            cv2.rectangle(
                frame,
                (x, label_y - text_height - 10),
                (x + text_width, label_y),
                color,
                -1
            )

            # Label text (white or black depending on color brightness)
            text_color = (0, 0, 0) if (color[0] * 0.299 + color[1] * 0.587 + color[2] * 0.114) > 150 else (255, 255, 255)
            cv2.putText(
                frame,
                label,
                (x, label_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2
            )

    # Display video feed
    cv2.imshow("AI Object Detection (YOLO11) - Press Q to Quit", frame)

    # Check for quit key
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Cleanup
camera.release()
cv2.destroyAllWindows()
print("Camera closed.")
