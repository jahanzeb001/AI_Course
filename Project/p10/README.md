# Construction Site Safety Monitoring System (YOLO PPE Detection)

An AI-powered construction site safety monitoring system built using **Ultralytics YOLO**, **OpenCV**, and **ByteTrack**. The system monitors construction site video feeds to detect workers, verify Personal Protective Equipment (**Hardhat / Safety Helmet** and **Safety Vest**), identify safety violations in real time, and trigger hazard warning alerts with snapshot evidence and audit logging.

---

## Key Features

- **PPE Detection**: Real-time detection of:
  - Workers (`Person`)
  - Helmets (`Hardhat` & `NO-Hardhat`)
  - Safety Vests (`Safety Vest` & `NO-Safety Vest`)
- **Spatial PPE Association**: Evaluates anatomical regions (head region for hardhats, torso region for vests) to determine individual worker compliance.
- **Worker Tracking**: Integrates ByteTrack multi-object tracking to assign persistent Worker IDs (`Worker #1`, `Worker #2`, etc.) across video frames.
- **Temporal Status Smoothing**: Multi-frame smoothing window to prevent single-frame flickering or occlusion false alarms.
- **Real-Time HUD Dashboard**:
  - Live compliance metrics: Total Workers, Compliant (Safe) count, Violations count, Safety Compliance Rate (%).
  - Active hazard warning banners for immediate risk notification.
  - Recent incident alert ticker.
- **Automated Incident Logging & Alert Snapshots**:
  - Logs every violation incident to `output/safety_violations_log.csv` with timestamps and confidence scores.
  - Automatically captures high-resolution cropped alert snapshots in `output/alerts/`.

---

## Project Structure

```
p10/
├── asset/
│   └── 8965526-uhd_2160_3840_25fps.mp4  # Input construction video
├── models/
│   └── yolov8_ppe.pt                    # Pretrained YOLO PPE model weights
├── output/
│   ├── alerts/                          # Automated violation snapshot captures
│   ├── safety_monitoring_output.mp4     # Rendered output video with HUD
│   └── safety_violations_log.csv        # Incident audit report
├── src/
│   ├── __init__.py
│   ├── config.py                        # System settings, thresholds & color palettes
│   ├── ppe_detector.py                  # YOLO detection & spatial PPE association
│   ├── tracker.py                       # ByteTrack manager & temporal smoothing
│   ├── alert_system.py                  # Hazard alert engine & CSV audit logging
│   └── visualizer.py                    # Industrial HUD dashboard & bounding boxes
├── main.py                              # Main execution pipeline
├── requirements.txt                     # Dependencies
└── README.md                            # Documentation
```

---

## Virtual Environment Activation Guide

### 1. PowerShell (Default in VS Code)
```powershell
.\.venv\Scripts\Activate.ps1
```
*Note: If you encounter an execution policy error, run:*
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Command Prompt (CMD)
```cmd
.venv\Scripts\activate.bat
```

### 3. Git Bash
```bash
source .venv/Scripts/activate
```

---

## How to Run

### Run on Default Video:
```bash
python main.py
```

### Command-line Options:
```bash
# Run with a custom video source and custom output path
python main.py --source path/to/video.mp4 --output output/custom_result.mp4

# Adjust confidence threshold (default: 0.25)
python main.py --conf 0.35

# Limit number of frames processed (useful for rapid testing)
python main.py --max-frames 100

# Adjust scaling factor (default: 0.35 for fast processing of 4K video)
python main.py --scale 0.5

# Display live OpenCV preview window
python main.py --display
```

---

## Safety Compliance & Alert Hierarchy

| Status | Condition | Bounding Box | Alert Banner | Action |
| :--- | :--- | :--- | :--- | :--- |
| **SAFE** | Helmet OK + Vest OK | **Green** | None | Normal site operation |
| **WARNING** | Missing Helmet OR Missing Vest | **Red / Orange** | Active Warning Banner | Alert supervisor; notify worker |
| **CRITICAL** | Missing BOTH Helmet & Vest | **Flashing Red** | Critical Hazard Banner | Immediate work stoppage required |

---

## Audit Logs & Evidence

When a safety violation is detected, an alert event is logged:
1. **CSV Log (`output/safety_violations_log.csv`)**: Contains `Timestamp_Sec`, `Frame_Index`, `Worker_ID`, `Violation_Type`, `Severity`, `Confidence`, and `Snapshot_File`.
2. **Alert Snapshot (`output/alerts/`)**: High-resolution image crop of the non-compliant worker with a safety margin around the bounding box.
