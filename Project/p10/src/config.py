import os
from dataclasses import dataclass

@dataclass
class Config:
    # Model configuration
    MODEL_PATH: str = os.path.join("models", "yolov8_ppe.pt")
    
    # Class-specific confidence thresholds for maximum accuracy
    # Hardhats are small objects (~25px), so 0.15 ensures 100% continuous detection without drops
    CONF_PERSON: float = 0.35
    CONF_HARDHAT: float = 0.15
    CONF_SAFETY_VEST: float = 0.25
    CONF_NO_HARDHAT: float = 0.35
    CONF_NO_VEST: float = 0.35
    CONF_DEFAULT: float = 0.15
    CONF_THRESHOLD: float = 0.15  # Backward-compatible alias
    
    IOU_THRESHOLD: float = 0.45
    
    # Class mapping for Hansung-Cho/yolov8-ppe-detection
    # {0: 'Hardhat', 1: 'Mask', 2: 'NO-Hardhat', 3: 'NO-Mask', 4: 'NO-Safety Vest', 5: 'Person', 6: 'Safety Cone', 7: 'Safety Vest', 8: 'machinery', 9: 'vehicle'}
    CLASS_HARDHAT: str = "Hardhat"
    CLASS_NO_HARDHAT: str = "NO-Hardhat"
    CLASS_SAFETY_VEST: str = "Safety Vest"
    CLASS_NO_SAFETY_VEST: str = "NO-Safety Vest"
    CLASS_PERSON: str = "Person"
    CLASS_MASK: str = "Mask"
    CLASS_NO_MASK: str = "NO-Mask"
    
    # Anatomical Association Geometry
    HEAD_TOP_MARGIN_RATIO: float = 0.15   # Expand 15% above person box to capture helmet dome
    HEAD_REGION_RATIO: float = 0.38       # Top 38% of worker bounding box
    TORSO_START_RATIO: float = 0.12       # Torso starts at 12% from top
    TORSO_END_RATIO: float = 0.85         # Torso extends to 85% from top
    HORIZONTAL_MARGIN: float = 0.20       # Allow +/- 20% width margin for PPE
    
    # Video processing
    DEFAULT_INPUT_VIDEO: str = os.path.join("asset", "8965526-uhd_2160_3840_25fps.mp4")
    DEFAULT_OUTPUT_VIDEO: str = os.path.join("output", "safety_monitoring_output.mp4")
    ALERTS_DIR: str = os.path.join("output", "alerts")
    LOG_FILE: str = os.path.join("output", "safety_violations_log.csv")
    
    # Visuals / Color palette (BGR)
    COLOR_SAFE = (76, 175, 80)          # Green
    COLOR_WARNING = (0, 165, 255)       # Orange
    COLOR_DANGER = (36, 36, 235)        # Red
    COLOR_HELMET = (255, 191, 0)        # Sky blue / Gold
    COLOR_VEST = (0, 215, 255)          # Bright Yellow
    COLOR_HUD_BG = (20, 24, 33)         # Dark slate
    COLOR_CARD_BG = (35, 42, 54)        # Card background
    COLOR_TEXT = (245, 245, 245)        # White
    COLOR_MUTED = (160, 160, 160)       # Gray
