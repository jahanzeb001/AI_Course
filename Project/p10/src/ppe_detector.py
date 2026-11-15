import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
from ultralytics import YOLO
from src.config import Config

class PPEDetector:
    def __init__(self, model_path: str = Config.MODEL_PATH, conf: float = None):
        self.model = YOLO(model_path)
        self.base_conf = conf if conf is not None else Config.CONF_DEFAULT
        self.names = self.model.names
        print(f"[PPEDetector] Initialized YOLO PPE model with {len(self.names)} classes: {self.names}")

    def detect_and_track(self, frame: np.ndarray, persist: bool = True) -> List[Dict[str, Any]]:
        """
        Runs tracking and detection on a single frame.
        Associates PPE equipment (Hardhat, Vest) with workers.
        Returns a list of analyzed workers with safety compliance details.
        """
        # Run tracking using ByteTrack with low base confidence to avoid dropping small hardhats
        results = self.model.track(
            frame,
            persist=persist,
            tracker="bytetrack.yaml",
            conf=self.base_conf,
            verbose=False
        )
        
        if not results or len(results) == 0:
            return []

        result = results[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []

        persons = []
        equipments = []

        for box in boxes:
            cls_id = int(box.cls[0])
            conf_val = float(box.conf[0])
            cls_name = self.names.get(cls_id, str(cls_id))
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            track_id = int(box.id[0]) if (box.id is not None and len(box.id) > 0) else None

            # Apply per-class confidence thresholding
            if cls_name == Config.CLASS_PERSON and conf_val < Config.CONF_PERSON:
                continue
            elif cls_name == Config.CLASS_HARDHAT and conf_val < Config.CONF_HARDHAT:
                continue
            elif cls_name == Config.CLASS_SAFETY_VEST and conf_val < Config.CONF_SAFETY_VEST:
                continue
            elif cls_name == Config.CLASS_NO_HARDHAT and conf_val < Config.CONF_NO_HARDHAT:
                continue
            elif cls_name == Config.CLASS_NO_SAFETY_VEST and conf_val < Config.CONF_NO_VEST:
                continue

            item = {
                "id": track_id,
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": conf_val,
                "bbox": [int(v) for v in xyxy]
            }

            if cls_name == Config.CLASS_PERSON:
                persons.append(item)
            else:
                equipments.append(item)

        # Associate equipment with each worker
        analyzed_workers = []
        for idx, person in enumerate(persons):
            worker_info = self._analyze_worker_ppe(person, equipments, fallback_id=idx + 1)
            analyzed_workers.append(worker_info)

        return analyzed_workers

    def _analyze_worker_ppe(self, person: Dict[str, Any], equipments: List[Dict[str, Any]], fallback_id: int) -> Dict[str, Any]:
        px1, py1, px2, py2 = person["bbox"]
        pw = px2 - px1
        ph = py2 - py1
        worker_id = person["id"] if person["id"] is not None else fallback_id

        # Define Head Region (with upward expansion for helmet dome)
        hx1 = px1 - int(pw * Config.HORIZONTAL_MARGIN)
        hx2 = px2 + int(pw * Config.HORIZONTAL_MARGIN)
        hy1 = py1 - int(ph * Config.HEAD_TOP_MARGIN_RATIO)
        hy2 = py1 + int(ph * Config.HEAD_REGION_RATIO)

        # Define Torso Region
        tx1 = px1 - int(pw * Config.HORIZONTAL_MARGIN)
        tx2 = px2 + int(pw * Config.HORIZONTAL_MARGIN)
        ty1 = py1 + int(ph * Config.TORSO_START_RATIO)
        ty2 = py1 + int(ph * Config.TORSO_END_RATIO)

        has_hardhat = False
        explicit_no_hardhat = False
        hardhat_bbox = None
        hardhat_conf = 0.0

        has_vest = False
        explicit_no_vest = False
        vest_bbox = None
        vest_conf = 0.0

        for eq in equipments:
            eq_name = eq["class_name"]
            ex1, ey1, ex2, ey2 = eq["bbox"]
            ecx = (ex1 + ex2) // 2
            ecy = (ey1 + ey2) // 2

            # Check overlap or center in Head Region
            in_head_x = (hx1 <= ecx <= hx2) or not (ex2 < hx1 or ex1 > hx2)
            in_head_y = (hy1 <= ecy <= hy2) or not (ey2 < hy1 or ey1 > hy2)
            is_in_head = (hx1 <= ecx <= hx2 and hy1 <= ecy <= hy2) or (in_head_x and in_head_y)

            if is_in_head:
                if eq_name == Config.CLASS_HARDHAT:
                    has_hardhat = True
                    if eq["conf"] > hardhat_conf:
                        hardhat_bbox = eq["bbox"]
                        hardhat_conf = eq["conf"]
                elif eq_name == Config.CLASS_NO_HARDHAT and eq["conf"] >= Config.CONF_NO_HARDHAT:
                    explicit_no_hardhat = True

            # Check overlap or center in Torso Region
            in_torso_x = (tx1 <= ecx <= tx2) or not (ex2 < tx1 or ex1 > tx2)
            in_torso_y = (ty1 <= ecy <= ty2) or not (ey2 < ty1 or ey1 > ty2)
            is_in_torso = (tx1 <= ecx <= tx2 and ty1 <= ecy <= ty2) or (in_torso_x and in_torso_y)

            if is_in_torso:
                if eq_name == Config.CLASS_SAFETY_VEST:
                    has_vest = True
                    if eq["conf"] > vest_conf:
                        vest_bbox = eq["bbox"]
                        vest_conf = eq["conf"]
                elif eq_name == Config.CLASS_NO_SAFETY_VEST and eq["conf"] >= Config.CONF_NO_VEST:
                    explicit_no_vest = True

        # PPE Compliance status
        missing_hardhat = explicit_no_hardhat or (not has_hardhat)
        missing_vest = explicit_no_vest or (not has_vest)

        violations = []
        if missing_hardhat:
            violations.append("NO HELMET")
        if missing_vest:
            violations.append("NO SAFETY VEST")

        is_safe = len(violations) == 0

        # Severity
        if is_safe:
            status = "SAFE"
            severity = "OK"
        elif len(violations) >= 2:
            status = "CRITICAL VIOLATION"
            severity = "CRITICAL"
        else:
            status = f"WARNING: {violations[0]}"
            severity = "WARNING"

        return {
            "worker_id": worker_id,
            "bbox": person["bbox"],
            "conf": person["conf"],
            "has_hardhat": has_hardhat,
            "has_vest": has_vest,
            "missing_hardhat": missing_hardhat,
            "missing_vest": missing_vest,
            "is_safe": is_safe,
            "violations": violations,
            "status": status,
            "severity": severity,
            "hardhat_bbox": hardhat_bbox,
            "vest_bbox": vest_bbox,
            "head_region": [hx1, hy1, hx2, hy2],
            "torso_region": [tx1, ty1, tx2, ty2]
        }
