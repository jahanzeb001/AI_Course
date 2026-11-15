import os
import csv
import time
import cv2
import numpy as np
from typing import List, Dict, Any, Optional
from src.config import Config

class AlertSystem:
    def __init__(self, log_path: str = Config.LOG_FILE, alerts_dir: str = Config.ALERTS_DIR, cooldown_seconds: float = 2.0):
        self.log_path = log_path
        self.alerts_dir = alerts_dir
        self.cooldown_seconds = cooldown_seconds
        
        # Last alert time per worker_id
        self.last_alert_time: Dict[int, float] = {}
        # Active alert messages for on-screen HUD ticker
        self.active_alerts: List[Dict[str, Any]] = []
        
        os.makedirs(self.alerts_dir, exist_ok=True)
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        
        # Initialize CSV header if not exists
        if not os.path.exists(self.log_path):
            with open(self.log_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Timestamp_Sec",
                    "Frame_Index",
                    "Worker_ID",
                    "Violation_Type",
                    "Severity",
                    "Confidence",
                    "Snapshot_File"
                ])

    def process_alerts(self, workers: List[Dict[str, Any]], frame: np.ndarray, frame_idx: int, fps: float) -> List[Dict[str, Any]]:
        """
        Evaluates violations, emits warning alerts, captures snapshots, and logs to CSV.
        """
        current_time_sec = frame_idx / fps if fps > 0 else frame_idx / 25.0
        frame_alerts = []

        for worker in workers:
            is_safe = worker.get("smoothed_is_safe", worker["is_safe"])
            if is_safe:
                continue

            wid = worker["worker_id"]
            violations = worker.get("smoothed_violations", worker["violations"])
            severity = worker.get("smoothed_severity", worker["severity"])
            violation_str = " & ".join(violations)

            last_time = self.last_alert_time.get(wid, -999.0)
            if (current_time_sec - last_time) >= self.cooldown_seconds:
                # Trigger alert!
                self.last_alert_time[wid] = current_time_sec
                
                # Save snapshot of the worker violation
                snapshot_filename = f"alert_f{frame_idx:04d}_worker_{wid}.jpg"
                snapshot_path = os.path.join(self.alerts_dir, snapshot_filename)
                
                # Crop worker region with margin for context
                h, w = frame.shape[:2]
                bx1, by1, bx2, by2 = worker["bbox"]
                margin_x = int((bx2 - bx1) * 0.25)
                margin_y = int((by2 - by1) * 0.20)
                cx1 = max(0, bx1 - margin_x)
                cy1 = max(0, by1 - margin_y)
                cx2 = min(w, bx2 + margin_x)
                cy2 = min(h, by2 + margin_y)
                
                worker_crop = frame[cy1:cy2, cx1:cx2]
                if worker_crop.size > 0:
                    cv2.imwrite(snapshot_path, worker_crop)

                # Write to CSV log
                with open(self.log_path, mode="a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        f"{current_time_sec:.2f}",
                        frame_idx,
                        wid,
                        violation_str,
                        severity,
                        f"{worker['conf']:.2f}",
                        snapshot_filename
                    ])

                alert_event = {
                    "worker_id": wid,
                    "violations": violations,
                    "violation_str": violation_str,
                    "severity": severity,
                    "time_sec": current_time_sec,
                    "frame_idx": frame_idx,
                    "snapshot": snapshot_filename,
                    "message": f"Worker #{wid}: {violation_str}"
                }
                frame_alerts.append(alert_event)

                # Maintain recent HUD active alerts
                self.active_alerts.insert(0, alert_event)
                if len(self.active_alerts) > 5:
                    self.active_alerts.pop()

        return frame_alerts

    def get_recent_alerts(self) -> List[Dict[str, Any]]:
        return self.active_alerts
