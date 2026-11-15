import cv2
import numpy as np
from typing import List, Dict, Any
from src.config import Config

class Visualizer:
    def __init__(self):
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def draw_hud(self, frame: np.ndarray, workers: List[Dict[str, Any]], recent_alerts: List[Dict[str, Any]], frame_idx: int, fps: float) -> np.ndarray:
        """
        Renders the safety monitoring HUD and worker annotations with responsive layout.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        total_workers = len(workers)
        compliant_workers = sum(1 for w in workers if w.get("smoothed_is_safe", w["is_safe"]))
        violation_workers = total_workers - compliant_workers
        compliance_rate = (compliant_workers / total_workers * 100) if total_workers > 0 else 100.0

        # 1. Draw Worker Bounding Boxes & Badges
        for worker in workers:
            self._draw_worker(out, worker)

        # 2. Draw Top Dashboard / Header (Responsive for portrait and landscape)
        self._draw_header_dashboard(out, total_workers, compliant_workers, violation_workers, compliance_rate, frame_idx, fps)

        # 3. Draw Active Warning Banner if violations exist
        if violation_workers > 0:
            violating_ids = [str(w["worker_id"]) for w in workers if not w.get("smoothed_is_safe", w["is_safe"])]
            msg = f"SAFETY VIOLATION: WORKER #{', #'.join(violating_ids)} MISSING PPE!"
            self._draw_warning_banner(out, msg)

        # 4. Draw Recent Alerts Log Ticker
        if recent_alerts:
            self._draw_alert_ticker(out, recent_alerts)

        return out

    def _draw_worker(self, img: np.ndarray, worker: Dict[str, Any]):
        h, w = img.shape[:2]
        x1, y1, x2, y2 = worker["bbox"]
        # Ensure within image bounds
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        wid = worker["worker_id"]
        is_safe = worker.get("smoothed_is_safe", worker["is_safe"])
        has_hardhat = worker["has_hardhat"]
        has_vest = worker["has_vest"]

        color = Config.COLOR_SAFE if is_safe else Config.COLOR_DANGER

        # Main bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw Equipment boxes if detected
        if worker.get("hardhat_bbox") and has_hardhat:
            hx1, hy1, hx2, hy2 = worker["hardhat_bbox"]
            hx1, hy1 = max(0, hx1), max(0, hy1)
            hx2, hy2 = min(w - 1, hx2), min(h - 1, hy2)
            cv2.rectangle(img, (hx1, hy1), (hx2, hy2), Config.COLOR_HELMET, 2)
            cv2.putText(img, "HELMET", (hx1, max(14, hy1 - 3)), self.font, 0.40, Config.COLOR_HELMET, 1, cv2.LINE_AA)

        if worker.get("vest_bbox") and has_vest:
            vx1, vy1, vx2, vy2 = worker["vest_bbox"]
            vx1, vy1 = max(0, vx1), max(0, vy1)
            vx2, vy2 = min(w - 1, vx2), min(h - 1, vy2)
            cv2.rectangle(img, (vx1, vy1), (vx2, vy2), Config.COLOR_VEST, 2)
            cv2.putText(img, "VEST", (vx1, max(14, vy1 - 3)), self.font, 0.40, Config.COLOR_VEST, 1, cv2.LINE_AA)

        # Top Badge: WORKER #X | SAFE / VIOLATION
        status_text = f"WORKER #{wid} | {'SAFE' if is_safe else 'PPE VIOLATION'}"
        font_scale = 0.45
        (tw, th), _ = cv2.getTextSize(status_text, self.font, font_scale, 1)

        badge_y1 = max(0, y1 - th - 10)
        badge_y2 = y1
        badge_x1 = x1
        badge_x2 = min(w - 1, x1 + tw + 12)

        cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(img, status_text, (badge_x1 + 6, badge_y2 - 5), self.font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Bottom detail pills: HELMET and VEST status
        pill_y = min(h - 35, y2 + 15)
        h_text = "HELMET: OK" if has_hardhat else "HELMET: NO"
        h_color = Config.COLOR_SAFE if has_hardhat else Config.COLOR_DANGER
        self._draw_pill(img, x1, pill_y, h_text, h_color)

        v_text = "VEST: OK" if has_vest else "VEST: NO"
        v_color = Config.COLOR_SAFE if has_vest else Config.COLOR_DANGER
        self._draw_pill(img, x1, pill_y + 18, v_text, v_color)

    def _draw_pill(self, img: np.ndarray, x: int, y: int, text: str, bg_color: tuple):
        h, w = img.shape[:2]
        font_scale = 0.40
        (tw, th), _ = cv2.getTextSize(text, self.font, font_scale, 1)
        pad_x, pad_y = 5, 3
        
        px1 = max(0, min(w - tw - pad_x * 2 - 2, x))
        py1 = max(0, min(h - th - pad_y * 2 - 2, y - th - pad_y))
        px2 = px1 + tw + pad_x * 2
        py2 = py1 + th + pad_y * 2

        overlay = img.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), bg_color, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        cv2.putText(img, text, (px1 + pad_x, py2 - pad_y), self.font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_header_dashboard(self, img: np.ndarray, total: int, compliant: int, violations: int, rate: float, frame_idx: int, fps: float):
        h, w = img.shape[:2]
        panel_h = 100

        # Background overlay for the top HUD
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), Config.COLOR_HUD_BG, -1)
        cv2.line(overlay, (0, panel_h), (w, panel_h), (60, 70, 85), 2)
        cv2.addWeighted(overlay, 0.88, img, 0.12, 0, img)

        # Top line: Title & Time info
        cv2.circle(img, (18, 22), 5, (0, 0, 255), -1)
        cv2.putText(img, "LIVE", (28, 26), self.font, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(img, "PPE SAFETY MONITOR", (72, 26), self.font, 0.55, Config.COLOR_TEXT, 2, cv2.LINE_AA)

        current_sec = frame_idx / fps if fps > 0 else 0
        time_str = f"Time: {int(current_sec//60):02d}:{int(current_sec%60):02d}.{int((current_sec%1)*10):01d}"
        (tw_t, _), _ = cv2.getTextSize(time_str, self.font, 0.42, 1)
        cv2.putText(img, time_str, (w - tw_t - 15, 26), self.font, 0.42, Config.COLOR_MUTED, 1, cv2.LINE_AA)

        # Second line: 4 Metric Cards evenly spaced across full width w
        margin_x = 10
        gap_x = 8
        card_y = 38
        card_h = 52
        num_cards = 4
        card_w = (w - 2 * margin_x - (num_cards - 1) * gap_x) // num_cards

        cards = [
            ("WORKERS", str(total), Config.COLOR_TEXT),
            ("SAFE", str(compliant), Config.COLOR_SAFE),
            ("VIOLATIONS", str(violations), Config.COLOR_DANGER if violations > 0 else Config.COLOR_SAFE),
            ("SAFETY", f"{rate:.0f}%", Config.COLOR_SAFE if rate >= 80 else Config.COLOR_DANGER)
        ]

        for i, (title, value, val_color) in enumerate(cards):
            cx = margin_x + i * (card_w + gap_x)
            cv2.rectangle(img, (cx, card_y), (cx + card_w, card_y + card_h), Config.COLOR_CARD_BG, -1)
            cv2.rectangle(img, (cx, card_y), (cx + card_w, card_y + card_h), (60, 70, 85), 1)

            # Center text inside card
            (tw_title, _), _ = cv2.getTextSize(title, self.font, 0.35, 1)
            title_x = cx + max(4, (card_w - tw_title) // 2)
            cv2.putText(img, title, (title_x, card_y + 16), self.font, 0.35, Config.COLOR_MUTED, 1, cv2.LINE_AA)

            (tw_val, _), _ = cv2.getTextSize(value, self.font, 0.65, 2)
            val_x = cx + max(4, (card_w - tw_val) // 2)
            cv2.putText(img, value, (val_x, card_y + 42), self.font, 0.65, val_color, 2, cv2.LINE_AA)

    def _draw_warning_banner(self, img: np.ndarray, message: str):
        h, w = img.shape[:2]
        banner_y1 = 104
        banner_y2 = 138

        # Alert banner
        overlay = img.copy()
        cv2.rectangle(overlay, (0, banner_y1), (w, banner_y2), Config.COLOR_DANGER, -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        text = f"WARNING [ALERT]: {message}"
        font_scale = 0.48
        (tw, th), _ = cv2.getTextSize(text, self.font, font_scale, 1)
        tx = max(10, (w - tw) // 2)
        cv2.putText(img, text, (tx, banner_y1 + 23), self.font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_alert_ticker(self, img: np.ndarray, alerts: List[Dict[str, Any]]):
        h, w = img.shape[:2]
        panel_w = min(w - 20, 360)
        item_h = 24
        num_alerts = min(3, len(alerts))
        panel_h = num_alerts * item_h + 30
        
        px1 = w - panel_w - 10
        py1 = h - panel_h - 10
        px2 = w - 10
        py2 = h - 10

        overlay = img.copy()
        cv2.rectangle(overlay, (px1, py1), (px2, py2), Config.COLOR_HUD_BG, -1)
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (60, 70, 85), 1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

        cv2.putText(img, "RECENT SAFETY ALERTS", (px1 + 10, py1 + 18), self.font, 0.40, (0, 165, 255), 1, cv2.LINE_AA)

        for i, alert in enumerate(alerts[:num_alerts]):
            iy = py1 + 36 + i * item_h
            t_sec = alert.get("time_sec", 0.0)
            msg = f"[{int(t_sec//60):02d}:{int(t_sec%60):02d}] #{alert['worker_id']}: {alert['violation_str']}"
            cv2.circle(img, (px1 + 15, iy - 4), 3, Config.COLOR_DANGER, -1)
            cv2.putText(img, msg, (px1 + 25, iy), self.font, 0.38, Config.COLOR_TEXT, 1, cv2.LINE_AA)
