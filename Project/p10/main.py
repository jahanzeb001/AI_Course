import os
import sys
import subprocess

# Auto-redirect to .venv python if invoked with system python
venv_python = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".venv", "Scripts", "python.exe")
if os.path.exists(venv_python):
    try:
        is_same = os.path.samefile(sys.executable, venv_python)
    except Exception:
        is_same = os.path.abspath(sys.executable).lower() == os.path.abspath(venv_python).lower()
    if not is_same:
        sys.exit(subprocess.call([venv_python] + sys.argv))

import time
import argparse
import cv2
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.config import Config
from src.ppe_detector import PPEDetector
from src.tracker import WorkerTrackerManager
from src.alert_system import AlertSystem
from src.visualizer import Visualizer

def parse_args():
    parser = argparse.ArgumentParser(description="Construction Site PPE Safety Monitoring System")
    parser.add_argument("--source", type=str, default=Config.DEFAULT_INPUT_VIDEO, help="Path to input video file")
    parser.add_argument("--output", type=str, default=Config.DEFAULT_OUTPUT_VIDEO, help="Path to save annotated video")
    parser.add_argument("--conf", type=float, default=Config.CONF_THRESHOLD, help="YOLO confidence threshold (default: 0.25)")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum number of frames to process (None for full video)")
    parser.add_argument("--display", action="store_true", help="Display live video window during processing")
    parser.add_argument("--display-height", type=int, default=720, help="Window display height in pixels to fit your monitor (default: 720)")
    parser.add_argument("--scale", type=float, default=0.3333, help="Resize scale factor (default: 0.3333 produces 720x1280 standard vertical HD)")
    parser.add_argument("--skip-frames", type=int, default=1, help="Process every N-th frame (1 = every frame)")
    return parser.parse_args()

def run_monitoring_system(source: str, output: str, conf: float = 0.25, max_frames: int = None, display: bool = False, display_height: int = 720, scale: float = 0.3333, skip_frames: int = 1):
    print("=" * 70)
    print("  CONSTRUCTION SITE PPE SAFETY MONITORING SYSTEM")
    print("=" * 70)
    print(f"[*] Input Source:      {source}")
    print(f"[*] Output Video:      {output}")
    print(f"[*] Confidence Thresh: {conf}")
    print(f"[*] Display Window:    {'Enabled (Height: ' + str(display_height) + 'px)' if display else 'Disabled'}")

    if not os.path.exists(source):
        print(f"[!] Error: Source video '{source}' does not exist.")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[!] Error: Unable to open video source '{source}'.")
        return

    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_w = int(orig_w * scale)
    out_h = int(orig_h * scale)
    # Ensure even dimensions for video codecs
    out_w = out_w if out_w % 2 == 0 else out_w - 1
    out_h = out_h if out_h % 2 == 0 else out_h - 1

    print(f"[*] Original Video:    {orig_w}x{orig_h} @ {fps:.1f} FPS | Total Frames: {total_video_frames}")
    print(f"[*] Processed Res:     {out_w}x{out_h} @ {fps / skip_frames:.1f} FPS")

    os.makedirs(os.path.dirname(output), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_writer = cv2.VideoWriter(output, fourcc, fps / skip_frames, (out_w, out_h))

    # Calculate preview window size to fit screen
    disp_h = display_height
    disp_w = int(out_w * (disp_h / out_h))
    if display:
        cv2.namedWindow("Safety Monitoring System", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Safety Monitoring System", disp_w, disp_h)

    # Initialize modules
    detector = PPEDetector(conf=conf)
    tracker = WorkerTrackerManager(history_len=5)
    alert_system = AlertSystem(cooldown_seconds=1.5)
    visualizer = Visualizer()

    frame_idx = 0
    processed_count = 0
    start_time = time.time()
    total_violations_logged = 0
    all_workers_recorded = set()

    print("\n[*] Starting video analysis and safety monitoring...\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if max_frames and frame_idx > max_frames:
                break

            if skip_frames > 1 and (frame_idx % skip_frames != 0):
                continue

            # Resize frame for efficient processing & HUD rendering
            if (orig_w, orig_h) != (out_w, out_h):
                frame_resized = cv2.resize(frame, (out_w, out_h))
            else:
                frame_resized = frame

            # 1. Detection & ByteTrack tracking
            raw_workers = detector.detect_and_track(frame_resized, persist=True)

            # 2. Temporal Smoothing & Stats
            smoothed_workers = tracker.update_workers(raw_workers)

            for w in smoothed_workers:
                all_workers_recorded.add(w["worker_id"])

            # 3. Alert Processing & Snapshot Saving
            new_alerts = alert_system.process_alerts(smoothed_workers, frame_resized, frame_idx, fps)
            total_violations_logged += len(new_alerts)

            for alert in new_alerts:
                print(f"  [ALERT @ {alert['time_sec']:.1f}s] Frame {frame_idx}: Worker #{alert['worker_id']} -> {alert['violation_str']} ({alert['severity']})")

            # 4. Render HUD & Bounding Boxes
            annotated_frame = visualizer.draw_hud(
                frame_resized,
                smoothed_workers,
                alert_system.get_recent_alerts(),
                frame_idx,
                fps
            )

            # 5. Write to output video
            out_writer.write(annotated_frame)
            processed_count += 1

            if display:
                preview_frame = cv2.resize(annotated_frame, (disp_w, disp_h))
                cv2.imshow("Safety Monitoring System", preview_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[*] Processing stopped by user.")
                    break

            # Progress log every 25 frames
            if processed_count % 25 == 0:
                elapsed = time.time() - start_time
                curr_fps = processed_count / elapsed
                pct = (frame_idx / total_video_frames) * 100 if total_video_frames > 0 else 0
                print(f"  [Progress] Frame {frame_idx}/{total_video_frames} ({pct:.1f}%) | {curr_fps:.2f} FPS | Active Workers: {len(smoothed_workers)}")

    finally:
        cap.release()
        out_writer.release()
        if display:
            cv2.destroyAllWindows()

    total_time = time.time() - start_time
    avg_fps = processed_count / total_time if total_time > 0 else 0

    print("\n" + "=" * 70)
    print("  MONITORING PIPELINE COMPLETED")
    print("=" * 70)
    print(f"[*] Frames Processed:      {processed_count}")
    print(f"[*] Total Time:            {total_time:.2f} seconds ({avg_fps:.2f} FPS)")
    print(f"[*] Output Video Saved:    {output}")
    print(f"[*] Total Unique Workers:  {len(all_workers_recorded)}")
    print(f"[*] Total Violations:      {total_violations_logged}")
    print(f"[*] Incident Log:          {Config.LOG_FILE}")
    print(f"[*] Alert Snapshots:       {Config.ALERTS_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    args = parse_args()
    run_monitoring_system(
        source=args.source,
        output=args.output,
        conf=args.conf,
        max_frames=args.max_frames,
        display=args.display,
        display_height=args.display_height,
        scale=args.scale,
        skip_frames=args.skip_frames
    )
