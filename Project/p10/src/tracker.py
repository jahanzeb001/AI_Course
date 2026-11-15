from typing import Dict, List, Any
from collections import deque

class WorkerTrackerManager:
    def __init__(self, history_len: int = 10, grace_frames: int = 15):
        self.history_len = history_len
        self.grace_frames = grace_frames
        
        # Maps worker_id to violation history deque
        self.worker_history: Dict[int, deque] = {}
        # Stores cumulative stats
        self.worker_stats: Dict[int, Dict[str, Any]] = {}
        # Tracks frames since last seen with hardhat / vest
        self.worker_ppe_memory: Dict[int, Dict[str, int]] = {}

    def update_workers(self, detected_workers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Applies PPE temporal memory and smoothing over time.
        Eliminates single-frame detector drops while quickly catching real violations.
        """
        smoothed_workers = []

        for worker in detected_workers:
            wid = worker["worker_id"]
            if wid not in self.worker_history:
                self.worker_history[wid] = deque(maxlen=self.history_len)
                self.worker_stats[wid] = {
                    "total_frames": 0,
                    "safe_frames": 0,
                    "violation_frames": 0
                }
                self.worker_ppe_memory[wid] = {
                    "frames_without_hardhat": 0,
                    "frames_without_vest": 0,
                    "ever_had_hardhat": False,
                    "ever_had_vest": False
                }

            mem = self.worker_ppe_memory[wid]

            # Update hardhat memory
            if worker["has_hardhat"]:
                mem["frames_without_hardhat"] = 0
                mem["ever_had_hardhat"] = True
                effective_has_hardhat = True
            else:
                mem["frames_without_hardhat"] += 1
                # If worker previously had a hardhat and it has only been missing for <= grace_frames, maintain state
                if mem["ever_had_hardhat"] and mem["frames_without_hardhat"] <= self.grace_frames:
                    effective_has_hardhat = True
                else:
                    effective_has_hardhat = False

            # Update vest memory
            if worker["has_vest"]:
                mem["frames_without_vest"] = 0
                mem["ever_had_vest"] = True
                effective_has_vest = True
            else:
                mem["frames_without_vest"] += 1
                if mem["ever_had_vest"] and mem["frames_without_vest"] <= self.grace_frames:
                    effective_has_vest = True
                else:
                    effective_has_vest = False

            # Recalculate compliance with PPE memory
            violations = []
            if not effective_has_hardhat:
                violations.append("NO HELMET")
            if not effective_has_vest:
                violations.append("NO SAFETY VEST")

            is_safe = len(violations) == 0

            # Record stats
            self.worker_stats[wid]["total_frames"] += 1
            if is_safe:
                self.worker_stats[wid]["safe_frames"] += 1
                severity = "OK"
                status = "SAFE"
            elif len(violations) >= 2:
                self.worker_stats[wid]["violation_frames"] += 1
                severity = "CRITICAL"
                status = "CRITICAL VIOLATION"
            else:
                self.worker_stats[wid]["violation_frames"] += 1
                severity = "WARNING"
                status = f"WARNING: {violations[0]}"

            worker_copy = dict(worker)
            worker_copy["has_hardhat"] = effective_has_hardhat
            worker_copy["has_vest"] = effective_has_vest
            worker_copy["missing_hardhat"] = not effective_has_hardhat
            worker_copy["missing_vest"] = not effective_has_vest
            worker_copy["is_safe"] = is_safe
            worker_copy["violations"] = violations
            worker_copy["status"] = status
            worker_copy["severity"] = severity
            worker_copy["smoothed_is_safe"] = is_safe
            worker_copy["smoothed_violations"] = violations
            worker_copy["smoothed_status"] = status
            worker_copy["smoothed_severity"] = severity

            smoothed_workers.append(worker_copy)

        return smoothed_workers
