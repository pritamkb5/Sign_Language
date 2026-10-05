"""
data_logger.py — Module 3: Landmark Logging Tool (Phase 1 Upgraded)
==================================================================
Responsibility: Capture hand landmark coordinates and append them to a CSV file
for static training, or save sequences to .npy files for dynamic training.

CSV Schema (129 columns):
  x0,y0,z0,...,x41,y41,z41,label,timestamp,session_tag
  - Columns 0–125: 42 landmarks * 3 coordinates (2 hands * 21 * 3 = 126 features)
  - Column 126: Target label (string)
  - Column 127: UNIX timestamp (float)
  - Column 128: Session tag / metadata (string)

Dynamic Sequence Storage:
  - Saved as: data/dynamic/[label]_[session_timestamp]_[index].npy
  - Data shape: (num_frames, 126)
"""

import argparse
import csv
import os
import time
from pathlib import Path
import cv2
import numpy as np

# Import our capture and preprocessing helpers
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.capture import open_camera, compute_fps, draw_fps
from src.landmarks import get_hand_landmarks, draw_landmarks_on_frame, process_frame_for_landmarks
from src.preprocessing import prepare_input_vector
from src.labels import load_class_labels, load_dynamic_labels

# Pinned defaults
DEFAULT_CSV_PATH = str(Path(__file__).resolve().parent.parent / "data" / "landmarks.csv")
DEFAULT_LABEL = "A"
NUM_LANDMARKS = 21
TOTAL_COORDS = 126  # 2 hands * 21 points * 3 coordinates

# Full ASL static alphabet labels and dynamic word vocabulary
LABEL_CYCLE = load_class_labels()
DYNAMIC_WORDS = load_dynamic_labels()



def _build_csv_header() -> list[str]:
    """Return the header row for the landmark CSV."""
    cols = []
    for i in range(42):
        cols += [f"x{i}", f"y{i}", f"z{i}"]
    cols.append("label")
    cols.append("timestamp")
    cols.append("session_tag")
    return cols


def _ensure_csv_exists(csv_path: str) -> None:
    """Create the CSV file with a header row if it does not already exist."""
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(_build_csv_header())
        print(f"[INFO] Created new CSV file: {path.resolve()}")


def count_label_samples(csv_path: str, label: str) -> int:
    """Count existing rows in the CSV that carry the given label."""
    path = Path(csv_path)
    if not path.exists():
        return 0
    count = 0
    try:
        with open(path, newline="", errors="ignore") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("label") == label:
                    count += 1
    except Exception:
        pass
    return count


def save_landmark_sample(
    flat_features: list[float],
    label: str,
    csv_path: str = DEFAULT_CSV_PATH,
    session_tag: str = "default_session"
) -> int:
    """
    Appends a 126-dimensional feature vector, label, timestamp, and session tag to the CSV.
    """
    if len(flat_features) < TOTAL_COORDS:
        flat_features = flat_features + [0.0] * (TOTAL_COORDS - len(flat_features))
    elif len(flat_features) > TOTAL_COORDS:
        flat_features = flat_features[:TOTAL_COORDS]

    try:
        _ensure_csv_exists(csv_path)
        
        # Round features for file size compression
        rounded_features = [round(v, 6) for v in flat_features]
        row = rounded_features + [label, round(time.time(), 4), session_tag]

        # PHASE 2: add additional verification constraints on samples before writing
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(row)

        total = count_label_samples(csv_path, label)
        return total
    except Exception as e:
        print(f"[ERROR] Failed to save sample: {e}")
        return 0


def save_dynamic_sequence(
    sequence_data: list[list[float]],
    label: str,
    output_dir: str = "data/dynamic"
) -> str:
    """
    Saves a captured sequence of frames as a numpy binary (.npy) file.
    """
    if not sequence_data:
        return ""
    
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    
    timestamp = int(time.time())
    file_name = f"{label}_{timestamp}_{len(sequence_data)}f.npy"
    file_path = path / file_name
    
    # Save as numpy array
    arr = np.array(sequence_data, dtype=np.float32)
    np.save(file_path, arr)
    
    print(f"[SAVED DYNAMIC] Word '{label}' sequence saved to: {file_path.resolve()} (Shape: {arr.shape})")
    return str(file_path)


class SaveFlash:
    """Displays a brief banner on screen for visual feedback."""
    DURATION = 0.5

    def __init__(self):
        self._trigger_time: float | None = None
        self._label = ""
        self._color = (0, 180, 0)  # default green

    def trigger(self, label: str, color=(0, 180, 0)):
        self._trigger_time = time.time()
        self._label = label
        self._color = color

    def draw(self, frame) -> None:
        if self._trigger_time is None:
            return
        if time.time() - self._trigger_time > self.DURATION:
            self._trigger_time = None
            return

        text = self._label
        h, w = frame.shape[:2]
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1.0
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = (w - tw) // 2
        y = 80

        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - th - 10), (x + tw + 10, y + 10), self._color, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)


# ── Interactive Collection Shell loops ──────────────────────────────────────────

def run_guided_session(csv_path: str, session_tag: str, samples_per_class: int = 100) -> None:
    """Runs a guided step-by-step recording session for the static alphabet/poses."""
    print("\n" + "=" * 60)
    print(" GUIDED DATA COLLECTION SESSION")
    print(f" Output target: {Path(csv_path).resolve()}")
    print(f" Session Tag: {session_tag}")
    print(f" Capturing {samples_per_class} samples per sign pose.")
    print("=" * 60 + "\n")

    try:
        cap = open_camera(0)
    except RuntimeError as e:
        print(e)
        return

    flash = SaveFlash()
    prev_time = time.time()

    for label in LABEL_CYCLE:
        print(f"\n[NEXT SIGN] Prepare pose for: '{label}'")
        print("  -> Focus the camera window.")
        print("  -> Press [SPACE] to start capturing samples, or [s] to skip, or [q] to quit session.")
        
        start_capture = False
        skip_label = False
        
        while not start_capture and not skip_label:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            # Draw landmarks
            results, landmarks_list = process_frame_for_landmarks(frame)
            draw_landmarks_on_frame(frame, results)

            # Draw active instruction
            h, w = frame.shape[:2]
            cv2.putText(
                frame, f"Next: Show '{label}' and press [SPACE]",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2, cv2.LINE_AA
            )
            cv2.putText(
                frame, "[SPACE] Start | [s] Skip Sign | [q] Exit",
                (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
            )
            
            flash.draw(frame)
            cv2.imshow("Guided Data Collection Session", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):
                start_capture = True
            elif key == ord("s"):
                skip_label = True
                print(f"[INFO] Skipped sign: '{label}'")
            elif key == ord("q"):
                print("[INFO] Exiting guided session.")
                cap.release()
                cv2.destroyAllWindows()
                return

        if skip_label:
            continue

        # Start capturing samples at 10 FPS
        print(f"[CAPTURING] Collecting {samples_per_class} samples for label '{label}'...")
        samples_collected = 0
        last_capture_time = time.time()
        
        while samples_collected < samples_per_class:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)

            results, landmarks_list = process_frame_for_landmarks(frame)
            draw_landmarks_on_frame(frame, results)

            current_time = time.time()
            # Capture frame at ~10 FPS interval (every 0.1 seconds)
            if current_time - last_capture_time >= 0.10:
                last_capture_time = current_time
                if landmarks_list:
                    flat_vector = prepare_input_vector(landmarks_list)
                    save_landmark_sample(flat_vector, label, csv_path, session_tag)
                    samples_collected += 1
                    flash.trigger(f"CAPTURING '{label}' {samples_collected}/{samples_per_class}", color=(0, 120, 200))
                else:
                    # Warn in window if hand is missing during capture
                    flash.trigger("WARNING: NO HAND DETECTED", color=(0, 0, 200))

            # Progress overlay
            h, w = frame.shape[:2]
            cv2.rectangle(frame, (10, 15), (320, 45), (0, 0, 0), -1)
            cv2.putText(
                frame, f"Capturing '{label}': {samples_collected}/{samples_per_class}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA
            )

            flash.draw(frame)
            cv2.imshow("Guided Data Collection Session", frame)
            
            # Allow cancellation during capture
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Interrupted guided session.")
                cap.release()
                cv2.destroyAllWindows()
                return

        print(f"[SUCCESS] Collected '{label}'. Total in dataset: {count_label_samples(csv_path, label)}")
        flash.trigger(f"FINISHED '{label}'!", color=(0, 180, 0))
        time.sleep(0.5)

    cap.release()
    cv2.destroyAllWindows()
    print("\n[SUCCESS] Guided session completed successfully!")


def run_manual_session(csv_path: str, session_tag: str) -> None:
    """Runs a manual capture loop supporting Single, Batch, and Dynamic Sequence recording."""
    try:
        cap = open_camera(0)
    except RuntimeError as e:
        print(e)
        return

    flash = SaveFlash()
    prev_time = time.time()

    # Align active collection label defaults
    label_idx = 0
    active_label = LABEL_CYCLE[label_idx]
    
    # Dynamic sequence state
    dynamic_idx = 0
    active_word = DYNAMIC_WORDS[dynamic_idx]
    is_recording_dynamic = False
    sequence_buffer = []

    # Batch state
    batch_mode = False
    batch_count = 0
    batch_target = 30  # capture 30 frames at 10 FPS on batch trigger
    last_batch_time = 0.0

    print(f"\n[INFO] Manual Data Logger active.")
    print("[INFO] Controls: ")
    print("  -> [s] Save single static sample")
    print("  -> [b] Trigger batch capture (30 samples @ 10 FPS)")
    print("  -> [d] Toggle dynamic sequence capture (start/stop)")
    print("  -> [l] Cycle static labels  |  [w] Cycle dynamic word labels")
    print("  -> [q] Quit Window\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.flip(frame, 1)

        results, landmarks_list = process_frame_for_landmarks(frame)
        draw_landmarks_on_frame(frame, results)

        # Handle batch capture execution
        current_time = time.time()
        if batch_mode:
            if current_time - last_batch_time >= 0.10:
                last_batch_time = current_time
                if landmarks_list:
                    flat_vector = prepare_input_vector(landmarks_list)
                    save_landmark_sample(flat_vector, active_label, csv_path, session_tag)
                    batch_count += 1
                    flash.trigger(f"BATCH: {batch_count}/{batch_target}", color=(0, 120, 200))
                else:
                    flash.trigger("BATCH: NO HAND DETECTED", color=(0, 0, 200))
                
                if batch_count >= batch_target:
                    batch_mode = False
                    print(f"[BATCH SUCCESS] Saved {batch_target} samples for '{active_label}'")

        # Handle active dynamic sequence recording
        if is_recording_dynamic and current_time - last_batch_time >= 0.10:
            last_batch_time = current_time
            # Record regardless of hand presence, padding will resolve empty states
            flat_vector = prepare_input_vector(landmarks_list)
            sequence_buffer.append(flat_vector)

        # Text overlays
        h, w = frame.shape[:2]
        fps, prev_time = compute_fps(prev_time)
        draw_fps(frame, fps)

        # Rendering status overlay at top-left
        if is_recording_dynamic:
            cv2.rectangle(frame, (10, 15), (320, 45), (0, 0, 180), -1)
            cv2.putText(
                frame, f"REC DYNAMIC: {active_word} ({len(sequence_buffer)}f)",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
        elif batch_mode:
            cv2.rectangle(frame, (10, 15), (320, 45), (200, 100, 0), -1)
            cv2.putText(
                frame, f"BATCH ACTIVE: {batch_count}/{batch_target}",
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
        else:
            # Show active target details
            cv2.putText(
                frame, f"Static Label: {active_label}  |  Word Label: {active_word}",
                (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1, cv2.LINE_AA
            )
            cv2.putText(
                frame, "[s] Save  [b] Batch  [d] Rec Dynamic  [l]/[w] Cycle  [q] Quit",
                (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA
            )

        flash.draw(frame)
        cv2.imshow("Manual Data Logger", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            if is_recording_dynamic:
                print("[WARNING] Exited while recording. Sequence discarded.")
            break

        elif key == ord("s"):
            # Single sample save
            if not batch_mode and not is_recording_dynamic:
                if landmarks_list:
                    flat_vector = prepare_input_vector(landmarks_list)
                    tot = save_landmark_sample(flat_vector, active_label, csv_path, session_tag)
                    flash.trigger(f"SAVED: '{active_label}' ({tot})")
                else:
                    print("[WARNING] No hand detected — sample not logged.")
                    flash.trigger("ERR: NO HAND", color=(0, 0, 180))

        elif key == ord("b"):
            # Trigger batch capture
            if not batch_mode and not is_recording_dynamic:
                batch_mode = True
                batch_count = 0
                last_batch_time = time.time()
                print(f"[BATCH] Starting capture of {batch_target} frames for '{active_label}'...")

        elif key == ord("d"):
            # Toggle dynamic sequence record
            if not batch_mode:
                if not is_recording_dynamic:
                    is_recording_dynamic = True
                    sequence_buffer = []
                    last_batch_time = time.time()
                    print(f"[DYNAMIC RECORD] Starting sequence capture for '{active_word}'...")
                    flash.trigger("RECORDING STARTED...", color=(0, 0, 180))
                else:
                    is_recording_dynamic = False
                    save_dynamic_sequence(sequence_buffer, active_word)
                    flash.trigger("RECORDING SAVED!", color=(0, 180, 0))

        elif key == ord("l"):
            # Cycle static label
            if not batch_mode and not is_recording_dynamic:
                label_idx = (label_idx + 1) % len(LABEL_CYCLE)
                active_label = LABEL_CYCLE[label_idx]
                print(f"[INFO] Target static label cycle -> '{active_label}'")

        elif key == ord("w"):
            # Cycle dynamic word label
            if not batch_mode and not is_recording_dynamic:
                dynamic_idx = (dynamic_idx + 1) % len(DYNAMIC_WORDS)
                active_word = DYNAMIC_WORDS[dynamic_idx]
                print(f"[INFO] Target dynamic word cycle -> '{active_word}'")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Data Collection Tool")
    parser.add_argument("--mode", type=str, default="manual", choices=["manual", "guided"], help="Collection mode")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV_PATH, help="Path to output landmarks CSV")
    parser.add_argument("--tag", type=str, default="session_lighting_bright", help="Session metadata tag")
    parser.add_argument("--samples", type=int, default=100, help="Samples per class in guided mode")
    
    args = parser.parse_args()
    
    if args.mode == "guided":
        run_guided_session(args.csv, args.tag, args.samples)
    else:
        run_manual_session(args.csv, args.tag)
