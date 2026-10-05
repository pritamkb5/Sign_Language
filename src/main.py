from src.sentence_builder import SentenceBuilder
"""
main.py — Phase 2 Real-Time Dual-Mode Sign Language Translator
===============================================================
Connects all pipeline components in a real-time webcam feed:
  webcam -> frame -> landmarks -> normalized features -> rolling buffer -> 
  static (Model v2) / dynamic (Model v1) classifier -> temporal smoothing -> visual UI

Controls:
  [m] Toggle Mode (Static Alphabet vs Dynamic Word Sequence)
  [s] Save Static Sample to CSV
  [l] Cycle Target Static Label
  [w] Cycle Target Dynamic Word Label
  [q] Quit Application
"""

import argparse
import os
import time
from pathlib import Path
import cv2
import numpy as np

# Set environmental flags
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Import pipeline modules
from src.capture import open_camera, compute_fps, draw_fps
from src.landmarks import process_frame_for_landmarks, draw_landmarks_on_frame
from src.preprocessing import prepare_input_vector
from src.data_logger import save_landmark_sample, SaveFlash, DEFAULT_CSV_PATH
from src.inference.stability import PredictionStabilizer
from src.inference.sequence_buffer import SequenceBuffer
from src.models.static_model import StaticClassifierV2
from src.models.dynamic_model import DynamicClassifier
from src.dummy_classifier import predict as dummy_predict
from src.labels import load_class_labels

# Paths
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
STATIC_MODEL_PATH = str(_PROJECT_ROOT / "models" / "mlp_v2.pt")
STATIC_MODEL_FALLBACK = str(_PROJECT_ROOT / "models" / "mlp_v1.pt")
STATIC_LABELS_PATH = str(_PROJECT_ROOT / "models" / "class_labels.txt")

DYNAMIC_MODEL_PATH = str(_PROJECT_ROOT / "models" / "dynamic_v1.pt")
DYNAMIC_LABELS_PATH = str(_PROJECT_ROOT / "models" / "dynamic_labels.txt")


def load_dynamic_labels() -> list[str]:
    """Loads dynamic word labels or provides default 15-word vocabulary."""
    if os.path.exists(DYNAMIC_LABELS_PATH):
        with open(DYNAMIC_LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f.read().splitlines() if line.strip()]
            if labels:
                return labels
    return ["hello", "thanks", "yes", "no", "please", "help", "sorry", "name", "more", "stop", "love", "want", "eat", "drink", "friend"]


def draw_prediction_badge(frame, mode: str, label: str, confidence: float) -> None:
    """Render predicted label, confidence, and active model mode badge."""
    if not label:
        return

    text = f"{label} ({confidence * 100:.0f}%)" if confidence > 0 else f"{label}"

    font = cv2.FONT_HERSHEY_DUPLEX
    scale = 1.1
    thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, scale, thickness)

    h, w = frame.shape[:2]
    padding = 12
    x = w - text_w - padding - 15
    y = 55

    # Pill background
    bg_color = (30, 30, 30)
    border_color = (0, 200, 255) if mode == "STATIC" else (220, 100, 255)
    text_color = (100, 240, 255) if mode == "STATIC" else (240, 150, 255)

    cv2.rectangle(frame, (x - padding, y - text_h - padding), (x + text_w + padding, y + baseline + padding), bg_color, -1)
    cv2.rectangle(frame, (x - padding, y - text_h - padding), (x + text_w + padding, y + baseline + padding), border_color, 2)
    cv2.putText(frame, text, (x, y), font, scale, text_color, thickness, cv2.LINE_AA)


def draw_sequence_progress(frame, fill_ratio: float) -> None:
    """Draw rolling buffer fill progress bar when operating in Dynamic Mode."""
    h, w = frame.shape[:2]
    bar_w = 200
    bar_h = 14
    x = 15
    y = 65

    # Label
    cv2.putText(frame, "Seq Buffer:", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
    
    # Background bar
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (40, 40, 40), -1)
    
    # Filled bar
    filled_w = int(bar_w * fill_ratio)
    bar_color = (0, 220, 100) if fill_ratio >= 1.0 else (0, 165, 255)
    cv2.rectangle(frame, (x, y), (x + filled_w, y + bar_h), bar_color, -1)
    cv2.rectangle(frame, (x, y), (x + bar_w, y + bar_h), (120, 120, 120), 1)


def draw_status_bar(frame, mode: str, active_static: str, active_word: str) -> None:
    """Draw telemetry and controls status bar at the bottom."""
    h, w = frame.shape[:2]
    mode_str = f"MODE: [{mode}]"
    target_str = f"Static Target: '{active_static}'  |  Word Target: '{active_word}'"
    control_str = "[m] Toggle Mode  |  [s] Save  |  [l]/[w] Cycle Labels  |  [q] Quit"

    # Top overlay header
    cv2.rectangle(frame, (0, 0), (w, 35), (20, 20, 20), -1)
    cv2.putText(frame, mode_str, (15, 24), cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 255, 200) if mode == "STATIC" else (255, 120, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, target_str, (200, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)

    # Bottom control bar
    cv2.rectangle(frame, (0, h - 35), (w, h), (15, 15, 15), -1)
    cv2.putText(frame, control_str, (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)


def run_pipeline(csv_path: str = DEFAULT_CSV_PATH, camera_index: int = 0) -> None:
    """Integrated dual-mode execution loop."""
    print("\n" + "=" * 60)
    print(" SIGN LANGUAGE TRANSLATOR — PHASE 2 DUAL-MODE APP")
    print(" Press [m] to toggle between Static Alphabet and Dynamic Word modes.")
    print("=" * 60 + "\n")

    # 1. Initialize Classifiers
    # Static Model v2 / fallback
    static_model_file = STATIC_MODEL_PATH if os.path.exists(STATIC_MODEL_PATH) else STATIC_MODEL_FALLBACK
    static_labels = load_class_labels(STATIC_LABELS_PATH)
    static_classifier = StaticClassifierV2(model_path=static_model_file, class_labels=static_labels)
    print(f"[INFO] Static Classifier initialized with model: {static_model_file}")

    # Dynamic GRU Model v1
    dynamic_labels = load_dynamic_labels()
    dynamic_classifier = DynamicClassifier(model_path=DYNAMIC_MODEL_PATH, class_labels=dynamic_labels)
    print(f"[INFO] Dynamic GRU Classifier initialized with model: {DYNAMIC_MODEL_PATH}")

    # 2. Sequence Buffer & Prediction Stabilizer
    sequence_buffer = SequenceBuffer(sequence_length=30, feature_dim=126)
    stabilizer = PredictionStabilizer(k_threshold=4, confidence_threshold=0.45,
                                      dynamic_k_threshold=2, dynamic_confidence_threshold=0.40)
    stabilizer.set_mode("STATIC")

    # Mode state: 'STATIC' or 'DYNAMIC'
    active_mode = "STATIC"

    # Active collection labels
    static_idx = 0
    active_static_label = static_labels[static_idx]
    
    dynamic_idx = 0
    active_dynamic_word = dynamic_labels[dynamic_idx]

    # Open Camera
    try:
        cap = open_camera(camera_index)
    except RuntimeError as e:
        print(e)
        return

    flash = SaveFlash()
    sentence_builder = SentenceBuilder(idle_timeout=2.0)
    current_translation = ""

    def update_ui_with_translation(natural_english_sentence):
        nonlocal current_translation
        current_translation = natural_english_sentence
        print(f"\n[GEMINI TRANSLATION]: {natural_english_sentence}\n")
    prev_time = time.time()
    # Frame budget: if processing exceeds this, skip next frame's heavy pipeline
    _FRAME_BUDGET_MS = 40.0  # ~25fps threshold
    _skip_next = False

    print("[INFO] Live pipeline active. Focus camera window to control.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        frame_start = time.time()

        # Check if frame is pitch-black (e.g. physical camera lens cover closed)
        if np.mean(frame) < 2.0:
            h, w = frame.shape[:2]
            cv2.putText(frame, "[CAMERA FEED BLANK: Check physical camera cover or privacy slider]",
                        (20, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, cv2.LINE_AA)

        # Skip heavy processing if the previous frame overran the budget,
        # so the display can catch up to real-time
        if _skip_next:
            _skip_next = False
            fps, prev_time = compute_fps(prev_time)
            draw_fps(frame, fps)
            draw_status_bar(frame, active_mode, active_static_label, active_dynamic_word)
            flash.draw(frame)
            cv2.imshow("Sign Language Translator — Dual Mode", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        # Process hand landmarks
        results, landmarks_list = process_frame_for_landmarks(frame)
        draw_landmarks_on_frame(frame, results)


        # Prepare normalized 126-dim input vector
        flat_input = prepare_input_vector(landmarks_list) if landmarks_list else [0.0] * 126

        predicted_label = ""
        confidence = 0.0

        if active_mode == "STATIC":
            # --- Track A: Static Gesture Model v2 ---
            if landmarks_list:
                predicted_label, confidence = static_classifier.predict_with_confidence(flat_input)
        else:
            # --- Track B: Dynamic Sequence GRU Model v1 ---
            # Feed every frame to the gesture-segmented buffer
            sequence_buffer.add_frame(flat_input)
            
            # Only classify when a complete gesture segment is ready
            if sequence_buffer.has_gesture_ready():
                seq_tensor = sequence_buffer.get_sequence()
                if seq_tensor is not None:
                    predicted_label, confidence = dynamic_classifier.predict_with_confidence(seq_tensor)

        # Track C1: Temporal Smoothing over predictions
        stabilized_label = stabilizer.process_prediction(predicted_label, confidence)

        if stabilized_label:
            sentence_builder.add_char(stabilized_label)
        sentence_builder.check_idle_and_translate(on_translation_done=update_ui_with_translation)
        
        # Get live typing text
        live_raw_text = sentence_builder.get_display_text()
        # Render overlays
        fps, prev_time = compute_fps(prev_time)
        draw_fps(frame, fps)

        if active_mode == "DYNAMIC":
            draw_sequence_progress(frame, sequence_buffer.fill_ratio())

        draw_prediction_badge(frame, active_mode, stabilized_label, confidence)
        draw_status_bar(frame, active_mode, active_static_label, active_dynamic_word)
        flash.draw(frame)
        cv2.putText(frame, f"Typing: {live_raw_text}", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(frame, f"AI: {current_translation}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 100), 2, cv2.LINE_AA)

        cv2.imshow("Sign Language Translator — Dual Mode", frame)

        # If this frame took too long, skip heavy processing on the next frame
        frame_elapsed_ms = (time.time() - frame_start) * 1000.0
        if frame_elapsed_ms > _FRAME_BUDGET_MS:
            _skip_next = True

        # Keypress controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quit key received. Exiting...")
            break

        elif key == ord("m"):
            # Toggle Mode
            active_mode = "DYNAMIC" if active_mode == "STATIC" else "STATIC"
            stabilizer.reset()
            stabilizer.set_mode(active_mode)
            sequence_buffer.clear()
            flash.trigger(f"MODE: {active_mode}", color=(200, 100, 255) if active_mode == "DYNAMIC" else (0, 200, 255))
            print(f"[MODE TOGGLE] Switched pipeline to: {active_mode} MODE")

        elif key == ord("s"):
            # Save static sample
            if landmarks_list:
                save_landmark_sample(flat_input, active_static_label, csv_path, session_tag="manual_phase2")
                flash.trigger(f"SAVED: '{active_static_label}'")
            else:
                flash.trigger("ERR: NO HAND", color=(0, 0, 200))

        elif key == ord("l"):
            # Cycle static label
            static_idx = (static_idx + 1) % len(static_labels)
            active_static_label = static_labels[static_idx]
            print(f"[INFO] Static collection label -> '{active_static_label}'")

        elif key == ord("w"):
            # Cycle dynamic word label
            dynamic_idx = (dynamic_idx + 1) % len(dynamic_labels)
            active_dynamic_word = dynamic_labels[dynamic_idx]
            print(f"[INFO] Dynamic word collection label -> '{active_dynamic_word}'")

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Pipeline shut down cleanly.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sign Language Translator — Phase 2 Integrated App")
    parser.add_argument("--output", "-o", default=DEFAULT_CSV_PATH, help="CSV landmark output path")
    parser.add_argument("--camera-index", "-c", type=int, default=0, help="Webcam camera index")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(csv_path=args.output, camera_index=args.camera_index)

