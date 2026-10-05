import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sentence_builder import SentenceBuilder
import cv2
import threading
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, 
                             QComboBox, QTextEdit, QCheckBox, QMessageBox)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont

# Pipeline imports
from src.capture import open_camera, compute_fps
from src.landmarks import process_frame_for_landmarks, draw_landmarks_on_frame
from src.preprocessing import prepare_input_vector, prepare_raw_vector
from src.inference.stability import PredictionStabilizer
from src.inference.sequence_buffer import SequenceBuffer
from src.models.static_model import StaticClassifierV2
from src.models.dynamic_model import DynamicClassifier
from src.labels import load_class_labels

# Phase 3 imports
from src.speech_output import SpeechEngine
from src.caption_overlay import draw_caption_overlay

# Setup paths (similar to main.py)
from pathlib import Path
from src.resource_path import get_project_root
_PROJECT_ROOT = get_project_root()
STATIC_MODEL_PATH = str(_PROJECT_ROOT / "models" / "mlp_v2.pt")
STATIC_MODEL_FALLBACK = str(_PROJECT_ROOT / "models" / "mlp_v1.pt")
STATIC_LABELS_PATH = str(_PROJECT_ROOT / "models" / "class_labels.txt")
DYNAMIC_MODEL_PATH = str(_PROJECT_ROOT / "models" / "dynamic_v1.pt")
DYNAMIC_LABELS_PATH = str(_PROJECT_ROOT / "models" / "dynamic_labels.txt")

def load_dynamic_labels() -> list[str]:
    if os.path.exists(DYNAMIC_LABELS_PATH):
        with open(DYNAMIC_LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f.read().splitlines() if line.strip()]
            if labels: return labels
    return ["hello", "thanks", "yes", "no", "please", "help", "sorry", "name", "more", "stop", "love", "want", "eat", "drink", "friend"]


class VideoThread(QThread):
    """
    Camera capture thread that stores only the latest frame.
    The main thread pulls frames via grab_latest() on a QTimer,
    preventing Qt signal queue buildup that causes video feed delay.
    """
    error_signal = pyqtSignal(str)

    def __init__(self, camera_index=0):
        super().__init__()
        self.camera_index = camera_index
        self._run_flag = True
        self.cap = None
        self._lock = threading.Lock()
        self._latest_frame = None
        self._frame_seq = 0        # incremented on every new frame
        self._last_grabbed_seq = 0  # tracks which frame was last pulled

    def run(self):
        try:
            self.cap = open_camera(self.camera_index)
            while self._run_flag:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    with self._lock:
                        self._latest_frame = frame
                        self._frame_seq += 1
                else:
                    self.error_signal.emit("Camera disconnected or frame dropped.")
                    break
        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            if self.cap:
                self.cap.release()

    def grab_latest(self) -> np.ndarray | None:
        """
        Returns the most recent camera frame if a new one is available
        since the last call. Returns None if no new frame is ready.
        This ensures the UI always displays the freshest frame and
        never processes stale queued frames.
        """
        with self._lock:
            if self._frame_seq > self._last_grabbed_seq and self._latest_frame is not None:
                self._last_grabbed_seq = self._frame_seq
                return self._latest_frame.copy()
            return None

    def stop(self):
        self._run_flag = False
        self.wait()


class DynamicWorker(QThread):
    result_ready = pyqtSignal(str, float)

    def __init__(self, classifier):
        super().__init__()
        self.classifier = classifier
        self.seq_tensor = None

    def run(self):
        if self.seq_tensor is not None:
            try:
                label, conf = self.classifier.predict_with_confidence(self.seq_tensor)
                self.result_ready.emit(label, conf)
            except Exception:
                import traceback
                traceback.print_exc()


class TranslatorApp(QMainWindow):
    translation_updated = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sign Language Translator - Alpha")
        self.setGeometry(100, 100, 1000, 700)
        
        # Pipeline State
        self.active_mode = "STATIC"
        self.is_running = False
        self._process_timer = None
        
        self.translation_updated.connect(self.update_sentence_ui)
        
        # Initialize Models (wrapped in try-except for robustness)
        try:
            static_model_file = STATIC_MODEL_PATH if os.path.exists(STATIC_MODEL_PATH) else STATIC_MODEL_FALLBACK
            self.static_labels = load_class_labels(STATIC_LABELS_PATH)
            self.static_classifier = StaticClassifierV2(model_path=static_model_file, class_labels=self.static_labels)
            
            self.dynamic_labels = load_dynamic_labels()
            self.dynamic_classifier = DynamicClassifier(model_path=DYNAMIC_MODEL_PATH, class_labels=self.dynamic_labels)
            
            self.sequence_buffer = SequenceBuffer(sequence_length=16, feature_dim=126)
            self.static_stabilizer = PredictionStabilizer(k_threshold=4, confidence_threshold=0.40)
            self.static_stabilizer.set_mode("STATIC")
            
            self.dynamic_worker = DynamicWorker(self.dynamic_classifier)
            self.dynamic_worker.result_ready.connect(self.on_dynamic_prediction_ready)
            self.last_dynamic_pred = ""
        except Exception as e:
            QMessageBox.critical(self, "Initialization Error", f"Failed to load models: {e}")
            sys.exit(1)
            
        # Phase 3 Components
        self.sentence_builder = SentenceBuilder(idle_timeout=2.0)
        self.sentence_builder.on_update_callback = self.translation_updated.emit
        self.speech_engine = SpeechEngine(auto_speak=False)
        self.captions_enabled = True
        
        self.no_hand_frames = 0
        self.setup_ui()
        
    def setup_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)
        
        # Left Panel: Video & Basic Indicators
        left_layout = QVBoxLayout()
        
        self.video_label = QLabel("Camera Feed (Stopped)")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        left_layout.addWidget(self.video_label)
        
        # Live Prediction Indicator
        self.pred_label = QLabel("Prediction: --")
        self.pred_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.pred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.pred_label)
        
        self.warning_label = QLabel("")
        self.warning_label.setStyleSheet("color: orange;")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        left_layout.addWidget(self.warning_label)
        
        layout.addLayout(left_layout, stretch=2)
        
        # Right Panel: Controls & Sentence Buffer
        right_layout = QVBoxLayout()
        
        # Sentence Area
        right_layout.addWidget(QLabel("Raw Sign Input:"))
        self.sentence_text = QTextEdit()
        self.sentence_text.setReadOnly(True)
        self.sentence_text.setFont(QFont("Arial", 14))
        self.sentence_text.setMaximumHeight(60)
        right_layout.addWidget(self.sentence_text)

        right_layout.addWidget(QLabel("Translation:"))
        self.translation_text = QTextEdit()
        self.translation_text.setReadOnly(True)
        self.translation_text.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        self.translation_text.setStyleSheet("color: #2e8b57;")
        self.translation_text.setMaximumHeight(80)
        right_layout.addWidget(self.translation_text)
        
        # Controls for Sentence
        btn_layout = QHBoxLayout()
        self.btn_space = QPushButton("Space")
        self.btn_delete = QPushButton("Delete")
        self.btn_clear = QPushButton("Clear")
        self.btn_translate = QPushButton("Translate")
        self.btn_speak = QPushButton("Speak Sentence")
        
        self.btn_space.clicked.connect(self.on_space_clicked)
        self.btn_delete.clicked.connect(self.on_delete_clicked)
        self.btn_clear.clicked.connect(self.on_clear_clicked)
        self.btn_translate.clicked.connect(self.on_translate_clicked)
        self.btn_speak.clicked.connect(self.on_speak_clicked)
        
        btn_layout.addWidget(self.btn_space)
        btn_layout.addWidget(self.btn_delete)
        btn_layout.addWidget(self.btn_clear)
        btn_layout.addWidget(self.btn_translate)
        btn_layout.addWidget(self.btn_speak)
        right_layout.addLayout(btn_layout)
        
        # Settings Panel
        right_layout.addWidget(QLabel("Settings:"))
        
        # Mode Toggle Buttons
        mode_layout = QHBoxLayout()
        self.btn_static_mode = QPushButton("Static Mode")
        self.btn_dynamic_mode = QPushButton("Dynamic Mode")
        self.btn_static_mode.setCheckable(True)
        self.btn_dynamic_mode.setCheckable(True)
        self.btn_static_mode.clicked.connect(lambda: self.set_mode("STATIC"))
        self.btn_dynamic_mode.clicked.connect(lambda: self.set_mode("DYNAMIC"))
        mode_layout.addWidget(self.btn_static_mode)
        mode_layout.addWidget(self.btn_dynamic_mode)
        right_layout.addLayout(mode_layout)

        # Dynamic Vocabulary List
        self.vocab_label = QLabel("Dynamic Vocab: " + ", ".join(self.dynamic_labels))
        self.vocab_label.setWordWrap(True)
        self.vocab_label.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        self.vocab_label.setVisible(False)
        right_layout.addWidget(self.vocab_label)

        # Camera Selector
        cam_sel_layout = QHBoxLayout()
        cam_sel_layout.addWidget(QLabel("Camera Device:"))
        self.cam_combo = QComboBox()
        self.cam_combo.addItems([
            "Auto-Detect Active Camera",
            "Camera 1 (Logitech Brio 100)",
            "Camera 0 (Integrated Camera)",
            "Camera 2",
            "Synthetic Camera (Mock)"
        ])
        cam_sel_layout.addWidget(self.cam_combo)
        right_layout.addLayout(cam_sel_layout)
        
        self.auto_speak_check = QCheckBox("Auto-Speak Confirmed Words")
        self.auto_speak_check.stateChanged.connect(self.on_auto_speak_changed)
        right_layout.addWidget(self.auto_speak_check)

        self.captions_check = QCheckBox("Show Live Captions On Video")
        self.captions_check.setChecked(True)
        self.captions_check.stateChanged.connect(self.on_captions_toggle_changed)
        right_layout.addWidget(self.captions_check)
        
        # Sliders
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(30, 95)
        self.conf_slider.setValue(40)
        self.conf_slider.valueChanged.connect(self.on_settings_changed)
        conf_layout.addWidget(self.conf_slider)
        right_layout.addLayout(conf_layout)
        
        hold_layout = QHBoxLayout()
        hold_layout.addWidget(QLabel("Hold Duration (Frames):"))
        self.hold_slider = QSlider(Qt.Orientation.Horizontal)
        self.hold_slider.setRange(5, 45)
        self.hold_slider.setValue(15)
        self.hold_slider.valueChanged.connect(self.on_settings_changed)
        hold_layout.addWidget(self.hold_slider)
        right_layout.addLayout(hold_layout)
        
        # Camera Control
        cam_layout = QHBoxLayout()
        self.btn_camera = QPushButton("Start Camera")
        self.btn_camera.clicked.connect(self.toggle_camera)
        cam_layout.addWidget(self.btn_camera)
        right_layout.addLayout(cam_layout)
        
        layout.addLayout(right_layout, stretch=1)
        
        # Set default mode at the end of setup
        self.set_mode("STATIC")

    # --- UI Event Handlers ---
    def set_mode(self, mode):
        self.active_mode = mode
        self.btn_static_mode.setChecked(mode == "STATIC")
        self.btn_dynamic_mode.setChecked(mode == "DYNAMIC")
        self.static_stabilizer.reset()
        self.sequence_buffer.clear()
        if hasattr(self, 'vocab_label'):
            self.vocab_label.setVisible(mode == "DYNAMIC")
        
    def on_settings_changed(self):
        conf = self.conf_slider.value() / 100.0
        hold = self.hold_slider.value()
        
        self.static_stabilizer.static_k = hold
        self.static_stabilizer.static_conf = conf
        self.static_stabilizer.set_mode("STATIC")

    def on_auto_speak_changed(self, state):
        self.speech_engine.auto_speak = (state == 2)

    def on_captions_toggle_changed(self, state):
        self.captions_enabled = (state == 2)
        
    def on_space_clicked(self):
        self.sentence_builder.add_space()
        self.update_sentence_ui()
        
    def on_delete_clicked(self):
        self.sentence_builder.delete()
        self.update_sentence_ui()
        
    def on_clear_clicked(self):
        self.sentence_builder.clear()
        self.static_stabilizer.reset()
        self.sequence_buffer.clear()
        self.update_sentence_ui()
        
    def on_translate_clicked(self):
        self.sentence_builder.trigger_translation()
        
    def on_speak_clicked(self):
        text = self.sentence_builder.final_translated_sentence or self.sentence_builder.get_display_text()
        if text:
            self.speech_engine.speak(text)

    def toggle_camera(self):
        if not self.is_running:
            self.start_camera()
        else:
            self.stop_camera()

    def start_camera(self):
        # Determine camera index
        cam_text = self.cam_combo.currentText()
        if "Camera 1" in cam_text:
            idx = 1
        elif "Camera 0" in cam_text:
            idx = 0
        elif "Camera 2" in cam_text:
            idx = 2
        else:
            idx = 1 # Auto-detect defaults to 1 for active Brio 100

        self.video_thread = VideoThread(camera_index=idx)
        self.video_thread.error_signal.connect(self.handle_camera_error)
        self.video_thread.start()

        # Pull latest frame on a timer instead of processing every emitted signal.
        # If processing takes longer than the interval, the next tick is naturally
        # skipped by Qt's single-threaded event loop — no frame queue buildup.
        self._process_timer = QTimer()
        self._process_timer.timeout.connect(self._process_latest_frame)
        self._process_timer.start(30)  # ~33fps pull rate

        self.is_running = True
        self.btn_camera.setText("Stop Camera")


    def stop_camera(self):
        if self._process_timer is not None:
            self._process_timer.stop()
            self._process_timer = None
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
        self.is_running = False
        self.btn_camera.setText("Start Camera")
        self.video_label.setText("Camera Feed (Stopped)")
        self.video_label.setStyleSheet("background-color: black; color: white;")

    def handle_camera_error(self, err_msg):
        self.stop_camera()
        QMessageBox.warning(self, "Camera Error", f"Camera disconnected: {err_msg}")

    def on_dynamic_prediction_ready(self, label, conf):
        if self.active_mode != "DYNAMIC":
            return
            
        if conf >= 0.75:
            import time
            current_time = time.time()
            last_label = getattr(self, 'last_dynamic_label', "")
            last_time = getattr(self, 'last_dynamic_time', 0.0)
            
            if label == last_label and (current_time - last_time) < 2.0:
                pass # Debounce duplicate tokens
            else:
                self.last_dynamic_label = label
                self.last_dynamic_time = current_time
                if label == "more":
                    self.on_space_clicked()
                elif label == "no":
                    self.on_delete_clicked()
                elif label == "stop":
                    self.on_clear_clicked()
                else:
                    if self.sentence_builder.current_word:
                        self.sentence_builder.add_space()
                    self.sentence_builder.raw_words.append(label)
                    self.update_sentence_ui()
            self.last_dynamic_pred = f"{label} ({conf*100:.0f}%)"
        else:
            self.last_dynamic_pred = f"Noise/Unknown (<75%)"

    # --- Core Pipeline Processing ---
    def _process_latest_frame(self):
        """
        Called by QTimer. Pulls only the latest frame from the camera thread,
        skipping any frames that arrived while the previous call was processing.
        This eliminates the ever-growing signal queue that caused video feed delay.
        """
        if not hasattr(self, 'video_thread'):
            return
        frame = self.video_thread.grab_latest()
        if frame is not None:
            self.update_frame(frame)

    def update_frame(self, frame):
        try:
            # Process Landmarks
            results, landmarks_list = process_frame_for_landmarks(frame)
            draw_landmarks_on_frame(frame, results)
            
            predicted_label = ""
            confidence = 0.0
            
            if not landmarks_list:
                self.no_hand_frames += 1
                if self.no_hand_frames > 20:
                    self.warning_label.setText("No hand detected.")
                    self.last_dynamic_label = ""
                flat_input = [0.0] * 126
                raw_input = [0.0] * 126
                self.prev_flat_input = None
                motion = 0.0
            else:
                self.no_hand_frames = 0
                self.warning_label.setText("")
                flat_input = prepare_input_vector(landmarks_list)
                raw_input = prepare_raw_vector(landmarks_list)
                
                motion = 0.0
                if hasattr(self, 'prev_flat_input') and self.prev_flat_input is not None:
                    motion = float(np.mean(np.abs(np.array(flat_input) - self.prev_flat_input)))
                self.prev_flat_input = np.array(flat_input)

            # --- DIAGNOSTICS (Every 15 frames) ---
            self._frame_count = getattr(self, '_frame_count', 0) + 1
            if self._frame_count % 15 == 0:
                hands_found = bool(landmarks_list)
                print(f"Hands found: {hands_found}")
                if hands_found:
                    print(f"Raw coordinates shape: ({len(raw_input)},)")
                print(f"Active Mode: {self.active_mode}")
                if self.active_mode == "DYNAMIC":
                    print(f"[Dynamic Buffer: {len(self.sequence_buffer._gesture_frames)}/16]")
            # ------------------------------------

            # Route by Active Mode
            stabilized_label = ""
            
            if self.active_mode == "STATIC":
                if landmarks_list and motion < 0.008:
                    try:
                        predicted_label, confidence = self.static_classifier.predict_with_confidence(flat_input)
                    except Exception:
                        import traceback
                        traceback.print_exc()
                        predicted_label, confidence = "", 0.0
                else:
                    predicted_label, confidence = "", 0.0
                stabilized_label = self.static_stabilizer.process_prediction(predicted_label, confidence)
            else:
                # DYNAMIC mode: strict separation
                self.sequence_buffer.add_frame(raw_input)
                if self.sequence_buffer.has_gesture_ready():
                    seq_tensor = self.sequence_buffer.get_sequence()
                    if seq_tensor is not None and not self.dynamic_worker.isRunning():
                        self.dynamic_worker.seq_tensor = seq_tensor
                        self.dynamic_worker.start()

            # Sentence Builder Update
            if stabilized_label:
                if self.active_mode == "STATIC":
                    self.sentence_builder.add_char(stabilized_label)
                elif self.active_mode == "DYNAMIC":
                    if stabilized_label == "more":
                        self.on_space_clicked()
                    elif stabilized_label == "no":
                        self.on_delete_clicked()
                    elif stabilized_label == "stop":
                        self.on_clear_clicked()
                    else:
                        if self.sentence_builder.current_word:
                            self.sentence_builder.add_space()
                        self.sentence_builder.raw_words.append(stabilized_label)
                        self.update_sentence_ui()
            else:
                self.sentence_builder.reset_last_char()
                
            self.update_sentence_ui()

            # Update UI Elements
            if stabilized_label and self.active_mode == "STATIC":
                self.pred_label.setText(f"Prediction: {stabilized_label} ({confidence*100:.0f}%)")
            else:
                if self.active_mode == "DYNAMIC":
                    if self.sequence_buffer.is_recording():
                        # Display capturing progress based on frames added to the gesture buffer
                        frames_recorded = len(self.sequence_buffer._gesture_frames)
                        max_f = self.sequence_buffer._max_gesture_frames
                        self.pred_label.setText(f"Capturing: {frames_recorded}/{max_f} frames")
                    elif hasattr(self, 'last_dynamic_pred') and self.last_dynamic_pred:
                        self.pred_label.setText(f"Prediction: {self.last_dynamic_pred}")
                    else:
                        self.pred_label.setText("Prediction: -- (Waiting for motion)")
                else:
                    self.pred_label.setText("Prediction: --")

            # Add dynamic vocabulary overlay to the video feed
            if self.active_mode == "DYNAMIC":
                # Only show top ~8 words to avoid cluttering, or just wrap it. We'll show first few.
                top_words = self.dynamic_labels[:8]
                vocab_str = "Try: " + ", ".join(top_words) + ("..." if len(self.dynamic_labels) > 8 else "")
                cv2.putText(frame, vocab_str, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1, cv2.LINE_AA)

            # Live Caption Overlay (Mode 1: in-person captions burned onto video)
            if getattr(self, 'captions_enabled', True):
                current_sentence = self.sentence_builder.get_display_text()
                live_hint = stabilized_label if stabilized_label else ""
                draw_caption_overlay(frame, current_sentence, live_label=live_hint)

            # Convert to QImage and show
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_img = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            p = q_img.scaled(self.video_label.width(), self.video_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
            self.video_label.setPixmap(QPixmap.fromImage(p))
            
        except Exception as e:
            # Prevent single frame failure from crashing app
            import traceback
            traceback.print_exc()
            print(f"[ERROR] Frame drop/exception: {e}")

    def update_sentence_ui(self):
        self.sentence_text.setText(self.sentence_builder.get_display_text())
        # Auto-scroll to bottom
        cursor = self.sentence_text.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        self.sentence_text.setTextCursor(cursor)
        
        if hasattr(self, 'translation_text'):
            self.translation_text.setText(self.sentence_builder.final_translated_sentence)
        
    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = TranslatorApp()
    window.show()
    sys.exit(app.exec())
