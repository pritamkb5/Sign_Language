import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.sentence_builder import SentenceBuilder
import cv2
import threading
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QSlider, 
                             QComboBox, QTextEdit, QCheckBox, QMessageBox,
                             QFrame, QSizePolicy, QStatusBar, QScrollArea)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QImage, QPixmap, QFont, QPalette, QColor

# Theme / Stylesheet import
from src.styles import get_stylesheet

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
        self.resize(1180, 780)
        self.setMinimumSize(980, 680)
        self.setStyleSheet(get_stylesheet())
        
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
        
    def _set_camera_idle(self):
        """Displays a clean modern idle card placeholder when camera feed is not running."""
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText(
            "<div style='text-align: center; padding: 40px;'>"
            "<p style='font-size: 46px; margin: 0 0 14px 0; color: #475569;'>📷</p>"
            "<p style='font-size: 16px; font-weight: 600; color: #E2E8F0; margin: 0 0 8px 0;'>Camera is off</p>"
            "<p style='font-size: 13px; color: #64748B; margin: 0;'>Press <b style='color: #6C63FF;'>Start Camera</b> to begin recognition</p>"
            "</div>"
        )

    def setup_ui(self):
        central_widget = QWidget()
        central_widget.setObjectName("central_widget")
        self.setCentralWidget(central_widget)
        
        root_layout = QVBoxLayout(central_widget)
        root_layout.setContentsMargins(16, 12, 16, 12)
        root_layout.setSpacing(12)
        
        # --- Top Modern Header ---
        header_frame = QFrame()
        header_frame.setObjectName("header_bar")
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(4, 2, 4, 8)
        header_layout.setSpacing(10)
        
        lbl_title = QLabel("Sign Language Translator")
        lbl_title.setObjectName("app_title")
        
        lbl_badge = QLabel("ALPHA")
        lbl_badge.setObjectName("app_badge")
        
        lbl_sub = QLabel("• Real-Time Gesture & Sign Recognition")
        lbl_sub.setObjectName("app_subtitle")
        
        status_pill = QLabel("● Engine Ready")
        status_pill.setObjectName("status_pill")
        
        header_layout.addWidget(lbl_title)
        header_layout.addWidget(lbl_badge)
        header_layout.addWidget(lbl_sub)
        header_layout.addStretch()
        header_layout.addWidget(status_pill)
        root_layout.addWidget(header_frame)
        
        # --- Main Body Grid / Columns ---
        body_layout = QHBoxLayout()
        body_layout.setSpacing(14)
        
        # =========================================================
        # Left Panel: Live Camera Feed & Prediction Badge
        # =========================================================
        card_camera = QFrame()
        card_camera.setObjectName("card_camera")
        cam_card_layout = QVBoxLayout(card_camera)
        cam_card_layout.setContentsMargins(16, 16, 16, 16)
        cam_card_layout.setSpacing(10)
        
        lbl_cam_section = QLabel("LIVE CAMERA PREVIEW")
        lbl_cam_section.setProperty("section", True)
        cam_card_layout.addWidget(lbl_cam_section)
        
        self.video_label = QLabel()
        self.video_label.setObjectName("video_label")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(540, 400)
        self.video_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.video_label.setToolTip("Live webcam preview with real-time hand landmark tracking")
        self._set_camera_idle()
        cam_card_layout.addWidget(self.video_label, stretch=1)
        
        lbl_pred_section = QLabel("CURRENT PREDICTION")
        lbl_pred_section.setProperty("section", True)
        cam_card_layout.addWidget(lbl_pred_section)
        
        self.pred_label = QLabel("Prediction: --")
        self.pred_label.setObjectName("pred_badge")
        self.pred_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pred_label.setToolTip("Current recognized sign and model confidence")
        cam_card_layout.addWidget(self.pred_label)
        
        self.warning_label = QLabel("")
        self.warning_label.setObjectName("warning_label")
        self.warning_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cam_card_layout.addWidget(self.warning_label)
        
        body_layout.addWidget(card_camera, stretch=3)
        
        # =========================================================
        # Right Column: Scrollable Transcription & Settings Cards
        # =========================================================
        right_scroll = QScrollArea()
        right_scroll.setObjectName("right_scroll_area")
        right_scroll.setWidgetResizable(True)
        right_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        right_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        right_scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        right_container = QWidget()
        right_container.setObjectName("right_scroll_content")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 8, 0)
        right_layout.setSpacing(14)
        
        # --- Card 1: Input & Translation Card ---
        card_trans = QFrame()
        card_trans.setObjectName("card_transcription")
        trans_layout = QVBoxLayout(card_trans)
        trans_layout.setContentsMargins(18, 18, 18, 18)
        trans_layout.setSpacing(10)
        
        lbl_trans_header = QLabel("TRANSCRIPTION & TRANSLATION")
        lbl_trans_header.setProperty("section", True)
        lbl_trans_header.setMinimumHeight(18)
        trans_layout.addWidget(lbl_trans_header)
        
        lbl_raw = QLabel("RAW SIGN INPUT")
        lbl_raw.setProperty("section", True)
        lbl_raw.setMinimumHeight(16)
        trans_layout.addWidget(lbl_raw)
        
        self.sentence_text = QTextEdit()
        self.sentence_text.setObjectName("sentence_text")
        self.sentence_text.setReadOnly(True)
        self.sentence_text.setFont(QFont("Segoe UI Variable", 13))
        self.sentence_text.setMinimumHeight(60)
        self.sentence_text.setMaximumHeight(70)
        self.sentence_text.setPlaceholderText("Detected letters and signs will accumulate here...")
        self.sentence_text.setToolTip("Stream of raw detected sign tokens")
        pal_s = self.sentence_text.palette()
        pal_s.setColor(QPalette.ColorRole.PlaceholderText, QColor("#94A3B8"))
        self.sentence_text.setPalette(pal_s)
        trans_layout.addWidget(self.sentence_text)
        
        lbl_trans = QLabel("NATURAL LANGUAGE TRANSLATION")
        lbl_trans.setProperty("section", True)
        lbl_trans.setMinimumHeight(16)
        trans_layout.addWidget(lbl_trans)
        
        self.translation_text = QTextEdit()
        self.translation_text.setObjectName("translation_text")
        self.translation_text.setReadOnly(True)
        self.translation_text.setFont(QFont("Segoe UI Variable", 14, QFont.Weight.Bold))
        self.translation_text.setMinimumHeight(70)
        self.translation_text.setMaximumHeight(85)
        self.translation_text.setPlaceholderText("Translated fluent English sentence will appear here...")
        self.translation_text.setToolTip("Contextual AI translation")
        pal_t = self.translation_text.palette()
        pal_t.setColor(QPalette.ColorRole.PlaceholderText, QColor("#94A3B8"))
        self.translation_text.setPalette(pal_t)
        trans_layout.addWidget(self.translation_text)
        
        # Secondary Action Buttons Row
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(8)
        
        self.btn_space = QPushButton("␣ Space")
        self.btn_space.setObjectName("btn_space")
        self.btn_space.setMinimumHeight(36)
        self.btn_space.setToolTip("Insert a space delimiter (Space key)")
        
        self.btn_delete = QPushButton("⌫ Delete")
        self.btn_delete.setObjectName("btn_delete")
        self.btn_delete.setMinimumHeight(36)
        self.btn_delete.setToolTip("Delete the last token")
        
        self.btn_clear = QPushButton("✕ Clear")
        self.btn_clear.setObjectName("btn_clear")
        self.btn_clear.setMinimumHeight(36)
        self.btn_clear.setToolTip("Clear all text and reset stabilizers")
        
        self.btn_translate = QPushButton("✦ Translate")
        self.btn_translate.setObjectName("btn_translate")
        self.btn_translate.setMinimumHeight(36)
        self.btn_translate.setToolTip("Translate accumulated sign tokens into natural English")
        
        self.btn_speak = QPushButton("🔊 Speak")
        self.btn_speak.setObjectName("btn_speak")
        self.btn_speak.setMinimumHeight(36)
        self.btn_speak.setToolTip("Read aloud the sentence via Text-to-Speech")
        
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
        trans_layout.addLayout(btn_layout)
        
        right_layout.addWidget(card_trans)
        
        # --- Card 2: Settings & Controls Card ---
        card_settings = QFrame()
        card_settings.setObjectName("card_settings")
        set_layout = QVBoxLayout(card_settings)
        set_layout.setContentsMargins(18, 18, 18, 18)
        set_layout.setSpacing(16)
        
        lbl_set_header = QLabel("SETTINGS & CONTROLS")
        lbl_set_header.setProperty("section", True)
        lbl_set_header.setMinimumHeight(18)
        set_layout.addWidget(lbl_set_header)
        
        # Mode Segmented Control Section
        mode_box = QVBoxLayout()
        mode_box.setSpacing(6)
        lbl_mode = QLabel("RECOGNITION MODE")
        lbl_mode.setProperty("section", True)
        lbl_mode.setMinimumHeight(16)
        mode_box.addWidget(lbl_mode)
        
        seg_frame = QFrame()
        seg_frame.setObjectName("segmented_control")
        seg_frame.setFixedHeight(46)
        seg_layout = QHBoxLayout(seg_frame)
        seg_layout.setContentsMargins(3, 3, 3, 3)
        seg_layout.setSpacing(3)
        
        self.btn_static_mode = QPushButton("Static Mode")
        self.btn_dynamic_mode = QPushButton("Dynamic Mode")
        self.btn_static_mode.setFixedHeight(40)
        self.btn_dynamic_mode.setFixedHeight(40)
        self.btn_static_mode.setCheckable(True)
        self.btn_dynamic_mode.setCheckable(True)
        self.btn_static_mode.setToolTip("Recognize static ASL alphabet fingerspelling (A-Z, 0-9)")
        self.btn_dynamic_mode.setToolTip("Recognize dynamic continuous words and motion gestures")
        self.btn_static_mode.clicked.connect(lambda: self.set_mode("STATIC"))
        self.btn_dynamic_mode.clicked.connect(lambda: self.set_mode("DYNAMIC"))
        seg_layout.addWidget(self.btn_static_mode)
        seg_layout.addWidget(self.btn_dynamic_mode)
        mode_box.addWidget(seg_frame)
        
        # Dynamic Vocabulary List
        self.vocab_label = QLabel("Dynamic Vocab: " + ", ".join(self.dynamic_labels))
        self.vocab_label.setObjectName("vocab_label")
        self.vocab_label.setWordWrap(True)
        self.vocab_label.setVisible(False)
        self.vocab_label.setMinimumHeight(24)
        mode_box.addWidget(self.vocab_label)
        set_layout.addLayout(mode_box)
        
        # Camera Device Selector Section
        cam_sel_layout = QVBoxLayout()
        cam_sel_layout.setSpacing(6)
        lbl_cam = QLabel("CAMERA DEVICE")
        lbl_cam.setProperty("section", True)
        lbl_cam.setMinimumHeight(16)
        cam_sel_layout.addWidget(lbl_cam)
        
        self.cam_combo = QComboBox()
        self.cam_combo.setObjectName("cam_combo")
        self.cam_combo.setFixedHeight(40)
        self.cam_combo.addItems([
            "Auto-Detect Active Camera",
            "Camera 1 (Logitech Brio 100)",
            "Camera 0 (Integrated Camera)",
            "Camera 2",
            "Synthetic Camera (Mock)"
        ])
        self.cam_combo.setToolTip("Select camera video device")
        cam_sel_layout.addWidget(self.cam_combo)
        set_layout.addLayout(cam_sel_layout)
        
        # Toggle Switches Section (with generous spacing above and below)
        chk_layout = QVBoxLayout()
        chk_layout.setSpacing(10)
        chk_layout.setContentsMargins(0, 4, 0, 4)
        
        lbl_chk_header = QLabel("AUTOMATION & OVERLAYS")
        lbl_chk_header.setProperty("section", True)
        lbl_chk_header.setMinimumHeight(16)
        chk_layout.addWidget(lbl_chk_header)
        
        self.auto_speak_check = QCheckBox("Auto-Speak Confirmed Words")
        self.auto_speak_check.setObjectName("auto_speak_check")
        self.auto_speak_check.setMinimumHeight(32)
        self.auto_speak_check.setToolTip("Automatically speak confirmed signs as they are recognized")
        self.auto_speak_check.stateChanged.connect(self.on_auto_speak_changed)
        chk_layout.addWidget(self.auto_speak_check)
        
        self.captions_check = QCheckBox("Show Live Captions On Video")
        self.captions_check.setObjectName("captions_check")
        self.captions_check.setChecked(True)
        self.captions_check.setMinimumHeight(32)
        self.captions_check.setToolTip("Render real-time subtitles directly on the camera frame")
        self.captions_check.stateChanged.connect(self.on_captions_toggle_changed)
        chk_layout.addWidget(self.captions_check)
        set_layout.addLayout(chk_layout)
        
        # Detection Thresholds Sliders Section
        sliders_box = QVBoxLayout()
        sliders_box.setSpacing(12)
        
        lbl_sliders_header = QLabel("DETECTION THRESHOLDS")
        lbl_sliders_header.setProperty("section", True)
        lbl_sliders_header.setMinimumHeight(16)
        sliders_box.addWidget(lbl_sliders_header)
        
        # Confidence Slider
        conf_box = QVBoxLayout()
        conf_box.setSpacing(4)
        conf_head = QHBoxLayout()
        lbl_conf = QLabel("Confidence Threshold")
        lbl_conf.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: 500;")
        lbl_conf.setMinimumHeight(18)
        self.conf_val_label = QLabel("40%")
        self.conf_val_label.setProperty("class", "slider_val")
        self.conf_val_label.setMinimumHeight(18)
        conf_head.addWidget(lbl_conf)
        conf_head.addStretch()
        conf_head.addWidget(self.conf_val_label)
        conf_box.addLayout(conf_head)
        
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setObjectName("conf_slider")
        self.conf_slider.setRange(30, 95)
        self.conf_slider.setValue(40)
        self.conf_slider.setMinimumHeight(24)
        self.conf_slider.setToolTip("Minimum prediction confidence (30% - 95%)")
        self.conf_slider.valueChanged.connect(self.on_settings_changed)
        conf_box.addWidget(self.conf_slider)
        sliders_box.addLayout(conf_box)
        
        # Hold Duration Slider
        hold_box = QVBoxLayout()
        hold_box.setSpacing(4)
        hold_head = QHBoxLayout()
        lbl_hold = QLabel("Hold Duration (Frames)")
        lbl_hold.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: 500;")
        lbl_hold.setMinimumHeight(18)
        self.hold_val_label = QLabel("15 frames")
        self.hold_val_label.setProperty("class", "slider_val")
        self.hold_val_label.setMinimumHeight(18)
        hold_head.addWidget(lbl_hold)
        hold_head.addStretch()
        hold_head.addWidget(self.hold_val_label)
        hold_box.addLayout(hold_head)
        
        self.hold_slider = QSlider(Qt.Orientation.Horizontal)
        self.hold_slider.setObjectName("hold_slider")
        self.hold_slider.setRange(5, 45)
        self.hold_slider.setValue(15)
        self.hold_slider.setMinimumHeight(24)
        self.hold_slider.setToolTip("Consecutive stable frames required before accepting a sign (5 - 45)")
        self.hold_slider.valueChanged.connect(self.on_settings_changed)
        hold_box.addWidget(self.hold_slider)
        sliders_box.addLayout(hold_box)
        
        set_layout.addLayout(sliders_box)
        
        # Primary Action: Start/Stop Camera
        self.btn_camera = QPushButton("▶  Start Camera")
        self.btn_camera.setObjectName("btn_camera")
        self.btn_camera.setFixedHeight(42)
        self.btn_camera.setToolTip("Start or stop camera video capture and inference")
        self.btn_camera.clicked.connect(self.toggle_camera)
        set_layout.addWidget(self.btn_camera)
        
        right_layout.addWidget(card_settings)
        right_layout.addStretch()
        
        right_scroll.setWidget(right_container)
        body_layout.addWidget(right_scroll, stretch=2)
        
        root_layout.addLayout(body_layout, stretch=1)
        
        # Status Bar
        status = self.statusBar()
        if status:
            status.showMessage("Ready  |  Dual-Mode Recognition Engine Active")
        
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
        if hasattr(self, 'conf_val_label'):
            self.conf_val_label.setText(f"{self.conf_slider.value()}%")
        if hasattr(self, 'hold_val_label'):
            self.hold_val_label.setText(f"{hold} frames")

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
        self.btn_camera.setText("⏹  Stop Camera")
        self.btn_camera.setProperty("running", True)
        self.btn_camera.style().unpolish(self.btn_camera)
        self.btn_camera.style().polish(self.btn_camera)
        if self.statusBar():
            self.statusBar().showMessage("Camera Running  |  Real-time inference active")


    def stop_camera(self):
        if self._process_timer is not None:
            self._process_timer.stop()
            self._process_timer = None
        if hasattr(self, 'video_thread'):
            self.video_thread.stop()
        self.is_running = False
        self.btn_camera.setText("▶  Start Camera")
        self.btn_camera.setProperty("running", False)
        self.btn_camera.style().unpolish(self.btn_camera)
        self.btn_camera.style().polish(self.btn_camera)
        self._set_camera_idle()
        if self.statusBar():
            self.statusBar().showMessage("Camera Stopped  |  System ready")

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
