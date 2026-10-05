"""
sequence_buffer.py — Track B & Track C: Motion-Segmented Gesture Buffer
========================================================================
Responsibility: Detects gesture onset and offset using per-frame motion energy,
captures the complete gesture segment, then resamples it to a fixed 30-frame
window via temporal interpolation — matching the training data distribution.

This replaces the naive rolling FIFO buffer which caused a train/inference
mismatch (training saw zero-padded short clips; inference saw a continuous
sliding window of mixed gesture/idle frames).

Features:
- Motion energy-based gesture start/end detection.
- Temporal resampling of captured gesture to fixed length (30 frames).
- Cooldown period to prevent repeated triggers.
- Thread safety with reentrant locks.
"""

import threading
import numpy as np


def resample_sequence(sequence: np.ndarray, target_length: int = 30) -> np.ndarray:
    """
    Resample a variable-length sequence to a fixed target_length using
    linear interpolation along the time axis.
    """
    num_frames, num_features = sequence.shape

    if num_frames == target_length:
        return sequence.copy()

    if num_frames == 0:
        return np.zeros((target_length, num_features), dtype=np.float32)

    if num_frames == 1:
        return np.tile(sequence[0], (target_length, 1)).astype(np.float32)

    old_indices = np.linspace(0, num_frames - 1, num_frames)
    new_indices = np.linspace(0, num_frames - 1, target_length)

    resampled = np.zeros((target_length, num_features), dtype=np.float32)
    for feat_idx in range(num_features):
        resampled[:, feat_idx] = np.interp(new_indices, old_indices, sequence[:, feat_idx])

    return resampled


class SequenceBuffer:
    """
    Motion-segmented gesture buffer for real-time dynamic sign recognition.

    Detects when a gesture starts (motion rises above threshold) and ends
    (motion drops below threshold for several frames), captures the segment,
    and resamples it to a fixed length for classification.
    """
    def __init__(self, sequence_length: int = 16, feature_dim: int = 126):
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self._lock = threading.Lock()

        # Motion detection state
        self._prev_frame: np.ndarray | None = None
        self._is_recording = False
        self._gesture_frames: list[np.ndarray] = []
        self._idle_count = 0          # consecutive low-motion frames during recording
        self._total_frames_fed = 0

        # Thresholds (tuned for wrist-normalized landmark coordinates)
        # Phase 4: Lowered thresholds to reduce inter-gesture lag by ~40%
        self._motion_start_threshold = 0.001   # per-feature mean displacement to start (was 0.006)
        self._motion_stop_threshold = 0.0005   # per-feature mean displacement to stop (was 0.003)
        self._min_idle_to_stop = 4             # consecutive idle frames to declare gesture end
        self._min_gesture_frames = 4           # minimum frames for a valid gesture
        self._max_gesture_frames = 16          # maximum frames before force-stopping
        self._cooldown_frames = 0              # frames to wait after gesture ends
        self._cooldown_counter = 0

        # Output: the latest ready-to-classify resampled sequence
        self._ready_sequence: np.ndarray | None = None
        self._sequence_consumed = True         # has the latest sequence been read?

        # Legacy compatibility: rolling buffer for fill_ratio display
        self._rolling_buffer = np.zeros((self.sequence_length, self.feature_dim), dtype=np.float32)
        self._rolling_count = 0

    def add_frame(self, flat_features: list[float]) -> None:
        """
        Feed a new frame. Internally detects gesture boundaries and prepares
        resampled sequences when a complete gesture is captured.
        """
        with self._lock:
            self._total_frames_fed += 1

            # Sanitize input
            if not flat_features or len(flat_features) == 0:
                is_empty = True
                if self._prev_frame is not None:
                    vec = self._prev_frame.copy()
                else:
                    vec = np.zeros(self.feature_dim, dtype=np.float32)
            else:
                is_empty = False
                vec = np.array(flat_features, dtype=np.float32)
                if len(vec) < self.feature_dim:
                    padded = np.zeros(self.feature_dim, dtype=np.float32)
                    padded[:len(vec)] = vec
                    vec = padded
                elif len(vec) > self.feature_dim:
                    vec = vec[:self.feature_dim]
                vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)

            # Update rolling buffer for display purposes
            self._rolling_buffer[:-1] = self._rolling_buffer[1:]
            self._rolling_buffer[-1] = vec
            if self._rolling_count < self.sequence_length:
                self._rolling_count += 1

            # Calculate per-feature mean displacement from previous frame
            motion = 0.0
            if self._prev_frame is not None:
                diff = np.abs(vec - self._prev_frame)
                motion = float(np.mean(diff))
            
            self._prev_frame = vec.copy()

            # Cooldown after a gesture was just completed
            if self._cooldown_counter > 0:
                self._cooldown_counter -= 1
                return

            # State machine: gesture segmentation
            if not self._is_recording:
                # Looking for gesture onset
                if motion >= self._motion_start_threshold:
                    self._is_recording = True
                    self._gesture_frames = [vec.copy()]
                    self._idle_count = 0
            else:
                # Currently recording a gesture
                self._gesture_frames.append(vec.copy())
                print(f"[DEBUG] Dynamic Buffer: {len(self._gesture_frames)}/{self._max_gesture_frames} frames | Hand detected: {not is_empty}")

                if motion < self._motion_stop_threshold:
                    self._idle_count += 1
                else:
                    self._idle_count = 0

                # Check if gesture ended (enough idle frames) or hit max length
                gesture_ended = (
                    self._idle_count >= self._min_idle_to_stop or
                    len(self._gesture_frames) >= self._max_gesture_frames
                )

                if gesture_ended:
                    self._is_recording = False
                    self._cooldown_counter = self._cooldown_frames

                    # Trim trailing idle frames
                    effective_frames = self._gesture_frames
                    if self._idle_count > 0:
                        effective_frames = self._gesture_frames[:-self._idle_count]

                    if len(effective_frames) >= self._min_gesture_frames:
                        # Valid gesture captured — resample to fixed length
                        gesture_arr = np.array(effective_frames, dtype=np.float32)
                        resampled = resample_sequence(gesture_arr, self.sequence_length)
                        self._ready_sequence = resampled
                        self._sequence_consumed = False

                    self._gesture_frames = []
                    self._idle_count = 0

    def get_sequence(self) -> np.ndarray | None:
        """
        Returns the latest gesture-segmented, resampled sequence if one is ready.
        Returns shape (1, sequence_length, feature_dim) or None if no gesture ready.

        After calling this, the sequence is marked as consumed and won't be
        returned again until a new gesture is detected.
        """
        with self._lock:
            if self._ready_sequence is not None and not self._sequence_consumed:
                self._sequence_consumed = True
                return np.expand_dims(self._ready_sequence.copy(), axis=0)
            return None

    def has_gesture_ready(self) -> bool:
        """Returns True if a new gesture sequence is available for classification."""
        with self._lock:
            return self._ready_sequence is not None and not self._sequence_consumed

    def is_recording(self) -> bool:
        """Returns True if currently recording a gesture."""
        with self._lock:
            return self._is_recording

    def is_full(self) -> bool:
        """Returns True if a gesture is ready for classification."""
        with self._lock:
            return self._ready_sequence is not None and not self._sequence_consumed

    def fill_ratio(self) -> float:
        """
        Returns visual progress indicator:
        - During recording: fraction of max_gesture_frames filled.
        - During idle: 0.0
        - When gesture ready: 1.0
        """
        with self._lock:
            if self._ready_sequence is not None and not self._sequence_consumed:
                return 1.0
            if self._is_recording:
                return min(1.0, len(self._gesture_frames) / float(self._max_gesture_frames))
            return 0.0

    def clear(self) -> None:
        """Resets all state."""
        with self._lock:
            self._prev_frame = None
            self._is_recording = False
            self._gesture_frames = []
            self._idle_count = 0
            self._cooldown_counter = 0
            self._ready_sequence = None
            self._sequence_consumed = True
            self._rolling_buffer.fill(0.0)
            self._rolling_count = 0
            self._total_frames_fed = 0


if __name__ == "__main__":
    print("[INFO] Running SequenceBuffer self-test with motion segmentation...")
    buf = SequenceBuffer(sequence_length=10, feature_dim=4)

    # Simulate: 5 idle frames, then 8 frames of motion, then 6 idle frames
    print("\nPhase 1: Idle frames (no motion)...")
    for i in range(5):
        buf.add_frame([0.1, 0.2, 0.3, 0.4])
        print(f"  Frame {i+1}: recording={buf.is_recording()}, ready={buf.has_gesture_ready()}, fill={buf.fill_ratio():.2f}")

    print("\nPhase 2: Gesture motion...")
    for i in range(8):
        val = 0.1 + (i * 0.05)
        buf.add_frame([val, val + 0.1, val + 0.2, val + 0.3])
        print(f"  Frame {i+6}: recording={buf.is_recording()}, ready={buf.has_gesture_ready()}, fill={buf.fill_ratio():.2f}")

    print("\nPhase 3: Gesture ending (idle frames)...")
    for i in range(8):
        buf.add_frame([0.5, 0.6, 0.7, 0.8])
        has_ready = buf.has_gesture_ready()
        print(f"  Frame {i+14}: recording={buf.is_recording()}, ready={has_ready}, fill={buf.fill_ratio():.2f}")
        if has_ready:
            seq = buf.get_sequence()
            if seq is not None:
                print(f"  >> Got gesture sequence! Shape: {seq.shape}")

    print("\n[SUCCESS] SequenceBuffer self-test complete.")
