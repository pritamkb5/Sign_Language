"""
stability.py — Module 5.2: Dual-Mode Temporal Prediction Stabilizer
====================================================================
Responsibility: Filter noisy predictions from real-time webcam streams.
Applies majority-voting and confidence-threshold hysteresis over historical prediction
windows to eliminate single-frame label flickering in both static and dynamic modes.

Features:
- Separate tuning for STATIC (alphabet) and DYNAMIC (word) modes.
- Majority-vote sliding window across last K frames.
- Confidence hysteresis filter (requires confidence >= threshold).
- Clear state reset on mode switching.
"""

from collections import deque, Counter


class PredictionStabilizer:
    """
    Debounces prediction noise by enforcing a confidence threshold
    and majority-vote sliding window over historical frames.

    Supports separate tuning for static vs dynamic modes:
    - Static mode: larger window (4 frames), higher confidence — letters
      are held poses that produce many consistent predictions.
    - Dynamic mode: smaller window (2 frames), lower confidence — gestures
      are transient and may only produce 1-3 prediction opportunities.
    """
    def __init__(self, k_threshold: int = 4, confidence_threshold: float = 0.50,
                 dynamic_k_threshold: int = 2, dynamic_confidence_threshold: float = 0.35):
        """
        Args:
            k_threshold: Window size for majority voting in STATIC mode.
            confidence_threshold: Minimum confidence in STATIC mode.
            dynamic_k_threshold: Window size for majority voting in DYNAMIC mode.
            dynamic_confidence_threshold: Minimum confidence in DYNAMIC mode.
        """
        # Static mode settings
        self.static_k = k_threshold
        self.static_conf = confidence_threshold

        # Dynamic mode settings
        self.dynamic_k = dynamic_k_threshold
        self.dynamic_conf = dynamic_confidence_threshold

        # Active settings (default to static)
        self.k_threshold = k_threshold
        self.confidence_threshold = confidence_threshold

        self.current_confirmed_prediction = ""
        self.consecutive_count = 0
        self.candidate_prediction = ""
        self.history = deque(maxlen=k_threshold)

    def set_mode(self, mode: str) -> None:
        """
        Switch stabilizer parameters for the given mode.

        Args:
            mode: 'STATIC' or 'DYNAMIC'
        """
        if mode == "DYNAMIC":
            self.k_threshold = self.dynamic_k
            self.confidence_threshold = self.dynamic_conf
        else:
            self.k_threshold = self.static_k
            self.confidence_threshold = self.static_conf

        # Resize history to new window
        old_items = list(self.history)
        self.history = deque(maxlen=self.k_threshold)
        for item in old_items[-self.k_threshold:]:
            self.history.append(item)

    def reset(self) -> None:
        """Resets history and confirmed states upon mode switching."""
        self.current_confirmed_prediction = ""
        self.consecutive_count = 0
        self.candidate_prediction = ""
        self.history.clear()

    def process_prediction(self, label: str, confidence: float) -> str:
        """
        Processes a raw prediction and returns the stabilized confirmed label.

        Args:
            label: Raw string output from classifier.
            confidence: Confidence probability (0.0 to 1.0).

        Returns:
            Stabilized prediction string.
        """
        # If confidence is below threshold, filter out
        if confidence < self.confidence_threshold:
            label = ""

        if not label:
            self.history.append("")
            # If majority of recent window is empty, clear confirmed
            if self.history.count("") > self.k_threshold // 2:
                self.current_confirmed_prediction = ""
            return self.current_confirmed_prediction

        self.history.append(label)

        # Majority vote over history window
        non_empty = [item for item in self.history if item != ""]
        if non_empty:
            counts = Counter(non_empty)
            most_common, count = counts.most_common(1)[0]
            
            # Enforce strict hold: require 80% consistency for long windows (STATIC) to avoid garbage flickering
            required_count = max(1, int(self.k_threshold * 0.8)) if self.k_threshold > 2 else max(1, self.k_threshold // 2)
            
            if count >= required_count:
                self.current_confirmed_prediction = most_common

        return self.current_confirmed_prediction


if __name__ == "__main__":
    print("[INFO] Running dual-mode PredictionStabilizer self-test...")

    # Test Static mode
    print("\n--- Static Mode Test ---")
    stabilizer = PredictionStabilizer(k_threshold=4, confidence_threshold=0.50,
                                      dynamic_k_threshold=2, dynamic_confidence_threshold=0.40)
    stabilizer.set_mode("STATIC")

    test_stream = [
        ("A", 0.8),
        ("A", 0.85),
        ("B", 0.4),  # filtered out by low confidence
        ("A", 0.9),  # confirmed "A"
    ]

    for idx, (raw_pred, conf) in enumerate(test_stream):
        out = stabilizer.process_prediction(raw_pred, conf)
        print(f"Frame {idx+1}: Raw '{raw_pred}' ({conf}) -> Stabilized: '{out}'")

    # Test Dynamic mode
    print("\n--- Dynamic Mode Test ---")
    stabilizer.reset()
    stabilizer.set_mode("DYNAMIC")

    test_stream = [
        ("hello", 0.75),
        ("hello", 0.82),   # should confirm with k=2
        ("", 0.0),
        ("thanks", 0.90),
        ("thanks", 0.85),  # should confirm
    ]

    for idx, (raw_pred, conf) in enumerate(test_stream):
        out = stabilizer.process_prediction(raw_pred, conf)
        print(f"Frame {idx+1}: Raw '{raw_pred}' ({conf}) -> Stabilized: '{out}'")

    print("\n[SUCCESS] PredictionStabilizer self-test complete.")
