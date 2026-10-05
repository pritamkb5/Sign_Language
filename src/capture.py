"""
capture.py — Module 1: Automatic Multi-Camera & Multi-Backend Capture
======================================================================
Responsibility: Automatically scan all physical webcams connected to the PC
(e.g., Logitech Brio 100 USB webcam vs Integrated Laptop Camera), and select
the camera index that outputs a valid, optical non-black video feed.

Features:
- Prefers DirectShow (`CAP_DSHOW`) on Windows to avoid MSMF camera locks.
- Automatically searches indices 0, 1, 2, 3 and ranks them by pixel brightness / activity.
- Automatic fallback to Synthetic Camera Mode if all physical cameras are dark.
"""

import cv2
import time
import math
import numpy as np
import platform
import threading



class MockVideoCapture:
    """
    Synthetic camera stream generator. Used when physical webcam is missing,
    disabled, blocked by a privacy shutter, or unresponsive.
    Renders a bright 640x480 video feed with animated gridlines and hand skeleton.
    """
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        self._is_opened = True
        self._start_time = time.time()
        print("[INFO] Synthetic Camera Stream active (Mock VideoCapture enabled).")

    def isOpened(self) -> bool:
        return self._is_opened

    def set(self, propId: int, value: float) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        """Generate a bright 640x480 synthetic camera frame with realistic hand motion."""
        t = time.time() - self._start_time
        
        # Bright slate background
        frame = np.full((self.height, self.width, 3), (40, 45, 55), dtype=np.uint8)

        # Draw grid background
        for x in range(0, self.width, 40):
            cv2.line(frame, (x, 0), (x, self.height), (60, 65, 75), 1)
        for y in range(0, self.height, 40):
            cv2.line(frame, (0, y), (self.width, y), (60, 65, 75), 1)

        # Oscillating synthetic hand position
        cx = int(self.width / 2 + math.sin(t * 2.2) * 130)
        cy = int(self.height / 2 + math.cos(t * 1.4) * 70)

        # Draw synthetic hand palm
        cv2.circle(frame, (cx, cy), 38, (80, 180, 100), -1)
        cv2.circle(frame, (cx, cy), 38, (255, 255, 255), 2)
        
        # Draw fingers
        for angle_deg in [-55, -25, 5, 35, 65]:
            rad = math.radians(angle_deg)
            fx = int(cx + math.sin(rad + t * 0.8) * 75)
            fy = int(cy - math.cos(rad) * 85)
            cv2.line(frame, (cx, cy), (fx, fy), (180, 255, 180), 5, cv2.LINE_AA)
            cv2.circle(frame, (fx, fy), 8, (0, 230, 255), -1)

        # UI Notice banner at bottom
        cv2.rectangle(frame, (10, self.height - 55), (self.width - 10, self.height - 10), (20, 25, 35), -1)
        cv2.rectangle(frame, (10, self.height - 55), (self.width - 10, self.height - 10), (0, 180, 255), 1)
        cv2.putText(frame, "SYNTHETIC CAMERA ACTIVE (No Working Optical Stream on Selected Index)",
                    (20, self.height - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 230, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "Select another camera index (e.g. Camera 1 for Brio 100) or check camera slider.",
                    (20, self.height - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)

        time.sleep(1.0 / self.fps)
        return True, frame

    def release(self) -> None:
        self._is_opened = False


class ThreadedCamera:
    """
    Wraps cv2.VideoCapture to perform frame grabbing in a background thread.
    This ensures that .read() always returns the latest frame instantly,
    preventing buffer backlog and lag when processing is slower than camera frame rate.
    """
    def __init__(self, cap):
        self.cap = cap
        self.ret = False
        self.frame = None
        self.running = True
        self.lock = threading.Lock()
        
        # Read the first frame synchronously to initialize self.ret and self.frame
        self.ret, self.frame = self.cap.read()
        
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    with self.lock:
                        self.ret = ret
                        self.frame = frame
                else:
                    time.sleep(0.01)
            else:
                time.sleep(0.01)
        
    def read(self) -> tuple[bool, np.ndarray]:
        with self.lock:
            if self.frame is None:
                return False, np.zeros((480, 640, 3), dtype=np.uint8)
            return self.ret, self.frame.copy()

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    def set(self, propId, value) -> bool:
        return self.cap.set(propId, value)

    def get(self, propId) -> float:
        return self.cap.get(propId)

    def release(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


def find_working_camera_index(preferred_index: int = 0) -> tuple[int, int, float]:
    """
    Scans physical camera indices (0, 1, 2, 3) using DirectShow / MSMF.
    Returns (best_index, best_backend, max_brightness).
    """
    is_windows = platform.system() == "Windows"
    backends = [cv2.CAP_DSHOW] if is_windows else [cv2.CAP_ANY]
    indices_to_scan = [preferred_index] + [i for i in range(4) if i != preferred_index]

    best_idx = None
    best_backend = None
    best_mean = -1.0

    print("[INFO] Scanning connected physical webcams...")

    for idx in indices_to_scan:
        for backend in backends:
            try:
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        mean_val = float(np.mean(frame))
                        b_name = "DSHOW" if backend == cv2.CAP_DSHOW else "MSMF"
                        print(f"  -> Camera Index {idx} [{b_name}]: Pixel Mean = {mean_val:.2f}")

                        if mean_val > 5.0 and mean_val > best_mean:
                            best_mean = mean_val
                            best_idx = idx
                            best_backend = backend

                    cap.release()
            except Exception:
                pass

        if best_idx is not None and best_mean > 20.0:
            # Found a bright, active physical camera — stop scanning
            break

    if best_idx is not None:
        return best_idx, best_backend, best_mean
    return preferred_index, cv2.CAP_DSHOW if is_windows else cv2.CAP_ANY, 0.0


def open_camera(camera_index: int = 0, allow_mock: bool = True) -> ThreadedCamera | MockVideoCapture:
    # Robust Index Scanning
    indices_to_try = [camera_index]
    for i in [0, 1, 2]:
        if i not in indices_to_try:
            indices_to_try.append(i)

    for idx in indices_to_try:
        if idx < 0:
            continue
            
        # Use DirectShow, which we know connects to your camera
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            # --- THE FIX: Force the MJPG codec to eliminate the rainbow static ---
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            # ---------------------------------------------------------------------
            
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Read 10 throwaway frames to let the hardware sensor wake up
            for _ in range(10):
                cap.read()
                
            print(f"[SUCCESS] Physical camera opened on index {idx}!")
            return ThreadedCamera(cap)

    if allow_mock:
        print("[INFO] Falling back to Synthetic Camera Mode.")
        return MockVideoCapture(width=640, height=480, fps=30)
        
    raise RuntimeError("[ERROR] No working physical camera device found.")


def compute_fps(prev_time: float) -> tuple[float, float]:
    """Compute frames-per-second given timestamp of previous frame."""
    current_time = time.time()
    elapsed = current_time - prev_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    return fps, current_time


def draw_fps(frame, fps: float) -> None:
    """Overlay FPS counter in top-left corner of frame."""
    label = f"FPS: {fps:.1f}"
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(frame, (8, 8), (18 + text_w, 18 + text_h), (0, 0, 0), -1)
    cv2.putText(frame, label, (13, 13 + text_h), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)


def run_capture_loop(camera_index: int = 0) -> None:
    """Run camera test window."""
    cap = open_camera(camera_index)
    prev_time = time.time()
    print("[INFO] Press 'q' to quit window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        fps, prev_time = compute_fps(prev_time)
        draw_fps(frame, fps)

        cv2.imshow("Sign Language Translator — Camera Test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_capture_loop()
