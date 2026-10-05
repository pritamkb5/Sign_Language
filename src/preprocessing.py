"""
preprocessing.py — Module 2.1: Landmark Preprocessing, Normalization & Augmentation
===================================================================================
This module provides standardized methods to center, scale, pad, and augment
raw landmark lists before passing them to static or dynamic classifiers.

PHASE 2 ENHANCEMENTS:
- Robust occlusion handling & NaN/Inf sanitization for partial edge hand presence.
- Consistent scale-invariant normalization across live inference and training.
- Supplementary geometric feature extraction (finger flex angles & distances).
- Data augmentation with 3D rotation, scaling jitter, and joint micro-noise.
"""

import math
import random
import numpy as np


class SequenceSample:
    """
    Data structure representing dynamic/word-level gesture data sequences.
    
    Attributes:
        sequence_length: Number of frames in sequence (e.g. 30).
        num_features: Landmark feature dimension per frame (e.g. 126).
        sequence_data: numpy array of shape (sequence_length, num_features).
        label: Target classification label.
    """
    def __init__(self, sequence_length: int = 30, num_features: int = 126):
        self.sequence_length = sequence_length
        self.num_features = num_features
        # Initialize empty zero buffer
        self.sequence_data = np.zeros((self.sequence_length, self.num_features), dtype=np.float32)
        self.label = ""

    def load_from_frames(self, frames_list: list[np.ndarray], label: str) -> None:
        """
        Populates sequence data from a list of landmark feature arrays.
        Truncates or zero-pads to self.sequence_length as necessary.
        """
        self.label = label
        for i in range(self.sequence_length):
            if i < len(frames_list):
                vec = np.nan_to_num(np.array(frames_list[i], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
                if len(vec) < self.num_features:
                    padded = np.zeros(self.num_features, dtype=np.float32)
                    padded[:len(vec)] = vec
                    self.sequence_data[i] = padded
                else:
                    self.sequence_data[i] = vec[:self.num_features]
            else:
                self.sequence_data[i] = np.zeros(self.num_features, dtype=np.float32)


def sanitize_coords(coords: list[tuple]) -> list[tuple]:
    """
    Sanitize (x, y, z) coordinate tuples to prevent NaN, Inf, or extreme out-of-frame values.
    """
    sanitized = []
    for pt in coords:
        if pt is None or len(pt) < 3:
            sanitized.append((0.0, 0.0, 0.0))
            continue
        x, y, z = pt[0], pt[1], pt[2]
        x = 0.0 if (math.isnan(x) or math.isinf(x)) else float(np.clip(x, -5.0, 5.0))
        y = 0.0 if (math.isnan(y) or math.isinf(y)) else float(np.clip(y, -5.0, 5.0))
        z = 0.0 if (math.isnan(z) or math.isinf(z)) else float(np.clip(z, -5.0, 5.0))
        sanitized.append((x, y, z))
    return sanitized


def compute_angle(p1: tuple, p2: tuple, p3: tuple) -> float:
    """
    Calculate the joint flex angle (in degrees) between three 3D landmark points (p1-p2-p3).
    p2 is the joint vertex.
    """
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]], dtype=np.float32)
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1], p3[2] - p2[2]], dtype=np.float32)
    
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return float(np.degrees(angle_rad))


def compute_hand_geometric_features(coords: list[tuple]) -> list[float]:
    """
    Computes 15 additional hand geometric features (5 finger joint flex angles + 10 key landmark pair distances).
    Assumes `coords` is a list of 21 (x, y, z) tuples centered and scaled.
    """
    if len(coords) < 21:
        return [0.0] * 15

    # Finger joint flex angles (in degrees, normalized to [0, 1])
    # Thumb: 1-2-4
    # Index: 5-6-8
    # Middle: 9-10-12
    # Ring: 13-14-16
    # Pinky: 17-18-20
    angles = [
        compute_angle(coords[1], coords[2], coords[4]) / 180.0,
        compute_angle(coords[5], coords[6], coords[8]) / 180.0,
        compute_angle(coords[9], coords[10], coords[12]) / 180.0,
        compute_angle(coords[13], coords[14], coords[16]) / 180.0,
        compute_angle(coords[17], coords[18], coords[20]) / 180.0,
    ]

    # Fingertip pair distances (Euclidean distance between finger tips)
    # Tips: Thumb(4), Index(8), Middle(12), Ring(16), Pinky(20)
    tips = [4, 8, 12, 16, 20]
    distances = []
    for i in range(len(tips)):
        for j in range(i + 1, len(tips)):
            p1 = coords[tips[i]]
            p2 = coords[tips[j]]
            dist = math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)
            distances.append(dist)

    return angles + distances  # 5 + 10 = 15 features


def normalize_coords_tuples(coords: list[tuple]) -> list[float]:
    """
    Center and scale-normalize 21 raw (x, y, z) tuples.
    Uses wrist-origin (landmark 0) and middle-MCP (landmark 9) distance scaling.
    Handles partial occlusion and NaN values safely.
    """
    coords = sanitize_coords(coords)
    if not coords or len(coords) < 21:
        return [0.0] * 63

    wrist_x, wrist_y, wrist_z = coords[0]
    mcp_x, mcp_y, mcp_z = coords[9]

    scale_dist = math.sqrt(
        (mcp_x - wrist_x) ** 2 +
        (mcp_y - wrist_y) ** 2 +
        (mcp_z - wrist_z) ** 2
    )
    if scale_dist < 1e-4:
        scale_dist = 1.0

    normalized_coords = []
    for x, y, z in coords:
        normalized_coords.extend([
            (x - wrist_x) / scale_dist,
            (y - wrist_y) / scale_dist,
            (z - wrist_z) / scale_dist,
        ])
    return normalized_coords


def normalize_flat_vector(flat_vector: list[float]) -> list[float]:
    """
    Apply wrist-centering and scale normalization to each hand block in a
    126-dimensional feature vector. Safe to call on already-normalized data.
    """
    features = list(flat_vector)
    if len(features) < 126:
        features.extend([0.0] * (126 - len(features)))
    elif len(features) > 126:
        features = features[:126]

    # Clean NaN/Inf
    features = [0.0 if (math.isnan(v) or math.isinf(v)) else v for v in features]

    result = []
    for hand_idx in range(2):
        start = hand_idx * 63
        hand_slice = features[start:start + 63]

        if all(v == 0.0 for v in hand_slice):
            result.extend([0.0] * 63)
            continue

        coords = [
            (hand_slice[i * 3], hand_slice[i * 3 + 1], hand_slice[i * 3 + 2])
            for i in range(21)
        ]
        result.extend(normalize_coords_tuples(coords))

    return result


def normalize_hand_landmarks(landmarks) -> list[float]:
    """
    Center hand coordinates relative to wrist (landmark 0) and scale-normalize 
    by Euclidean distance to middle finger MCP (landmark 9).

    Args:
        landmarks: List of 21 landmark objects (with .x, .y, .z attributes).

    Returns:
        List of 63 floats (21 points * 3 coordinates) centered and scaled.
    """
    if not landmarks or len(landmarks) < 21:
        return [0.0] * 63

    raw_coords = []
    for lm in landmarks:
        raw_coords.append((getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)))
    
    return normalize_coords_tuples(raw_coords)


def prepare_input_vector(landmarks_list) -> list[float]:
    """
    Constructs a 126-dimensional flat list supporting up to 2 hands.
    If only one hand is detected, the second hand is padded with zeros.

    Args:
        landmarks_list: List of hands (each a list of 21 landmark objects).

    Returns:
        A list of 126 floats representing centered and scaled landmarks for both hands.
    """
    features = []

    # Process first hand if available
    if len(landmarks_list) > 0 and landmarks_list[0] is not None:
        features.extend(normalize_hand_landmarks(landmarks_list[0]))
    else:
        features.extend([0.0] * 63)

    # Process second hand if available
    if len(landmarks_list) > 1 and landmarks_list[1] is not None:
        features.extend(normalize_hand_landmarks(landmarks_list[1]))
    else:
        features.extend([0.0] * 63)

    return features

def prepare_raw_vector(landmarks_list) -> list[float]:
    """
    Extracts 126-dimensional raw landmarks (x, y, z) for 2 hands.
    Preserves absolute position to allow computing directional velocity.
    """
    features = []
    
    for hand_idx in range(2):
        if hand_idx < len(landmarks_list) and landmarks_list[hand_idx] is not None:
            for lm in landmarks_list[hand_idx]:
                features.extend([getattr(lm, 'x', 0.0), getattr(lm, 'y', 0.0), getattr(lm, 'z', 0.0)])
        else:
            features.extend([0.0] * 63)
            
    return features



def augment_landmarks(flat_vector: list[float], rotation_deg: float = 15.0, jitter_std: float = 0.02) -> list[float]:
    """
    Applies random 3D rotation, coordinate jittering, and scaling
    directly to a 126-dimensional normalized landmarks vector for data augmentation.

    Args:
        flat_vector: List of 126 floats.
        rotation_deg: Maximum rotation angle in degrees.
        jitter_std: Standard deviation of Gaussian coordinate noise.
    """
    features = np.array(flat_vector, dtype=np.float32)
    
    for hand_idx in range(2):
        start = hand_idx * 63
        end = start + 63
        hand_slice = features[start:end]
        
        if np.all(hand_slice == 0.0):
            continue

        coords = hand_slice.reshape((21, 3))

        # 1. Random XY plane rotation
        if rotation_deg > 0:
            angle = math.radians(random.uniform(-rotation_deg, rotation_deg))
            cos_a, sin_a = math.cos(angle), math.sin(angle)
            for i in range(21):
                x_val, y_val = coords[i, 0], coords[i, 1]
                coords[i, 0] = x_val * cos_a - y_val * sin_a
                coords[i, 1] = x_val * sin_a + y_val * cos_a

        # 2. Random 3D scale jittering (85% to 115%)
        scale_factor = random.uniform(0.85, 1.15)
        coords *= scale_factor

        # 3. Coordinate Gaussian Jittering
        if jitter_std > 0:
            noise = np.random.normal(0, jitter_std, size=(21, 3))
            coords += noise

        # 4. Horizontal Mirroring (50% probability)
        if random.random() < 0.5:
            coords[:, 0] = -coords[:, 0]

        features[start:end] = coords.flatten()

    return features.tolist()

