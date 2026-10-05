"""
dynamic_model.py — Track B: Dynamic (Word-Level) Gesture Classifier Model v1
=============================================================================
Responsibility: Defines the PyTorch GRU sequence model structure for word-level sign language recognition over landmark frames.

Features:
- Input sequence: (batch_size, 30 frames, 126 features per frame).
- Architecture: 2-layer Gated Recurrent Unit (GRU) + LayerNorm + Dropout + Linear Head.
- Motion energy gate: Filters out stationary/static hands and empty sequences to prevent false positive word predictions.
"""

import os
import torch
import torch.nn as nn
import numpy as np


def normalize_trajectory(seq: np.ndarray) -> np.ndarray:
    """
    Normalizes landmarks by subtracting wrist position and scaling by wrist-MCP distance per hand.
    Then computes velocity (delta between frame t and t-1) and concatenates it.
    Input seq: shape (frames, 126) raw coordinates.
    Returns: shape (frames, 252) normalized coords + velocities.
    """
    if seq.size == 0:
        return seq
    norm_seq = seq.copy()
    
    # 1. Compute velocity from RAW sequence (to capture absolute movement)
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    
    # 2. Normalize coordinates and velocities
    for t in range(norm_seq.shape[0]):
        # Right hand (features 0-62)
        rx, ry, rz = norm_seq[t, 0], norm_seq[t, 1], norm_seq[t, 2]
        mx, my, mz = norm_seq[t, 9], norm_seq[t, 10], norm_seq[t, 11]
        scale = np.sqrt((mx-rx)**2 + (my-ry)**2 + (mz-rz)**2)
        if scale < 1e-4: scale = 1.0
        
        if rx != 0 or ry != 0 or rz != 0:
            for i in range(0, 63, 3):
                if norm_seq[t, i] != 0:
                    norm_seq[t, i]   = (norm_seq[t, i] - rx) / scale
                    norm_seq[t, i+1] = (norm_seq[t, i+1] - ry) / scale
                    norm_seq[t, i+2] = (norm_seq[t, i+2] - rz) / scale
            vel[t, 0:63] /= scale

        # Left hand (features 63-125)
        if norm_seq.shape[1] >= 126:
            lx, ly, lz = norm_seq[t, 63], norm_seq[t, 64], norm_seq[t, 65]
            mx, my, mz = norm_seq[t, 72], norm_seq[t, 73], norm_seq[t, 74]
            scale_l = np.sqrt((mx-lx)**2 + (my-ly)**2 + (mz-lz)**2)
            if scale_l < 1e-4: scale_l = 1.0
            
            if lx != 0 or ly != 0 or lz != 0:
                for i in range(63, 126, 3):
                    if norm_seq[t, i] != 0:
                        norm_seq[t, i]   = (norm_seq[t, i] - lx) / scale_l
                        norm_seq[t, i+1] = (norm_seq[t, i+1] - ly) / scale_l
                        norm_seq[t, i+2] = (norm_seq[t, i+2] - lz) / scale_l
                vel[t, 63:126] /= scale_l

    return np.concatenate([norm_seq, vel], axis=-1)

class SpatialDropout1D(nn.Module):
    """Spatial dropout for sequences: drops entire feature channels across all time steps."""
    def __init__(self, p: float):
        super(SpatialDropout1D, self).__init__()
        self.dropout = nn.Dropout2d(p)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        # Dropout2d expects (batch, channels, height, width) or (batch, features, seq_len)
        x = x.permute(0, 2, 1) # shape: (batch, features, seq_len)
        x = self.dropout(x)
        x = x.permute(0, 2, 1) # back to (batch, seq_len, features)
        return x

class DynamicGRU(nn.Module):
    """
    Sequence classifier using Bidirectional GRU/LSTM over landmark coordinate vectors.
    """
    def __init__(self, input_dim: int = 252, hidden_dim: int = 128, num_layers: int = 2, num_classes: int = 15, dropout_prob: float = 0.3):
        super(DynamicGRU, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.spatial_dropout = SpatialDropout1D(dropout_prob)
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_prob if num_layers > 1 else 0.0,
            bidirectional=True
        )
        
        gru_out_dim = hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Linear(gru_out_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.spatial_dropout(x)
        gru_out, _ = self.gru(x)
        final_step = gru_out[:, -1, :]
        logits = self.fc(final_step)
        return logits


class DynamicClassifier:
    """Wrapper class used during runtime to predict dynamic word signs."""
    
    def __init__(self, model_path: str, class_labels: list[str], input_dim: int = 252, sequence_length: int = 16):
        self.device = torch.device("cpu")
        self.class_labels = class_labels
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        
        self.model = DynamicGRU(
            input_dim=self.input_dim,
            hidden_dim=128,
            num_layers=2,
            num_classes=len(self.class_labels),
            dropout_prob=0.3
        )
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                # Validation check for label mismatch
                out_features = self.model.fc[-1].out_features
                if out_features != len(self.class_labels):
                    raise ValueError(f"Dynamic Model output layer ({out_features}) does not match length of dynamic_labels.txt ({len(self.class_labels)}). Please retrain.")
                else:
                    print(f"[INFO] Loaded trained Dynamic GRU classifier from: {model_path}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Failed to load Dynamic GRU weights from {model_path}: {e}")
        else:
            print(f"[WARNING] Dynamic model file not found at: {model_path}. Predictor will run with random weights.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict_with_confidence(self, sequence_arr: np.ndarray, motion_threshold: float = 0.004) -> tuple[str, float]:
        """
        Runs model forward pass on a sequence of shape (1, sequence_length, input_dim).
        Filters out low-motion stationary frames before classifying.
        """
        if sequence_arr is None or sequence_arr.size == 0:
            return "", 0.0

        if len(sequence_arr.shape) == 2:
            sequence_arr = np.expand_dims(sequence_arr, axis=0)

        # 1. Motion Energy Gate: mean per-frame per-feature displacement
        # We calculate motion BEFORE normalization, otherwise stationary hand in space seems to have 0 motion
        diffs = np.diff(sequence_arr[0], axis=0)  # shape: (seq_len-1, features)
        mean_motion = float(np.mean(np.abs(diffs)))
        
        # Entropy check to filter erratic or garbage motion
        entropy = float(np.std(diffs))
        if entropy > 0.05:
            return "", 0.0
        
        # If hand is static/stationary or mostly empty, do not force a dynamic word prediction
        if mean_motion < motion_threshold:
            return "", 0.0

        # Trajectory normalization relative to palm center (wrist) + velocity calculation
        norm_seq = normalize_trajectory(sequence_arr[0])
        norm_seq = np.expand_dims(norm_seq, axis=0)

        tensor_input = torch.tensor(norm_seq, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_input)
            probabilities = torch.softmax(logits, dim=1)
            
            max_prob, max_idx = torch.max(probabilities, dim=1)
            
            conf = float(max_prob.item())
            predicted_idx = int(max_idx.item())
            
            # Strict thresholding to eliminate false positives on resting motion
            if conf < 0.40:
                return "", conf

            if predicted_idx < len(self.class_labels):
                return self.class_labels[predicted_idx], conf
            return "Unknown", conf

    def predict(self, sequence_arr: np.ndarray) -> str:
        label, _ = self.predict_with_confidence(sequence_arr)
        return label
