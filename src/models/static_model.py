"""
static_model.py — Track A: Static Gesture Classifier Model v2
================================================================
Responsibility: Defines the upgraded PyTorch StaticModelV2 architecture for hand
gesture recognition with residual connections, LayerNorm, and customizable hyper-parameters.

Features:
- Input vector size: 126 features (2 hands x 21 landmarks x 3 coords).
- Architecture: Dense Residual Blocks + LayerNorm + Dropout.
- Predictor wrapper class with confidence score output.
"""

import os
import torch
import torch.nn as nn
import numpy as np


class ResidualBlock(nn.Module):
    """Residual dense block with LayerNorm and Dropout."""
    def __init__(self, dim: int, dropout_prob: float = 0.25):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.norm2(out)
        out = self.relu(out + residual)
        return out


class StaticModelV2(nn.Module):
    """
    Upgraded PyTorch Static Classifier v2.
    Uses input projection -> multiple Residual Blocks -> Classification Head.
    """
    def __init__(self, input_dim: int = 126, num_classes: int = 24, hidden_dim: int = 256, dropout_prob: float = 0.25):
        super(StaticModelV2, self).__init__()
        
        # Initial projection
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Residual backbone
        self.res_block1 = ResidualBlock(hidden_dim, dropout_prob)
        self.res_block2 = ResidualBlock(hidden_dim, dropout_prob)
        
        # Sub-projection layer
        self.mid_layer = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout_prob)
        )
        
        # Final classification head
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.mid_layer(x)
        logits = self.head(x)
        return logits


class StaticClassifierV2:
    """Wrapper class used during runtime to predict static hand signs using Model v2."""
    
    def __init__(self, model_path: str, class_labels: list[str], input_dim: int = 126):
        self.device = torch.device("cpu")
        self.class_labels = class_labels
        self.input_dim = input_dim
        
        self.model = StaticModelV2(input_dim=self.input_dim, num_classes=len(self.class_labels))
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                
                # Check for label mismatch
                out_features = self.model.head.out_features
                if out_features != len(self.class_labels):
                    raise ValueError(f"Static Model output layer ({out_features}) does not match length of class_labels.txt ({len(self.class_labels)}). Please retrain.")
                else:
                    print(f"[INFO] Static Model v2 loaded successfully from: {model_path}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                print(f"[ERROR] Failed to load Model v2 weights from {model_path}: {e}")
        else:
            print(f"[WARNING] Model weights file not found at: {model_path}. Initialized with random weights.")
            
        self.model.to(self.device)
        self.model.eval()

    def predict_with_confidence(self, flat_features: list[float]) -> tuple[str, float]:
        """
        Runs forward pass and returns (predicted_class_label, confidence_score).
        """
        if not flat_features or np.all(np.array(flat_features) == 0.0):
            return "", 0.0

        tensor_input = torch.tensor(flat_features, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor_input)
            probabilities = torch.softmax(logits, dim=1)
            
            max_prob, max_idx = torch.max(probabilities, dim=1)
            
            conf = float(max_prob.item())
            predicted_idx = int(max_idx.item())
            
            if predicted_idx < len(self.class_labels):
                return self.class_labels[predicted_idx], conf
            return "Unknown", conf

    def predict(self, flat_features: list[float]) -> str:
        label, _ = self.predict_with_confidence(flat_features)
        return label
