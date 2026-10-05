"""
train_dynamic.py — Track B: PyTorch Dynamic GRU Classifier Training Loop
========================================================================
Responsibility: Loads variable-length sequence dataset (.npy files), resamples
to fixed 30-frame length via temporal interpolation, applies data augmentation,
splits into train/val/test, trains DynamicGRU model with class-weighted loss,
evaluates sequence accuracy & top word confusions, and exports trained weights
and class label mapping.

Usage:
  python src/train_dynamic.py --data-dir data/dynamic --epochs 60
"""

import argparse
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from collections import Counter

from models.dynamic_model import DynamicGRU, normalize_trajectory

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ── Temporal Resampling ──────────────────────────────────────────────────────

def resample_sequence(sequence: np.ndarray, target_length: int = 16) -> np.ndarray:
    """
    Resample a variable-length sequence to a fixed target_length using
    linear interpolation along the time axis.

    This ensures the model sees the *entire* gesture motion spread across
    exactly target_length frames — no trailing zeros, no truncation.

    Args:
        sequence: numpy array of shape (num_frames, num_features).
        target_length: desired output frame count.

    Returns:
        Resampled array of shape (target_length, num_features).
    """
    num_frames, num_features = sequence.shape

    if num_frames == target_length:
        return sequence.copy()

    if num_frames == 0:
        return np.zeros((target_length, num_features), dtype=np.float32)

    if num_frames == 1:
        return np.tile(sequence[0], (target_length, 1)).astype(np.float32)

    # Linearly interpolate each feature dimension independently
    old_indices = np.linspace(0, num_frames - 1, num_frames)
    new_indices = np.linspace(0, num_frames - 1, target_length)

    resampled = np.zeros((target_length, num_features), dtype=np.float32)
    for feat_idx in range(num_features):
        resampled[:, feat_idx] = np.interp(new_indices, old_indices, sequence[:, feat_idx])

    return resampled


# ── Data Augmentation ────────────────────────────────────────────────────────

def augment_sequence(sequence: np.ndarray, target_length: int = 16) -> np.ndarray:
    """
    Apply temporal and spatial augmentation to a raw variable-length sequence,
    then resample to target_length.

    Augmentations applied:
      1. Temporal speed variation: stretch/compress raw frames by ±25%.
      2. Random frame dropout: drop up to 15% of frames.
      3. Coordinate Gaussian noise (σ=0.01).
      4. Horizontal mirroring (50% probability) — flips x-coordinates.
      5. Small scale jitter (90%–110%).

    Args:
        sequence: numpy array of shape (num_frames, num_features).
        target_length: desired output length after resampling.

    Returns:
        Augmented + resampled array of shape (target_length, num_features).
    """
    num_frames, num_features = sequence.shape

    if num_frames < 2:
        return resample_sequence(sequence, target_length)

    # 1. Temporal speed variation: resample to a random intermediate length
    speed_factor = np.random.uniform(0.75, 1.25)
    intermediate_len = max(3, int(num_frames * speed_factor))
    seq = resample_sequence(sequence, intermediate_len)

    # 2. Random frame dropout (drop up to 15% of frames, keep at least 3)
    if seq.shape[0] > 4:
        num_drop = max(0, int(seq.shape[0] * np.random.uniform(0.0, 0.15)))
        if num_drop > 0 and seq.shape[0] - num_drop >= 3:
            keep_indices = sorted(np.random.choice(seq.shape[0], seq.shape[0] - num_drop, replace=False))
            seq = seq[keep_indices]

    # 3. Coordinate Gaussian noise
    noise = np.random.normal(0, 0.01, size=seq.shape).astype(np.float32)
    seq = seq + noise

    # 4. Horizontal mirroring (50% chance) — flip x-coordinates
    #    Landmarks are stored as (x, y, z) triplets. x is at indices 0, 3, 6, ...
    if np.random.random() < 0.5:
        for i in range(0, num_features, 3):
            seq[:, i] = -seq[:, i]

    # 5. Scale jitter
    scale = np.random.uniform(0.90, 1.10)
    seq = seq * scale

    # Final resample to target_length
    return resample_sequence(seq, target_length)


# ── Dataset ──────────────────────────────────────────────────────────────────

class DynamicSequenceDataset(Dataset):
    """PyTorch dataset for fixed-length resampled landmark sequences."""
    def __init__(self, sequences: list[np.ndarray], labels: list[int], augment: bool = False, target_length: int = 16):
        self.raw_sequences = sequences  # variable-length raw arrays
        self.labels = labels
        self.augment = augment
        self.target_length = target_length

    def __len__(self):
        return len(self.raw_sequences)

    def __getitem__(self, idx):
        raw_seq = self.raw_sequences[idx]
        label = self.labels[idx]

        if self.augment:
            seq = augment_sequence(raw_seq, self.target_length)
        else:
            seq = resample_sequence(raw_seq, self.target_length)
            
        # Apply normalization relative to palm center for robust separation
        seq = normalize_trajectory(seq)

        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_dynamic_dataset(data_dirs: list[str], num_features: int = 126):
    """
    Loads all .npy sequence files from a list of data_dirs.
    Returns raw variable-length sequences (NOT padded/resampled yet).
    """
    npy_files = []
    for d in data_dirs:
        dir_path = Path(d)
        if dir_path.exists():
            npy_files.extend(list(dir_path.glob("*.npy")))
        else:
            alt_path = _PROJECT_ROOT / "src" / "data" / "dynamic"
            if alt_path.exists() and alt_path.name == dir_path.name:
                npy_files.extend(list(alt_path.glob("*.npy")))

    if not npy_files:
        raise FileNotFoundError(f"[ERROR] No .npy sequence files found in {data_dirs}")

    sequences = []
    labels_raw = []
    skipped = 0

    for fpath in npy_files:
        parts = fpath.stem.split("_")
        word_label = parts[2] if parts[0] == "aug" else parts[0]
        try:
            arr = np.load(fpath).astype(np.float32)

            # Ensure feature dimension matches
            if arr.ndim != 2:
                skipped += 1
                continue
            if arr.shape[1] != num_features:
                padded = np.zeros((arr.shape[0], num_features), dtype=np.float32)
                f_min = min(num_features, arr.shape[1])
                padded[:, :f_min] = arr[:, :f_min]
                arr = padded

            # Skip sequences that are too short (< 3 frames) or suspiciously long (> 80 frames)
            if arr.shape[0] < 3:
                print(f"[WARNING] Skipping too-short file {fpath.name} ({arr.shape[0]} frames)")
                skipped += 1
                continue
            if arr.shape[0] > 80:
                print(f"[WARNING] Trimming overly long file {fpath.name} ({arr.shape[0]} frames) to 80")
                arr = arr[:80]

            sequences.append(arr)
            labels_raw.append(word_label)
        except Exception as e:
            print(f"[WARNING] Skipping corrupt file {fpath.name}: {e}")
            skipped += 1

    if skipped:
        print(f"[INFO] Skipped {skipped} files during loading.")

    unique_words = sorted(list(set(labels_raw)))
    word_to_idx = {w: i for i, w in enumerate(unique_words)}
    labels = [word_to_idx[w] for w in labels_raw]

    return sequences, labels, unique_words


def compute_class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced loss."""
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        count = counts.get(c, 1)
        weights.append(total / (num_classes * count))
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    # Normalize so mean weight = 1.0
    weights_tensor = weights_tensor / weights_tensor.mean()
    return weights_tensor


# ── Training ─────────────────────────────────────────────────────────────────

def train_dynamic_model(data_dirs: list[str], model_save_path: str, labels_save_path: str,
                        epochs: int, batch_size: int, learning_rate: float) -> tuple[float, float]:
    """Trains DynamicGRU model on sequence dataset with resampling + augmentation."""
    print(f"[INFO] Loading dynamic sequence dataset from: {data_dirs}")
    sequences, labels, class_labels = load_dynamic_dataset(data_dirs)
    num_classes = len(class_labels)

    # Print data statistics
    frame_counts = [s.shape[0] for s in sequences]
    print(f"[INFO] Loaded {len(sequences)} sequence samples across {num_classes} dynamic words ({class_labels})")
    print(f"[INFO] Frame count stats: min={min(frame_counts)}, max={max(frame_counts)}, mean={np.mean(frame_counts):.1f}")
    
    label_counts = Counter(labels)
    for word, idx in sorted(zip(class_labels, range(num_classes))):
        print(f"  {word}: {label_counts[idx]} samples")

    # Stratified split: 70% Train, 15% Val, 15% Test
    # Falls back to a non-stratified split if any class has too few samples to
    # stratify (this happens with small, uneven real-data classes — e.g. a
    # word with only 1-3 total samples can't be split 3 ways while preserving
    # its class in every split).
    def _safe_split(X, y, train_size, stratify):
        try:
            return train_test_split(X, y, train_size=train_size, random_state=42, stratify=stratify)
        except ValueError as e:
            print(f"[WARN] Stratified split failed ({e}); falling back to a non-stratified split. "
                  f"Consider collecting more samples for your rarest word(s).")
            return train_test_split(X, y, train_size=train_size, random_state=42)

    X_train, X_temp, y_train, y_temp = _safe_split(sequences, labels, 0.70, stratify=labels)
    X_val, X_test, y_val, y_test = _safe_split(X_temp, y_temp, 0.50, stratify=y_temp)

    print(f"[INFO] Train samples: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # Create datasets — training set uses augmentation, val/test do not
    train_loader = DataLoader(
        DynamicSequenceDataset(X_train, y_train, augment=True, target_length=16),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        DynamicSequenceDataset(X_val, y_val, augment=False, target_length=16),
        batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        DynamicSequenceDataset(X_test, y_test, augment=False, target_length=16),
        batch_size=batch_size, shuffle=False
    )

    # Model initialization
    model = DynamicGRU(input_dim=252, hidden_dim=128, num_layers=2, num_classes=num_classes, dropout_prob=0.3)
    
    # Class-weighted loss to prevent majority-class collapse
    class_weights = compute_class_weights(y_train, num_classes)
    print(f"[INFO] Class weights: {class_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = 0.0
    patience = 20
    patience_counter = 0

    print("[INFO] Beginning Dynamic GRU training loop...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_loader.dataset)
        scheduler.step()

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        epoch_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:03d}/{epochs:03d} | Train Loss: {epoch_loss:.4f} | Val Loss: {epoch_val_loss:.4f} | Val Acc: {val_acc * 100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    print(f"\n[SUCCESS] Dynamic GRU Model v1 Training finished. Best Validation Accuracy: {best_val_acc * 100:.2f}%")
    print(f"[INFO] Saved best dynamic model weights to: {model_save_path}")

    # Export dynamic labels list
    os.makedirs(os.path.dirname(labels_save_path), exist_ok=True)
    with open(labels_save_path, "w") as f:
        f.write("\n".join(class_labels))
    print(f"[INFO] Saved dynamic class labels list to: {labels_save_path}")

    # Final Evaluation on Test Set & Error Analysis
    best_model = DynamicGRU(input_dim=252, hidden_dim=128, num_layers=2, num_classes=num_classes)
    best_model.load_state_dict(torch.load(model_save_path))
    best_model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = best_model(inputs)
            _, predicted = torch.max(outputs, dim=1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    test_acc = np.mean(all_preds == all_targets)

    print("\n" + "=" * 60)
    print(f" DYNAMIC GRU MODEL V1 TEST SET EVALUATION")
    print(f" Sequence Test Accuracy: {test_acc * 100:.2f}%")
    print("=" * 60)

    report = classification_report(all_targets, all_preds, target_names=class_labels, zero_division=0)
    print("\nDynamic Classification Report:\n", report)

    conf_mat = confusion_matrix(all_targets, all_preds)

    # Save metrics report
    results_dir = _PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = results_dir / "dynamic_v1_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write(f"Dynamic GRU Model v1 Test Accuracy: {test_acc * 100:.2f}%\n")
        f.write(f"Best Validation Accuracy: {best_val_acc * 100:.2f}%\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(np.array2string(conf_mat))
    print(f"[INFO] Dynamic evaluation metrics saved to: {metrics_file.resolve()}")

    return best_val_acc, test_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sign Language Dynamic GRU Model Trainer")
    parser.add_argument("--data-dir", type=str, default=str(_PROJECT_ROOT / "data" / "dynamic"), help="Path to sequence dataset directory")
    parser.add_argument("--model-out", type=str, default=str(_PROJECT_ROOT / "models" / "dynamic_v1.pt"), help="Path to output PyTorch model weights")
    parser.add_argument("--labels-out", type=str, default=str(_PROJECT_ROOT / "models" / "dynamic_labels.txt"), help="Path to output word labels list")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="DataLoader batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="AdamW learning rate")
    parser.add_argument("--augment-offline", action="store_true", help="Run offline data augmentation script before training")

    args = parser.parse_args()

    if args.augment_offline:
        import subprocess
        import sys
        print(f"[INFO] Running offline data augmentation before training...")
        script_path = _PROJECT_ROOT / "scripts" / "augment_dynamic_data.py"
        try:
            subprocess.run([sys.executable, str(script_path), "--input", args.data_dir, "--copies", "10"], check=True)
            print("[INFO] Offline augmentation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Offline augmentation failed: {e}")
            sys.exit(1)

    data_dirs = [
        args.data_dir,
        str(_PROJECT_ROOT / "data" / "dynamic_real")
    ]

    train_dynamic_model(
        data_dirs=data_dirs,
        model_save_path=args.model_out,
        labels_save_path=args.labels_out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
