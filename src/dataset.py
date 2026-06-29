import os
import json
import numpy as np
from tensorflow.keras.utils import to_categorical

def get_label_map():
    """
    Returns the label mapping for EMNIST ByClass dataset (62 classes).
    0-9: Digits
    10-35: Uppercase letters
    36-61: Lowercase letters
    """
    labels = {}
    for i in range(10):
        labels[i] = str(i)
    for i in range(26):
        labels[i + 10] = chr(ord('A') + i)
    for i in range(26):
        labels[i + 36] = chr(ord('a') + i)
    return labels

def save_label_map(label_map, filepath='models/label_map.json'):
    """Saves the label mapping to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(label_map, f, indent=4)
    print(f"Saved label map to {filepath}")

def load_dataset(subset=None):
    """
    Loads the EMNIST ByClass dataset.
    Returns:
        (X_train, y_train), (X_test, y_test)
    """
    print("Loading EMNIST ByClass dataset from local Kaggle extract...")
    import struct
    def parse_idx(fn):
        with open(fn, 'rb') as f:
            d = f.read()
        magic, = struct.unpack('>I', d[:4])
        if magic == 2049: return np.frombuffer(d, dtype=np.uint8, offset=8)
        if magic == 2051: return np.frombuffer(d, dtype=np.uint8, offset=16).reshape(struct.unpack('>I', d[4:8])[0], 28, 28)
        raise ValueError("Invalid magic number")
        
    base_path = os.path.join("data", "emnist_source_files")
    X_train_raw = parse_idx(os.path.join(base_path, 'emnist-byclass-train-images-idx3-ubyte'))
    y_train_raw = parse_idx(os.path.join(base_path, 'emnist-byclass-train-labels-idx1-ubyte'))
    X_test_raw = parse_idx(os.path.join(base_path, 'emnist-byclass-test-images-idx3-ubyte'))
    y_test_raw = parse_idx(os.path.join(base_path, 'emnist-byclass-test-labels-idx1-ubyte'))
    if subset:
        print(f"Using a subset of {subset} samples for training and {subset//5} for testing...")
        X_train_raw = X_train_raw[:subset]
        y_train_raw = y_train_raw[:subset]
        X_test_raw = X_test_raw[:subset//5]
        y_test_raw = y_test_raw[:subset//5]

    # EMNIST images in binary format are known to be column-major (transposed).
    # We fix the orientation by transposing spatial dimensions.
    X_train_raw = np.transpose(X_train_raw, (0, 2, 1))
    X_test_raw = np.transpose(X_test_raw, (0, 2, 1))
    
    # Normalize pixel values to 0.0 - 1.0
    X_train = X_train_raw.astype('float32') / 255.0
    X_test = X_test_raw.astype('float32') / 255.0
    
    # Add channel dimension (N, 28, 28, 1)
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    
    # One-hot encode labels
    num_classes = 62
    y_train = to_categorical(y_train_raw, num_classes)
    y_test = to_categorical(y_test_raw, num_classes)
    
    print(f"Train data shape: {X_train.shape}, labels: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}, labels: {y_test.shape}")
    
    return (X_train, y_train), (X_test, y_test)
