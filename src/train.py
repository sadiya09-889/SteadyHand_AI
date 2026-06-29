import argparse
import os
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

from dataset import load_dataset, get_label_map, save_label_map
from model import build_cnn_model

def train(epochs=20, batch_size=128, subset=None, model_path='models/hwr_cnn_model.keras'):
    print("--- Setting up Training Pipeline ---")
    
    # 1. Load Data
    (X_train, y_train), (X_test, y_test) = load_dataset(subset=subset)
    
    # 2. Save Label Map
    label_map = get_label_map()
    save_label_map(label_map, 'models/label_map.json')
    
    # 3. Build Model
    model = build_cnn_model(input_shape=X_train.shape[1:], num_classes=y_train.shape[1])
    model.summary()
    
    # 4. Callbacks
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    checkpoint = ModelCheckpoint(model_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-6, verbose=1)
    
    # 5. Train
    print("\n--- Starting Training ---")
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_test, y_test),
        callbacks=[checkpoint, early_stop, reduce_lr]
    )
    
    # 6. Save Training History Plot
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.legend()
    plt.title('Accuracy')
    
    plt.savefig('reports/training_history.png')
    print("Saved training history plot to reports/training_history.png")
    print(f"Best model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train HWR Model on EMNIST")
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--subset', type=int, default=None, help='Subset of data to use for testing/debugging')
    parser.add_argument('--model-path', type=str, default='models/hwr_cnn_model.keras', help='Output model path')
    args = parser.parse_args()
    
    train(epochs=args.epochs, batch_size=args.batch_size, subset=args.subset, model_path=args.model_path)
