import argparse
import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from dataset import load_dataset, get_label_map

def evaluate_model(model_path='models/hwr_cnn_model.keras'):
    print(f"Loading model from {model_path}...")
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    model = load_model(model_path)
    
    print("Loading test dataset...")
    # Load dataset
    _, (X_test, y_test) = load_dataset(subset=None)
    
    print("Evaluating model on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    
    print(f"\nTest Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Save metrics
    metrics = {
        'test_loss': float(loss),
        'test_accuracy': float(accuracy)
    }
    os.makedirs('reports', exist_ok=True)
    with open('reports/metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Generating predictions...")
    y_pred_prob = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_prob, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    label_map = get_label_map()
    labels = [label_map[i] for i in range(len(label_map))]
    
    print("Generating classification report...")
    report = classification_report(y_true, y_pred, target_names=labels, output_dict=False)
    with open('reports/classification_report.txt', 'w') as f:
        f.write(report)
    print("Classification report saved to reports/classification_report.txt")
        
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(24, 24))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('reports/confusion_matrix.png')
    print("Saved confusion matrix to reports/confusion_matrix.png")
    
    print("\nNote: EMNIST accuracy is highly meaningful compared to synthetic datasets.")
    print("Scores around 85-90% are very good for the 62-class ByClass dataset due to inherently ambiguous characters (e.g., 'c' vs 'C').")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate HWR Model")
    parser.add_argument('--model-path', type=str, default='models/hwr_cnn_model.keras', help='Model path to evaluate')
    args = parser.parse_args()
    evaluate_model(args.model_path)
