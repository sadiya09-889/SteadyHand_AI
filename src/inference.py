import argparse
import os
import json
import cv2
import numpy as np
from tensorflow.keras.models import load_model

from preprocessing import prepare_for_inference

def load_label_map(filepath='models/label_map.json'):
    if not os.path.exists(filepath):
        print(f"Error: Label map not found at {filepath}")
        return None
    with open(filepath, 'r') as f:
        # JSON keys are always strings, convert back to integer keys
        mapping = json.load(f)
        return {int(k): v for k, v in mapping.items()}

def predict_image(image_path, model_path='models/hwr_cnn_model.keras', label_map_path='models/label_map.json', invert=True):
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
        
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
        
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    label_map = load_label_map(label_map_path)
    if not label_map:
        return
        
    print(f"Reading image {image_path}...")
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error: Could not read image. Is it a valid image file?")
        return
        
    input_tensor = prepare_for_inference(img, invert=invert)
    
    print("Running inference...")
    predictions = model.predict(input_tensor, verbose=0)
    
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]
    predicted_char = label_map[predicted_class]
    
    print(f"\n--- Prediction Results ---")
    print(f"Predicted Character: '{predicted_char}'")
    print(f"Confidence: {confidence:.2%}")
    
    if confidence < 0.5:
        print("Note: Confidence is low. The prediction might be incorrect.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inference on a single image")
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model-path', type=str, default='models/hwr_cnn_model.keras', help='Path to trained model')
    parser.add_argument('--no-invert', action='store_true', help='Do not invert image colors (use if image is already white text on black)')
    args = parser.parse_args()
    
    predict_image(args.image, model_path=args.model_path, invert=not args.no_invert)
