import cv2
import numpy as np

def preprocess_image(image, target_size=(28, 28), invert=True):
    """
    Preprocess a grayscale image for the HWR model.
    EMNIST is natively white strokes on a black background.
    If 'invert' is True, the input image (assumed black strokes on white)
    will be inverted to match EMNIST.
    """
    # Resize to target shape
    resized = cv2.resize(image, target_size)
    
    # Invert if needed (e.g., smartboard provides black text on white background)
    if invert:
        resized = 255 - resized
        
    # Normalize to 0.0 - 1.0
    normalized = resized.astype('float32') / 255.0
    return normalized

def prepare_for_inference(image, invert=True):
    """
    Prepares a single grayscale image for inference by preprocessing
    and adding batch and channel dimensions.
    """
    processed = preprocess_image(image, invert=invert)
    # Add batch and channel dimension (1, 28, 28, 1)
    input_tensor = np.expand_dims(np.expand_dims(processed, axis=0), axis=-1)
    return input_tensor
