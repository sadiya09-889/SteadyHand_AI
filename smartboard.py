import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from tensorflow import keras  # Assuming TensorFlow/Keras
from spellchecker import SpellChecker  # For spell checking
import os  # Import os to check for model file existence
import sys
import json

# Add src to path so we can import our preprocessing pipeline
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
try:
    from preprocessing import prepare_for_inference
except ImportError:
    print("Warning: Could not import preprocessing. Ensure 'src' folder exists.")

# --- Existing Global Variables ---
canvas_size = (600, 800)
brush_thickness = 2
smooth_factor = 21
drawing = False
all_strokes = []
current_stroke = []
last_stroke_time = 0  # To detect pauses for segmentation
segment_buffer = []  # To hold strokes for current segment (character/word)

# --- New Global Variables for AI Feature ---
current_tab = "drawing"  # "drawing" or "text_enhancement"
hwr_model = None  # Placeholder for your loaded HWR model
label_map = {}    # Placeholder for the loaded label mapping
spell = SpellChecker()  # Initialize spell checker


# --- Model Loading Function (Call this once at startup) ---
def load_hwr_model():
    """
    Loads the pre-trained Handwriting Recognition (HWR) CNN model.
    This model should be trained separately and saved as 'models/hwr_cnn_model.keras'.
    If the model fails to load, text enhancement features will be limited.
    """
    global hwr_model, label_map
    model_path = 'models/hwr_cnn_model.keras'  # Expected path of the trained model
    label_path = 'models/label_map.json'
    
    if not os.path.exists(model_path) or not os.path.exists(label_path):
        print(f"Warning: Model file or label map not found in 'models/'. Please run 'python src/train.py' first.")
        print("Text enhancement features will be disabled.")
        hwr_model = None
        return

    try:
        hwr_model = keras.models.load_model(model_path)
        with open(label_path, 'r') as f:
            mapping = json.load(f)
            label_map = {int(k): v for k, v in mapping.items()}
        print("HWR Model and label map loaded successfully.")
    except Exception as e:
        print(f"Error loading HWR model from '{model_path}': {e}. Text enhancement will be limited.")
        hwr_model = None  # Ensure it's None if loading fails


# --- Character Recognition Function ---
# This function's ALL_CHARS mapping must match the training script exactly.
def recognize_character(char_image):
    """
    Recognizes a single character from an input image using the loaded HWR model.

    Args:
        char_image (numpy.ndarray): A grayscale image of the handwritten character.
                                    Expected to be a NumPy array.

    Returns:
        str: The recognized character (e.g., 'A', '5', 'g'), or "uncertain" if confidence is low.
    """
    if hwr_model is None or not label_map:
        return ""  # Cannot recognize without model

    # Preprocess image using shared pipeline. 
    # Smartboard draws black on white, so invert=True matches EMNIST (white on black)
    try:
        input_tensor = prepare_for_inference(char_image, invert=True)
    except NameError:
        print("Error: preprocessing module not loaded.")
        return ""

    # Make a prediction
    # verbose=0 suppresses prediction progress bar, useful in interactive apps
    predictions = hwr_model.predict(input_tensor, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Return "uncertain" if confidence is below threshold
    if confidence < 0.5:
        return "uncertain"

    # Map the predicted class index back to a character
    if predicted_class in label_map:
        return label_map[predicted_class]
    else:
        return ""  # Should not happen if model is trained correctly, but good for safety


# --- Text Enhancement Logic (Called when a segment of strokes is complete) ---
def process_text_segment(strokes_in_segment):
    """
    Processes a segment of handwritten strokes to recognize characters,
    apply spell correction, and store the digital text.

    Args:
        strokes_in_segment (list): A list of NumPy arrays, where each array
                                   represents a single stroke's (x, y) points.
    """
    if not strokes_in_segment:
        return  # Do nothing if no strokes in segment

    # 1. Create a temporary canvas to draw the collected strokes for the segment
    segment_canvas = np.ones((canvas_size[0], canvas_size[1]), dtype=np.uint8) * 255  # White background

    # Initialize bounding box coordinates for the entire segment
    # Use float('-inf') and float('inf') for robust min/max initialization
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    # Draw all strokes within the segment onto the temporary canvas
    # and update the bounding box to encompass all strokes.
    for stroke in strokes_in_segment:
        if len(stroke) > 1:  # Ensure stroke has at least 2 points to draw a line
            for i in range(1, len(stroke)):
                cv2.line(segment_canvas, tuple(stroke[i - 1]), tuple(stroke[i]), (0,), brush_thickness)  # Draw in black

            # Update bounding box for the segment based on current stroke's points
            min_x = min(min_x, np.min(stroke[:, 0]))
            min_y = min(min_y, np.min(stroke[:, 1]))
            max_x = max(max_x, np.max(stroke[:, 0]))
            max_y = max(max_y, np.max(stroke[:, 1]))

    # If no valid strokes were drawn, exit
    if min_x == float('inf') or max_x == float('-inf'):
        return

    # Add a small padding to the bounding box to ensure the whole character is captured
    padding = 5
    min_x = max(0, int(min_x) - padding)
    min_y = max(0, int(min_y) - padding)
    max_x = min(canvas_size[1], int(max_x) + padding)
    max_y = min(canvas_size[0], int(max_y) + padding)

    recognized_text = ""

    # Extract the Region of Interest (ROI) from the temporary canvas
    if max_x > min_x and max_y > min_y:
        roi = segment_canvas[min_y:max_y, min_x:max_x]

        # 2. Recognize characters in the ROI
        # IMPORTANT SIMPLIFICATION: For this basic implementation, we are treating
        # the entire ROI (which might contain multiple characters or a word)
        # as a single character for recognition.
        # For a full-featured app, you would need advanced image processing
        # here to segment the ROI into individual characters (e.g., using contours)
        # before passing them to recognize_character().
        recognized_text = recognize_character(roi)

        # 3. Basic Spell Check on the recognized text
        # spell.correction is most effective on full words.
        # If 'recognized_text' is a single character, correction might not change it.
        corrected_word = spell.correction(recognized_text) if recognized_text else ""

        # If spell checker provides a correction, use it; otherwise, use the raw recognition
        final_text_to_display = corrected_word if corrected_word else recognized_text

        # 4. Store the corrected digital text
        # Store as a 'text' type item with its content and top-left position.
        if final_text_to_display:
            all_strokes.append({"type": "text", "content": final_text_to_display, "position": (min_x, min_y)})
        else:
            # If nothing was recognized or corrected, keep the original drawing strokes
            # This ensures that even unreadable scribbles are retained as drawing.
            for stroke in strokes_in_segment:
                all_strokes.append({"type": "drawing", "content": stroke})
    else:
        # If ROI is invalid (e.g., single click, no drawing), add original strokes as drawing
        for stroke in strokes_in_segment:
            all_strokes.append({"type": "drawing", "content": stroke})


# --- Modified draw_callback to handle tabs and stroke capture ---
def draw_callback(event, x, y, flags, param):
    """
    Callback function for mouse events on the smartboard canvas.
    Handles drawing in 'drawing' mode and stroke capture/segmentation
    in 'text_enhancement' mode.
    """
    global drawing, current_stroke, all_strokes, last_stroke_time, segment_buffer, current_tab

    # Logic for the "Drawing" tab (retains existing functionality)
    if current_tab == "drawing":
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_stroke = [[x, y]]
        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_stroke.append([x, y])
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(current_stroke) > 1:
                # Store as a 'drawing' type item
                all_strokes.append({"type": "drawing", "content": np.array(current_stroke)})
            current_stroke = []

    # Logic for the "Text Enhancement" tab (new AI feature)
    elif current_tab == "text_enhancement":
        current_time = cv2.getTickCount() / cv2.getTickFrequency()  # Get current time for pause detection

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            current_stroke = [[x, y]]
            # If starting a new stroke after a significant pause,
            # it indicates the end of the previous text segment.
            if segment_buffer and (current_time - last_stroke_time > 0.5):  # Pause detection threshold (0.5 seconds)
                process_text_segment(segment_buffer)
                segment_buffer = []  # Clear buffer for the new segment
            last_stroke_time = current_time  # Update time for the start of the new stroke

        elif event == cv2.EVENT_MOUSEMOVE and drawing:
            current_stroke.append([x, y])

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if len(current_stroke) > 1:
                segment_buffer.append(np.array(current_stroke))  # Add completed stroke to the segment buffer
                last_stroke_time = current_time  # Update time for the end of the stroke
            current_stroke = []

            # The actual processing of the segment (character/word) will be triggered
            # in the main loop based on a pause, allowing for multi-stroke characters/words.


# --- Main Application Loop ---
if __name__ == "__main__":
    cv2.namedWindow('Smartboard', cv2.WINDOW_NORMAL)  # Create a resizable window
    cv2.resizeWindow('Smartboard', canvas_size[1], canvas_size[0])  # Set initial window size
    cv2.setMouseCallback('Smartboard', draw_callback)  # Set the mouse callback function

    # Load the AI model when the application starts
    load_hwr_model()

    while True:
        # Create a blank white canvas for drawing
        canvas = np.ones((canvas_size[0], canvas_size[1], 3), dtype=np.uint8) * 255  # Use 3 channels for color

        # Render all stored strokes and recognized text
        for item in all_strokes:
            if item["type"] == "drawing":
                # Render raw drawing strokes (black lines)
                stroke = item["content"]
                for i in range(1, len(stroke)):
                    cv2.line(canvas, tuple(stroke[i - 1]), tuple(stroke[i]), (0, 0, 0), brush_thickness)  # Black
            elif item["type"] == "text":
                # Render recognized and corrected digital text (red color)
                text_content = item["content"]
                position = item["position"]
                # OpenCV uses BGR, so (0, 0, 255) is red
                cv2.putText(canvas, text_content, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

                # Render the currently ongoing stroke (if drawing)
        if len(current_stroke) > 1:
            for i in range(1, len(current_stroke)):
                cv2.line(canvas, tuple(current_stroke[i - 1]), tuple(current_stroke[i]), (0, 0, 0),
                         brush_thickness)  # Black

        # In "Text Enhancement" mode, render strokes currently in the segment buffer
        # These are strokes that are part of the current character/word being formed.
        if current_tab == "text_enhancement" and len(segment_buffer) > 0:
            for stroke in segment_buffer:
                for i in range(1, len(stroke)):
                    cv2.line(canvas, tuple(stroke[i - 1]), tuple(stroke[i]), (0, 0, 0), brush_thickness)  # Black

            # Trigger processing of the text segment if drawing has stopped for a while
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            if drawing == False and (current_time - last_stroke_time > 0.8) and len(
                    segment_buffer) > 0:  # 0.8 seconds pause
                process_text_segment(segment_buffer)
                segment_buffer = []  # Clear the buffer after processing

        # Display current tab indicator and instructions on the canvas
        cv2.putText(canvas, f"Tab: {current_tab.replace('_', ' ').title()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (100, 100, 100), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Press 'T' for Text, 'D' for Drawing", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (100, 100, 100), 1, cv2.LINE_AA)

        # Show the updated canvas
        cv2.imshow('Smartboard', canvas)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  # Clear canvas
            all_strokes = []
            current_stroke = []
            segment_buffer = []  # Clear segment buffer too
            print("Canvas cleared.")

        elif key == ord('t'):  # Switch to Text Enhancement Tab
            current_tab = "text_enhancement"
            print("Switched to Text Enhancement Tab")
            # Process any pending strokes from drawing tab before switching
            if len(current_stroke) > 1:
                all_strokes.append({"type": "drawing", "content": np.array(current_stroke)})
            current_stroke = []
            segment_buffer = []  # Clear any pending strokes from previous tab

        elif key == ord('d'):  # Switch to Drawing Tab
            current_tab = "drawing"
            print("Switched to Drawing Tab")
            # Process any pending text segment before switching tabs, to avoid losing input
            if len(segment_buffer) > 0:
                process_text_segment(segment_buffer)
                segment_buffer = []
            current_stroke = []

        elif key == ord('s'):  # Trigger smoothing visualization (for 'drawing' type strokes)
            # This part of the code is for visualizing smoothing on 'drawing' strokes.
            # It creates two temporary images: one for original drawing, one for smoothed.
            # It does not affect the 'text' type items which are already digital.

            original_img = np.ones_like(canvas) * 255
            smoothed_img = np.ones_like(canvas) * 255

            for item in all_strokes:
                if item["type"] == "drawing":
                    stroke = item["content"]
                    if len(stroke) < 7:  # Apply smoothing only to longer strokes
                        for i in range(1, len(stroke)):
                            cv2.line(original_img, tuple(stroke[i - 1]), tuple(stroke[i]), (0,), brush_thickness)
                            cv2.line(smoothed_img, tuple(stroke[i - 1]), tuple(stroke[i]), (0,), brush_thickness)
                    else:
                        pts = stroke
                        w = min(smooth_factor, len(pts) // 2 * 2 + 1)  # Ensure window size is odd and within bounds
                        if w >= 5:  # Only smooth if window size is reasonable
                            x_smooth = savgol_filter(pts[:, 0], w, polyorder=2)
                            y_smooth = savgol_filter(pts[:, 1], w, polyorder=2)
                            smoothed_pts = np.column_stack((x_smooth, y_smooth)).astype(int)

                            for i in range(1, len(pts)):
                                cv2.line(original_img, tuple(pts[i - 1]), tuple(pts[i]), (0,), brush_thickness)

                            for i in range(1, len(smoothed_pts)):
                                cv2.line(smoothed_img, tuple(smoothed_pts[i - 1]), tuple(smoothed_pts[i]), (0,),
                                         brush_thickness)
                elif item["type"] == "text":
                    # For text items, just draw them as is on both comparison images
                    text_content = item["content"]
                    position = item["position"]
                    cv2.putText(original_img, text_content, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,), 2, cv2.LINE_AA)
                    cv2.putText(smoothed_img, text_content, position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0,), 2, cv2.LINE_AA)

            # Display the original and smoothed images side-by-side using Matplotlib
            plt.figure(figsize=(12, 6))
            plt.subplot(1, 2, 1)
            plt.imshow(original_img, cmap='gray')
            plt.title("Original Drawing")
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(smoothed_img, cmap='gray')
            plt.title(f"Smoothed Drawing (window={smooth_factor})")
            plt.axis('off')
            plt.show()

        elif key == ord('q'):  # Quit the application
            break

    cv2.destroyAllWindows()  # Close all OpenCV windows on exit