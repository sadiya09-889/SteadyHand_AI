# Smart Assistive AI - Handwriting Recognition

This project contains a real-time handwriting recognition application (`smartboard.py`) and a production-grade training pipeline to teach the AI to recognize handwritten characters.

## Why EMNIST over Synthetic Data?
Previously, this project generated synthetic data by drawing computer fonts and adding noise. This resulted in an artificially high 96% accuracy that didn't translate to real-world user handwriting.
The pipeline now uses the **EMNIST ByClass dataset** (Extended MNIST). It contains over 800,000 real handwritten characters, grouped into 62 classes (0-9, A-Z, a-z). Training on real human handwriting gives the model true robustness.

## Project Structure
- `src/`: Training, dataset loading, and evaluation code.
- `models/`: Where the trained `.keras` model and JSON label mapping are stored.
- `reports/`: Accuracy reports, confusion matrices, and loss plots generated during evaluation.
- `data/`: Extracted dataset cache (handled internally by the `emnist` pip package).
- `smartboard.py`: The interactive canvas application.

## How to Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## How to Train the Model
The model trains a Convolutional Neural Network (CNN) with Dropout and Batch Normalization.
```bash
# Train on the full dataset (can take a long time on CPU)
python src/train.py

# Train on a smaller subset for quick testing (e.g. 5000 samples)
python src/train.py --epochs 5 --subset 5000
```

## How to Evaluate the Model
Evaluation runs the model against a held-out test set, generating metrics in `reports/`.
```bash
python src/evaluate.py
```
> Note: A test accuracy of ~85-88% is standard for EMNIST ByClass due to many characters being identical regardless of casing (e.g., 'C' vs 'c', 'O' vs 'o').

## How to Run the Smartboard
```bash
python smartboard.py
```
**Usage**:
- Press `T` to switch to Text Enhancement Mode.
- Draw characters and wait for ~0.8s. The model will predict the character.
- If confidence is below 50%, it will output "uncertain" to avoid guessing.
- Press `D` to switch to Drawing Mode (saves raw strokes).
- Press `C` to clear the canvas.

## Improving the System
To further improve accuracy for specific users, you can use the smartboard to collect custom handwritten samples (saving the `canvas` arrays) and fine-tune the model with them.
"# SteadyHand_AI" 
