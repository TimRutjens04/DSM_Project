import pandas as pd
import numpy as np
from PIL import Image
import os
import time
import cloudpickle
import traceback

# =========================
# Configuration
# =========================
IMAGE_SIZE = (64, 64)
LABEL_FILE = "labels.csv"
TEST_DIR = "test_images"

# =========================
# Helpers
# =========================
def image_to_features(image: Image.Image) -> np.ndarray:
    """Convert image to feature vector (grayscale flattened)"""
    image = image.resize(IMAGE_SIZE)
    return np.array(image.convert("L")).flatten()

def load_test_data():
    """Load test images and labels"""
    if not os.path.exists(LABEL_FILE) or not os.path.exists(TEST_DIR):
        raise FileNotFoundError("Missing labels.csv or test_images/ folder.")
    
    df = pd.read_csv(LABEL_FILE)
    
    X_test, y_test = [], []
    
    for _, row in df.iterrows():
        img_path = os.path.join(TEST_DIR, row["filename"])
        try:
            img = Image.open(img_path)
            X_test.append(image_to_features(img))
            y_test.append(row["label"])
        except Exception as e:
            print(f"❌ Error loading {img_path}: {e}")
            failed_images += 1
    
    return np.array(X_test), np.array(y_test)

# =========================
# Evaluation Logic
# =========================
def evaluate_model(model_path: str):
    try:        
        # Load model
        with open(model_path, "rb") as f:
            model = cloudpickle.load(f)
        
        # Load test data
        X_test, y_test = load_test_data()
        
        # Make predictions
        start = time.time()
        y_pred = model.predict(X_test)
        elapsed = (time.time() - start) * 1000
        
        if len(y_pred) != len(y_test):
            print(f"[ERROR] : Model output length ({len(y_pred)}) does not match test set ({len(y_test)})")
            return False
        
        # Calculate metrics
        accuracy = (y_pred == y_test).mean()
        avg_time = elapsed / len(X_test)
        
        print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"Total time: {elapsed:.3f} ms")
        print(f"Avg time per image: {avg_time:.3f} ms")
        print(f"Correct predictions: {np.sum(y_pred == y_test)}/{len(y_test)}")
        
        # Class-wise accuracy if labels are strings
        unique_labels = np.unique(y_test)
        if len(unique_labels) > 1:
            print(f"\n📋 CLASS-WISE RESULTS")
            print("-" * 30)
            for label in unique_labels:
                mask = y_test == label
                class_accuracy = (y_pred[mask] == y_test[mask]).mean()
                class_count = np.sum(mask)
                print(f"{label}: {class_accuracy:.4f} ({class_count} samples)")
        
        return True
        
    except Exception as e:
        print(f"❌ Error evaluating model:")
        print(traceback.format_exc())
        return False

# =========================
# Main Script
# =========================
def main():
    print("Olive Fly Classifier Evaluator")
    print("=" * 40)
    
    # Check if required files exist
    missing_files = []
    if not os.path.exists(LABEL_FILE):
        missing_files.append(LABEL_FILE)
    if not os.path.exists(TEST_DIR):
        missing_files.append(TEST_DIR)
    
    if missing_files:
        print(f"Missing required files/folders: {', '.join(missing_files)}")
        return

    model_path = "olive_fly_model_1.cloudpkl"
    
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return
    
    # Evaluate model
    success = evaluate_model(model_path)
    
    if success:
        print(f"\nEvaluation complete!")
    else:
        print(f"\nEvaluation failed!")

if __name__ == "__main__":
    main()