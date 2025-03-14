import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from keras.saving import register_keras_serializable

CSV_FILE = "lesion_coverage.csv"
TEST_IMAGE = "sample_image.png"  # Replace with an actual image path

# Register custom Dice Coefficient function
@register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

# Load dataset
if os.path.exists(CSV_FILE):
    lesion_data = pd.read_csv(CSV_FILE)
    print("Dataset Overview:")
    print(lesion_data.head())
    print("\nLesion Coverage Stats:")
    print(lesion_data["Lesion Coverage"].describe())
    print("\nUnique Class Labels:", lesion_data["Class Label"].unique())
else:
    print(f"Error: {CSV_FILE} not found.")
    exit()

# Load model with custom function
try:
    model = tf.keras.models.load_model("skin-cancer-segmentation.keras", custom_objects={"dice_coefficient": dice_coefficient})
except Exception as e:
    print("Error loading model:", e)
    exit()

def preprocess_image(image_path):
    """Prepares an image for model input with correct dimensions."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"Error: Image '{image_path}' not found.")
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))  # Change from 224x224 to 256x256
    img = img / 255.0
    return np.expand_dims(img, axis=0)


# Test model prediction
img = preprocess_image(TEST_IMAGE)
if img is not None:
    mask = model.predict(img)
    mask = (mask.squeeze() * 255).astype(np.uint8)
    
    print("\nMask Shape:", mask.shape)
    print("Unique Mask Values:", np.unique(mask))

    lesion_coverage = np.count_nonzero(mask) / mask.size
    print("Calculated Lesion Coverage:", round(lesion_coverage, 4))

    # Find nearest class
    lesion_data["Diff"] = abs(lesion_data["Lesion Coverage"] - lesion_coverage)
    best_match = lesion_data.nsmallest(1, "Diff")
    predicted_class = best_match["Class Label"].values[0] if not best_match.empty else "Unknown"
    print("Predicted Class:", predicted_class)
