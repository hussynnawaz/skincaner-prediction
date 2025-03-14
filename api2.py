import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Load model
import tensorflow as tf
from keras.saving import register_keras_serializable

@register_keras_serializable()
def dice_coefficient(y_true, y_pred, smooth=1e-6):
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# Load model with custom object
model = tf.keras.models.load_model("skin-cancer-segmentation.keras", custom_objects={"dice_coefficient": dice_coefficient})


# Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
MASK_FOLDER = "HAM1000_Segmentation_Dataset/Masks/Predicted"
CSV_FILE = "lesion_coverage.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)

# Load lesion coverage CSV
lesion_data = pd.read_csv(CSV_FILE)

def generate_mask(image_path):
    """Generates a segmentation mask for the input image."""
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (256, 256))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)

    mask = model.predict(img)
    mask = mask.squeeze() if mask.ndim > 2 else mask
    mask = (mask * 255).astype(np.uint8)

    mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path))
    cv2.imwrite(mask_path, mask)

    return mask_path, mask

def calculate_lesion_coverage(mask):
    """Calculates lesion coverage percentage with 4 decimal places."""
    if mask is None:
        return None
    total_pixels = mask.size
    lesion_pixels = np.count_nonzero(mask)
    return round(lesion_pixels / total_pixels, 4)  # 4 decimal places

def find_nearest_class(lesion_coverage):
    """Finds the closest matching class from CSV."""
    if lesion_coverage is None:
        return "Unknown"
    lesion_data["Diff"] = abs(lesion_data["Lesion Coverage"] - lesion_coverage)
    best_match = lesion_data.nsmallest(1, "Diff")
    return best_match["Class Label"].values[0]

@app.route("/predict", methods=["POST"])
def predict():
    """Handles the prediction API request."""
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)
    ext = filename.rsplit(".", 1)[-1].lower()

    if ext not in {"jpg", "jpeg", "png"}:
        return jsonify({"error": "Invalid image format. Only JPG, JPEG, and PNG allowed"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    mask_path, mask = generate_mask(filepath)
    if mask is None:
        return jsonify({"error": "Failed to process image"}), 500

    lesion_coverage = calculate_lesion_coverage(mask)
    predicted_class = find_nearest_class(lesion_coverage)

    return jsonify({
        "Lesion Coverage": lesion_coverage,  # 4 decimal places
        "Predicted Class": predicted_class,
        "Mask Image Path": mask_path  # Direct file path instead of Base64
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
