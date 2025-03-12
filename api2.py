import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import base64
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

# Load model
model = tf.keras.models.load_model("model_densenet.h5")

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
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  
    img = np.expand_dims(img, axis=0)

    mask = model.predict(img)
    mask = mask.squeeze() if mask.ndim > 2 else mask
    mask = (mask * 255).astype(np.uint8)

    mask_path = os.path.join(MASK_FOLDER, os.path.basename(image_path))
    cv2.imwrite(mask_path, mask)

    return mask_path, mask

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def calculate_lesion_coverage(mask):
    if mask is None:
        return None
    total_pixels = mask.size
    lesion_pixels = np.count_nonzero(mask)
    return round(lesion_pixels / total_pixels, 2)

def find_nearest_class(lesion_coverage):
    if lesion_coverage is None:
        return "Unknown"
    lesion_data["Diff"] = abs(lesion_data["Lesion Coverage"] - lesion_coverage)
    best_match = lesion_data.nsmallest(1, "Diff")
    return best_match["Class Label"].values[0]

@app.route("/predict", methods=["POST"])
def predict():
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

    base64_mask = image_to_base64(mask_path) if os.path.exists(mask_path) else None

    return jsonify({
        "Lesion Coverage": lesion_coverage,
        "Predicted Class": predicted_class,
        "Mask Image (Base64)": base64_mask
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
