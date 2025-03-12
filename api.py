from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from PIL import Image

app = Flask(__name__)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="skin-cancer-segmentation.keras")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define class labels
CLASS_NAMES = ["Actinic Keratoses", "Basal Cell Carcinoma", "Benign Keratosis", 
               "Dermatofibroma", "Melanoma", "Nevus", "Vascular Lesions"]

# Define image preprocessing function
def preprocess_image(image):
    image = image.resize((224, 224))  # Resize to model input shape
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    image = Image.open(file).convert("RGB")  # Ensure RGB format

    # Preprocess image
    processed_img = preprocess_image(image)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_img)
    interpreter.invoke()  # Run inference

    # Get output tensor
    preds = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(preds)
    predicted_label = CLASS_NAMES[predicted_class]

    return jsonify({
        "predicted_class": int(predicted_class),
        "predicted_label": predicted_label,
        "probabilities": preds.tolist()
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
