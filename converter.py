import tensorflow as tf

# Load the trained Keras model
model = tf.keras.models.load_model("resnet50_skin_cancer_model.keras")

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("model.tflite", "wb") as f:
    f.write(tflite_model)
