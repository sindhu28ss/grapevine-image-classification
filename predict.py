from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import io
from PIL import Image
import os

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
MODEL_PATH = os.path.join(os.getcwd(), "grapevine_xception_model.keras")
model = load_model(MODEL_PATH)

# Define image size & class labels
INPUT_SIZE = (299, 299)  # Xception input size
CLASS_LABELS = ['AK', 'Ala_Idris', 'Buzgulu', 'Dimnit', 'Nazli']  # Update class labels

@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']

    try:
        # Read the image correctly from the uploaded file
        img = Image.open(io.BytesIO(file.read()))
        img = img.resize(INPUT_SIZE)  # Resize image
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize

        # Make prediction
        predictions = model.predict(img_array)
        probabilities = tf.nn.softmax(predictions[0]).numpy()  # Apply softmax

        predicted_class_index = np.argmax(probabilities)  # Get highest probability index
        predicted_class = CLASS_LABELS[predicted_class_index]
        confidence = probabilities[predicted_class_index]  # Corrected confidence score

        return jsonify({
            'predicted_class': predicted_class,
            'confidence': float(confidence)  # Now between 0 and 1
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)
