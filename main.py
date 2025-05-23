from flask import Flask, request, jsonify
import os
import requests
import base64
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.models import load_model

# Download model if not present
MODEL_PATH = "skin_diagnosis_model.h5"
MODEL_URL = "https://huggingface.co/lesiv561/skinvoice-model/resolve/main/skin_diagnosis_model.h5"
if not os.path.exists(MODEL_PATH):
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load model
model = load_model(MODEL_PATH)

app = Flask(__name__)

labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def decode_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).resize((224, 224))
    return np.expand_dims(np.array(image) / 255.0, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    image_data = data.get("image")
    image = decode_image(image_data)
    preds = model.predict(image)[0]
    return jsonify(dict(zip(labels, preds.tolist())))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=10000)
