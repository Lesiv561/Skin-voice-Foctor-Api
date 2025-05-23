from flask import Flask, request, jsonify
import numpy as np
from PIL import Image
import io
import base64
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model (make sure you have the .h5 model uploaded in the repo or path)
model = load_model("model.h5")
label_map = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

def decode_image(base64_str):
    image_data = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_data)).resize((224, 224))
    return np.array(image) / 255.0

@app.route("/diagnose", methods=["POST"])
def diagnose():
    data = request.get_json()
    image = decode_image(data["image"])
    image = np.expand_dims(image, axis=0)
    prediction = model.predict(image)[0]
    result = label_map[np.argmax(prediction)]
    return jsonify({"diagnosis": result})

if __name__ == "__main__":
    app.run(debug=True)
