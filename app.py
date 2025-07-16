import os
import platform
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if platform.system() != 'Windows':
    import fcntl
else:
    # Handle Windows-specific file control operations
    pass
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
app = Flask(__name__)
from flask import Flask

model = load_model('BrainTumor10Epochs.keras')
print('Model loaded. Check http://127.0.0.1:5000/')


def get_class_name(class_no):
    if class_no == 0:
        return "No Brain Tumor"
    elif class_no == 1:
        return "Yes Brain Tumor"


def get_result(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    predicted_class = np.argmax(result, axis=-1)
    return predicted_class


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = get_result(file_path)
        result = get_class_name(value[0])  # Extracting first prediction
        return result

    return "Prediction failed"


if __name__ == '__main__':
    app.run(debug=True)
