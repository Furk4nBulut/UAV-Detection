import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force TensorFlow to use CPU
from flask import Flask, request, render_template, send_file
from ultralytics import YOLO
from PIL import Image
import io
import uuid
import time
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input as resnet_preprocess

app = Flask(__name__)

# Define upload and result folders
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load YOLO model
YOLO_MODEL_PATH = "/home/furkanblt/PycharmProjects/UAV-Detection/Web/models/best.pt"
yolo_model = YOLO(YOLO_MODEL_PATH)

# Load CNN model
CNN_MODEL_PATH = "/home/furkanblt/PycharmProjects/UAV-Detection/Web/models/best_cnn_two_class_model.h5"
CLASS_NAMES = ['Non-UAV', 'UAV']
try:
    cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
    print("CNN Model Summary:")
    cnn_model.summary()
except Exception as e:
    print(f"Error loading CNN model: {e}")
    raise

# Load EfficientNet model
EFFICIENTNET_MODEL_PATH = "/home/furkanblt/PycharmProjects/UAV-Detection/Web/models/efficientnet_model.h5"
try:
    efficientnet_model = tf.keras.models.load_model(EFFICIENTNET_MODEL_PATH)
    print("EfficientNet Model Summary:")
    efficientnet_model.summary()
except Exception as e:
    print(f"Error loading EfficientNet model: {e}")
    efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False)
    print("Loaded pretrained EfficientNetB0")

# Load ResNet50 model
RESNET_MODEL_PATH = "/home/furkanblt/PycharmProjects/UAV-Detection/Web/models/resnet50_model.h5"
try:
    resnet_model = tf.keras.models.load_model(RESNET_MODEL_PATH)
    print("ResNet50 Model Summary:")
    resnet_model.summary()
except Exception as e:
    print(f"Error loading ResNet50 model: {e}")
    resnet_model = ResNet50(weights='imagenet', include_top=False)
    print("Loaded pretrained ResNet50")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files or 'model' not in request.form:
        return "No image or model selected", 400

    file = request.files['image']
    model_choice = request.form['model']

    if file.filename == '':
        return "No image selected", 400

    # Save the uploaded image
    image_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(image_path)

    if model_choice == 'yolo':
        return process_yolo(image_path, file.filename)
    elif model_choice == 'cnn':
        return process_cnn(image_path, file.filename)
    elif model_choice == 'efficientnet':
        return process_efficientnet(image_path, file.filename)
    elif model_choice == 'resnet':
        return process_resnet(image_path, file.filename)
    else:
        return "Invalid model selection", 400

def process_yolo(image_path, original_filename):
    start_time = time.time()
    results = yolo_model(image_path, conf=0.25, iou=0.45, stream=True)
    processing_time = time.time() - start_time

    result_image_name = f"result_{uuid.uuid4().hex}.jpg"
    result_image_path = os.path.join(RESULT_FOLDER, result_image_name)

    detections = []
    uav_count = 0
    total_confidence = 0.0

    for result in results:
        result.save(filename=result_image_path)
        boxes = result.boxes
        uav_count = len(boxes)
        for box in boxes:
            class_name = result.names[int(box.cls)]
            confidence = float(box.conf) * 100
            total_confidence += confidence
            detections.append({
                'class': class_name,
                'confidence': f"{confidence:.1f}"
            })

    avg_confidence = (total_confidence / uav_count) if uav_count > 0 else 0.0
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    return render_template(
        'result.html',
        model='YOLO',
        result_image=f'results/{result_image_name}',
        uav_count=uav_count,
        avg_confidence=f"{avg_confidence:.1f}",
        processing_time=f"{processing_time:.2f}",
        detections=detections,
        timestamp=timestamp
    )

def process_cnn(image_path, original_filename):
    try:
        start_time = time.time()
        image = Image.open(image_path).convert('RGB')
        image = image.resize((64, 64))
        img_array = img_to_array(image)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Input shape to CNN: {img_array.shape}")

        prediction = cnn_model.predict(img_array)
        confidence = prediction[0][0] * 100
        predicted_class = 1 if confidence >= 40 else 0
        processing_time = time.time() - start_time

        result_image_name = f"cnn_result_{uuid.uuid4().hex}.jpg"
        result_image_path = os.path.join(RESULT_FOLDER, result_image_name)
        Image.open(image_path).save(result_image_path)

        detection = {
            'label': CLASS_NAMES[predicted_class],
            'confidence': f"{confidence:.1f}" if predicted_class == 1 else f"{100 - confidence:.1f}"
        }
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        return render_template(
            'result.html',
            model='CNN',
            result_image=f'results/{result_image_name}',
            uav_count=1 if detection['label'] == 'UAV' else 0,
            avg_confidence=detection['confidence'],
            processing_time=f"{processing_time:.2f}",
            detections=[detection],
            timestamp=timestamp
        )
    except Exception as e:
        return f"CNN prediction failed: {str(e)}", 500

def process_efficientnet(image_path, original_filename):
    try:
        start_time = time.time()
        image = Image.open(image_path).convert('RGB')
        image = image.resize((64, 64))
        img_array = img_to_array(image)
        img_array = efficientnet_preprocess(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Input shape to EfficientNet: {img_array.shape}")

        prediction = efficientnet_model.predict(img_array)
        confidence = prediction[0][0] * 100
        predicted_class = 1 if confidence >= 40 else 0
        processing_time = time.time() - start_time

        result_image_name = f"efficientnet_result_{uuid.uuid4().hex}.jpg"
        result_image_path = os.path.join(RESULT_FOLDER, result_image_name)
        Image.open(image_path).save(result_image_path)

        detection = {
            'label': CLASS_NAMES[predicted_class],
            'confidence': f"{confidence:.1f}" if predicted_class == 1 else f"{100 - confidence:.1f}"
        }
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        return render_template(
            'result.html',
            model='EfficientNet',
            result_image=f'results/{result_image_name}',
            uav_count=1 if detection['label'] == 'UAV' else 0,
            avg_confidence=detection['confidence'],
            processing_time=f"{processing_time:.2f}",
            detections=[detection],
            timestamp=timestamp
        )
    except Exception as e:
        return f"EfficientNet prediction failed: {str(e)}", 500

def process_resnet(image_path, original_filename):
    try:
        start_time = time.time()
        image = Image.open(image_path).convert('RGB')
        image = image.resize((64, 64))
        img_array = img_to_array(image)
        img_array = resnet_preprocess(img_array)
        img_array = np.expand_dims(img_array, axis=0)
        print(f"Input shape to ResNet: {img_array.shape}")

        prediction = resnet_model.predict(img_array)
        confidence = prediction[0][0] * 100
        predicted_class = 1 if confidence >= 40 else 0
        processing_time = time.time() - start_time

        result_image_name = f"resnet_result_{uuid.uuid4().hex}.jpg"
        result_image_path = os.path.join(RESULT_FOLDER, result_image_name)
        Image.open(image_path).save(result_image_path)

        detection = {
            'label': CLASS_NAMES[predicted_class],
            'confidence': f"{confidence:.1f}" if predicted_class == 1 else f"{100 - confidence:.1f}"
        }
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        return render_template(
            'result.html',
            model='ResNet50',
            result_image=f'results/{result_image_name}',
            uav_count=1 if detection['label'] == 'UAV' else 0,
            avg_confidence=detection['confidence'],
            processing_time=f"{processing_time:.2f}",
            detections=[detection],
            timestamp=timestamp
        )
    except Exception as e:
        return f"ResNet prediction failed: {str(e)}", 500

@app.route('/download/<format>')
def download_result(format):
    result_files = [f for f in os.listdir(RESULT_FOLDER) if f.endswith('.jpg')]
    if not result_files:
        return "No result image available", 404
    latest_result = max(result_files, key=lambda x: os.path.getctime(os.path.join(RESULT_FOLDER, x)))
    result_image_path = os.path.join(RESULT_FOLDER, latest_result)

    img = Image.open(result_image_path)
    img_io = io.BytesIO()

    if format.lower() == 'jpg':
        img = img.convert('RGB')
        img.save(img_io, 'JPEG', quality=95)
        mimetype = 'image/jpeg'
        filename = 'result.jpg'
    elif format.lower() == 'png':
        img.save(img_io, 'PNG')
        mimetype = 'image/png'
        filename = 'result.png'
    else:
        return "Unsupported format", 400

    img_io.seek(0)
    return send_file(img_io, mimetype=mimetype, as_attachment=True, download_name=filename)

@app.route('/download-data/<format>')
def download_data(format):
    if format.lower() != 'json':
        return "Unsupported format", 400

    result_files = [f for f in os.listdir(RESULT_FOLDER) if f.endswith('.jpg')]
    if not result_files:
        return "No detection data available", 404
    latest_result = max(result_files, key=lambda x: os.path.getctime(os.path.join(RESULT_FOLDER, x)))

    result_image_path = os.path.join(RESULT_FOLDER, latest_result)
    results = yolo_model(result_image_path, conf=0.25, iou=0.45, stream=True)

    detections = []
    for result in results:
        for box in result.boxes:
            detections.append({
                'class': result.names[int(box.cls)],
                'confidence': f"{float(box.conf) * 100:.1f}",
                'bbox': {
                    'xywh': box.xywh.tolist()[0],
                    'xyxy': box.xyxy.tolist()[0]
                }
            })

    data = {
        'image': latest_result,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'detections': detections
    }
    json_io = io.BytesIO(json.dumps(data, indent=2).encode('utf-8'))

    return send_file(json_io, mimetype='application/json', as_attachment=True, download_name='detections.json')

if __name__ == '__main__':
    app.run(debug=True)