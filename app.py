import os
import logging
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import joblib
import tensorflow as tf
from pose_format import Pose
from sign_language_recognition.kaggle_asl_signs import predict
import subprocess
from sklearn.neighbors import KNeighborsClassifier

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'some_secret_key'

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pose_and_elan(video_file, label):
    pose_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{label}.pose")
    try:
        # Generate pose file
        subprocess.run(
            ['video_to_pose', '--format', 'mediapipe', '-i', video_file, '-o', pose_file], 
            check=True, 
            capture_output=True, 
            text=True
        )
        return pose_file
    except subprocess.CalledProcessError as e:
        logger.error(f"Error in video_to_pose: {e.stdout}\n{e.stderr}")
        raise RuntimeError(f"Failed to extract pose: {e}")

def initialize_or_load_knn():
    try:
        knn = joblib.load('current_knn_model.joblib')
        logger.info("Loaded existing KNN model")
    except FileNotFoundError:
        logger.info("No existing KNN model found. Initializing new model.")
        knn = KNeighborsClassifier(n_neighbors=5)  # Adjust n_neighbors as needed
        # Initialize with dummy data
        X = np.array([[0] * 100])  # Adjust the dimension based on your vector size
        y = np.array(['dummy'])
        knn.fit(X, y)
        joblib.dump(knn, 'current_knn_model.joblib')
    return knn

knn = initialize_or_load_knn()

def predict_label(video_file):
    label = os.path.basename(video_file).split('.')[0]
    try:
        pose_file = extract_pose_and_elan(video_file, label)
        
        if not os.path.exists(pose_file):
            raise FileNotFoundError(f"Pose file not created: {pose_file}")

        with open(pose_file, "rb") as f:
            data_buffer = f.read()
        
        pose = Pose.read(data_buffer)
        vector = predict(pose)

        predicted_label = knn.predict(vector.reshape(1,-1))[0]

        return predicted_label, vector

    except Exception as e:
        logger.error(f"Error in predict_label: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            return redirect(url_for('translate', filename=filename))
    return render_template('upload.html')

@app.route('/translate/<filename>')
def translate(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    try:
        prediction, vector = predict_label(filepath)
        return render_template('result.html', filename=filename, prediction=prediction, vector=vector.tolist())
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('upload_file'))

@app.route('/update_dataset', methods=['POST'])
def update_dataset():
    data = request.json
    label = data['label']
    vector = np.array(data['vector'])

    logger.debug(f"Received data - Label: {label}, Vector shape: {vector.shape}")

    try:
        # Load the current dataset
        X = knn._fit_X
        y = knn._y

        logger.debug(f"Current dataset - X shape: {X.shape}, y shape: {y.shape}")

        # Add the new data point
        vector = vector.reshape(1, -1)  # Ensure vector is 2D
        X = np.vstack((X, vector))
        y = np.append(y, label)

        logger.debug(f"Updated dataset - X shape: {X.shape}, y shape: {y.shape}")

        # Retrain the model
        knn.fit(X, y)

        # Save the updated model
        joblib.dump(knn, 'current_knn_model.joblib')
        logger.info("Model updated and saved successfully")

        return jsonify({"status": "success", "message": "Dataset updated and model retrained"})
    except Exception as e:
        logger.error(f"Error updating dataset: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)