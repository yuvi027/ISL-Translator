import os
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
import joblib
import tensorflow as tf
import numpy as np
from pose_format import Pose
from sign_language_recognition.kaggle_asl_signs import predict
import subprocess

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # Required for flashing messages

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the KNN model
knn_model = joblib.load('knn_model.joblib')

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_pose_and_elan(video_file, label):
    pose_file = os.path.join(app.config['UPLOAD_FOLDER'], f"{label}.pose")
    try:
        subprocess.run(
            ['video_to_pose', '--format', 'mediapipe', '-i', video_file, '-o', pose_file], 
            check=True, 
            capture_output=True, 
            text=True
        )
        return pose_file
    except subprocess.CalledProcessError as e:
        print(f"Error in video_to_pose: {e.stdout}\n{e.stderr}")
        raise RuntimeError(f"Failed to extract pose: {e}")

import numpy as np

def pad_vector(vector, target_size):
    if vector.shape[0] < target_size:
        padding = np.zeros(target_size - vector.shape[0])
        return np.concatenate([vector, padding])
    else:
        return vector[:target_size]

import numpy as np
from scipy.spatial.distance import euclidean
import pandas as pd

import numpy as np
import pandas as pd
import joblib

import numpy as np
import joblib

import numpy as np
import joblib

# Load the new Random Forest model and scaler
# rf_model = joblib.load('rf_model.joblib')
# scaler = joblib.load('minmax_scaler.joblib')
knn = joblib.load('current_knn_model.joblib')

import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the KNN model
knn = joblib.load('current_knn_model.joblib')

# Create a scaler (you might need to load a pre-fitted scaler if you used one during training)
scaler = StandardScaler()

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

        print(f"Original vector shape: {vector.shape}")

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        expected_shape = input_details[0]['shape']
        print(f"Expected input shape: {expected_shape}")

        # Pad the vector to 543 elements
        padded_vector = np.pad(vector, (0, 543 - vector.shape[0]), mode='constant')

        # Reshape the vector to match the expected input shape (1, 543, 3)
        reshaped_vector = padded_vector.reshape(1, 543, 1)
        reshaped_vector = np.repeat(reshaped_vector, 3, axis=2)

        print(f"Reshaped vector shape: {reshaped_vector.shape}")

        interpreter.set_tensor(input_details[0]['index'], reshaped_vector.astype(np.float32))
        interpreter.invoke()
        output_vector = interpreter.get_tensor(output_details[0]['index'])

        # Ensure the output vector is 1D and has 250 elements
        output_vector = output_vector.flatten()[:250]
        
        print(f"Processed output vector shape: {output_vector.shape}")

        # Reshape the output vector to 2D array with one sample
        output_vector_2d = output_vector.reshape(1, -1)

        # Scale the features
        output_vector_2d_scaled = scaler.fit_transform(output_vector_2d)

        # Print KNN model information
        print(f"KNN model n_neighbors: {knn.n_neighbors}")
        print(f"KNN model metric: {knn.metric}")
        print(f"KNN model n_samples_fit: {knn.n_samples_fit_}")
        print(f"KNN model n_features_in: {knn.n_features_in_}")

        # Get predictions and probabilities
        predicted_label = knn.predict(output_vector_2d_scaled)[0]
        probabilities = knn.predict_proba(output_vector_2d_scaled)[0]

        print(f"Predicted label: {predicted_label}")
        print(f"Prediction probabilities: {probabilities}")
        print(f"Unique classes: {knn.classes_}")

        return predicted_label

    except Exception as e:
        print(f"Error in predict_label: {str(e)}")
        print(f"Error type: {type(e)}")
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
        prediction = predict_label(filepath)
        return render_template('result.html', filename=filename, prediction=prediction)
    except Exception as e:
        flash(f"An error occurred: {str(e)}")
        return redirect(url_for('upload_file'))
    

@app.route('/debug')
def debug_info():
    return render_template('debug.html', 
                           input_details=interpreter.get_input_details(),
                           output_details=interpreter.get_output_details())

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)