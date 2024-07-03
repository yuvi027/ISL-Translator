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
        print(f"Vector type: {vector.dtype}")

        # Use the TFLite model to get the vector
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print(f"Input details: {input_details}")
        print(f"Output details: {output_details}")

        expected_shape = input_details[0]['shape']
        print(f"Expected input shape: {expected_shape}")

        # Reshape or pad the vector to match the expected input shape
        if expected_shape[1] > vector.shape[0]:
            # If the expected shape is larger, pad the vector
            pad_width = expected_shape[1] - vector.shape[0]
            padded_vector = np.pad(vector, (0, pad_width), mode='constant')
            reshaped_vector = padded_vector.reshape(expected_shape)
        elif expected_shape[1] < vector.shape[0]:
            # If the expected shape is smaller, truncate the vector
            reshaped_vector = vector[:expected_shape[1]].reshape(expected_shape)
        else:
            reshaped_vector = vector.reshape(expected_shape)

        print(f"Reshaped vector shape: {reshaped_vector.shape}")

        interpreter.set_tensor(input_details[0]['index'], reshaped_vector.astype(np.float32))
        interpreter.invoke()
        output_vector = interpreter.get_tensor(output_details[0]['index'])[0]

        print(f"Output vector shape: {output_vector.shape}")

        # Predict using KNN model
        prediction = knn_model.predict([output_vector])
        return prediction[0]
    except Exception as e:
        print(f"Error in predict_label: {str(e)}")
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