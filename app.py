import os
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import joblib
import tensorflow as tf
from pose_format import Pose
from sign_language_recognition.kaggle_asl_signs import predict
import subprocess
from IPython.display import Video

app = Flask(__name__)
app.secret_key = 'some_secret_key'  # Required for flashing messages

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'mp4'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the TFLite model FOR DEBUGGING
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

# Load the KNN model
knn = joblib.load('current_knn_model.joblib')

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

        # Print KNN model information FOR DEBUGGING
        print(f"KNN model n_neighbors: {knn.n_neighbors}")
        print(f"KNN model metric: {knn.metric}")
        print(f"KNN model n_samples_fit: {knn.n_samples_fit_}")
        print(f"KNN model n_features_in: {knn.n_features_in_}")

        # Get predictions and probabilities
        predicted_label = knn.predict(vector.reshape(1,-1))[0]
        probabilities = knn.predict_proba(vector.reshape(1,-1))[0]

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

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)