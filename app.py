from flask import Flask, request, jsonify
import os
from werkzeug.utils import secure_filename
from ourModel import predict_label  # Import your function from ourModel.py

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'result': 'No video part in the request'}), 400
    file = request.files['video']
    if file.filename == '':
        return jsonify({'result': 'No selected video'}), 400
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    result = predict_label(file_path)  # Call the imported function
    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
