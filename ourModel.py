import os
import subprocess
import json
import glob
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
from pose_format import Pose


# Install necessary packages
# pip install git+https://github.com/sign-language-processing/pose.git
# pip install pose-format
# pip install mediapipe vidgear
# pip install git+https://github.com/sign-language-processing/segmentation
# pip install git+https://github.com/sign-language-processing/recognition
# pip install scikit-learn pandas joblib
# pip install flask
# pip install tensorflow
# pip install tensorflow_hub
# pip install flask numpy tensorflow tensorflow-hub opencv-python-headless

from IPython.display import Video


def extract_pose_and_elan(
		video_file="/content/drive/My Drive/EngineeringProject/videos/train/Can1.mp4",
		label="Can"):
	# Define the output pose and ELAN filenames
	base_name = os.path.splitext(video_file)[0]

	pose_file = "/content/drive/My Drive/EngineeringProject/videos/test/extras/" + label + ".pose"

	# Extract the pose file
	subprocess.run(
		['video_to_pose', '--format', 'mediapipe', '-i', video_file, '-o',
		 pose_file], check=True)

	return pose_file
# interpreter = tf.lite.Interpreter(model_path="model.tflite")

def predict_label(new_video_file, model_name="kaggle_asl_signs", knn_model_path="/workspaces/ISL-Translator/knn_model.joblib"):
    # Extract pose and ELAN files from the new video
    new_pose_file = extract_pose_and_elan(new_video_file)

    # Extract vector from the new video
    # vector = extract_vector(new_pose_file, new_elan_file, model_name)
    data_buffer = open(new_pose_file, "rb").read()
    pose = Pose.read(data_buffer)
	# TODO: this predict is from the model we need to download the model files
    vector = predict(pose)

    # Load the trained KNN model
    knn = joblib.load(knn_model_path)

    # Predict the label for the vector
    prediction = knn.predict([vector])
    return prediction

import os
import glob
import pandas as pd
from sign_language_recognition.kaggle_asl_signs import predict
from pose_format import Pose

# Specify the directory in the project containing your video files
video_directory = os.path.join('data', 'videos', 'train')

# List all video files in the directory
video_files = glob.glob(os.path.join(video_directory, '*1.mp4'))
# Ensure the lists are sorted
video_files.sort()

# Create an empty DataFrame to hold the vectors and labels
dataset = pd.DataFrame()

# Function to extract pose (you'll need to implement this based on your specific requirements)
def extract_pose_and_elan(video_file, label):
    # Implement your pose extraction logic here
    # This should return the path to the extracted pose file
    pose_file = os.path.join('data', 'poses', f"{label}.pose")
    subprocess.run(['video_to_pose', '--format', 'mediapipe', '-i', video_file, '-o', pose_file], check=True)
    # Your pose extraction code here
    return pose_file

# Process each video, extract vectors, and add to the dataset
for video_file in video_files:
    # Example: Infer label from filename (assuming the format is 'label1.mp4')
    label = os.path.basename(video_file).split('1')[0]
    print(label)

    pose = extract_pose_and_elan(video_file, label)
    print("pose = ",pose)
    pose_file="/workspaces/ISL-Translator/"+pose
    print("pose_file = ",pose_file)


    print("pose_file name:", pose_file)

    with open(pose_file, "rb") as f:
        data_buffer = f.read()
    pose = Pose.read(data_buffer)

    vector = predict(pose)

    vector_df = pd.DataFrame([vector])
    vector_df['label'] = label
    dataset = pd.concat([dataset, vector_df], ignore_index=True)

# Save the dataset to a file in the project
dataset_path = os.path.join('data', 'Dataset1.csv')
dataset.to_csv(dataset_path, index=False)


# EVALUATION
# #TODO: import as the model_path our knn model from drive
# import os
# import glob
# import pandas as pd
# # from sign_language_recognition.kaggle_asl_signs import predict
# from pose_format import Pose

# # Specify the directory in Google Drive containing your video files
# video_directory = '/content/drive/My Drive/EngineeringProject/videos/test'

# # List all video files in the directory
# video_files = glob.glob(os.path.join(video_directory, '*2.mp4'))
# # Ensure the lists are sorted
# video_files.sort()

# # Create an empty DataFrame to hold the vectors and labels
# eval = pd.DataFrame(columns=['Label', 'Prediction', 'Match'])

# # Helper function to normalize strings
# def normalize_string(s):
#     return ''.join(e for e in s.lower() if e.isalnum())

# # Process each video, extract vectors, and add to the dataset
# for video_file in video_files:
#     # Example: Infer label from filename (assuming the format is 'label.mp4')
#     label = os.path.basename(video_file).split('1')[0]
#     print(f"Processing: {label}")
#     prediction = predict_label(video_file)  # Assuming predict_label is a typo and should be predict

#     # Normalize label and prediction for comparison
#     # normalized_label = normalize_string(label)
#     # normalized_prediction = normalize_string(prediction)

#     # Determine if they match
#     match = 1 if label == prediction else 0

#     # Append to DataFrame
#     eval = pd.concat([eval, pd.DataFrame({'Label': [label], 'Prediction': [prediction], 'Match': [match]})], ignore_index=True)

# # Calculate accuracy and number of matches
# num_matches = eval['Match'].sum()
# accuracy = (num_matches / len(eval)) * 100 if len(eval) > 0 else 0

# print(f"Accuracy: {accuracy}%")
# print(f"Number of matches: {num_matches}")
# print(f"Number of words: {len(eval)}")

# # Add accuracy and number of matches as new rows in the DataFrame (optional)
# eval = pd.concat([eval, pd.DataFrame({'Label': ['Accuracy'], 'Prediction': [f'{accuracy}%'], 'Match': [num_matches]})], ignore_index=True)

# # Save the dataset to Google Drive
# eval_path = '/content/drive/My Drive/EngineeringProject/KNN-evaluation.csv'
# eval.to_csv(eval_path, index=False)

