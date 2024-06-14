import os
import subprocess
import json
import glob
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import joblib
from pose_format import Pose


# Install necessary packages
!pip install git+https://github.com/sign-language-processing/pose.git
!pip install mediapipe vidgear
!pip install git+https://github.com/sign-language-processing/segmentation
!pip install git+https://github.com/sign-language-processing/recognition
!pip install scikit-learn pandas joblib

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

def predict_label(new_video_file, model_name="kaggle_asl_signs", knn_model_path=model_path):
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

from sign_language_recognition.kaggle_asl_signs import predict
from pose_format import Pose

# Specify the directory in Google Drive containing your video files
video_directory = '/content/drive/My Drive/EngineeringProject/videos/train'

# List all video files in the directory
video_files = glob.glob(os.path.join(video_directory, '*1.mp4'))
# video_files = glob.glob("/content/drive/My Drive/EngineeringProject/videos/train/Can1.mp4")
# Ensure the lists are sorted
video_files.sort()


# Create an empty DataFrame to hold the vectors and labels
dataset = pd.DataFrame()

# Process each video, extract vectors, and add to the dataset
for video_file in video_files:
  # Example: Infer label from filename (assuming the format is 'label.mp4')
    label = os.path.basename(video_file).split('1')[0]
    print(label)

    pose_file = extract_pose_and_elan(video_file, label)

    print("pose_file name:",pose_file)
    # ! visualize_pose -i "$pose_file" -o example-skeleton.mp4
    # print("\n\nhi\n\n")
    # Video("example-skeleton.mp4", embed=True)
    data_buffer = open(pose_file, "rb").read()
    pose = Pose.read(data_buffer)

    vector = predict(pose)
    # Test the vector extraction

    vector_df = pd.DataFrame([vector])
    # print(vector_df)
    vector_df['label'] = label
    dataset = pd.concat([dataset, vector_df], ignore_index=True)

# Save the dataset to Google Drive
dataset_path = '/content/drive/My Drive/EngineeringProject/Dataset.csv'
dataset.to_csv(dataset_path, index=False)

#TODO: import as the model_path our knn model from drive
import os
import glob
import pandas as pd
# from sign_language_recognition.kaggle_asl_signs import predict
from pose_format import Pose

# Specify the directory in Google Drive containing your video files
video_directory = '/content/drive/My Drive/EngineeringProject/videos/test'

# List all video files in the directory
video_files = glob.glob(os.path.join(video_directory, '*2.mp4'))
# Ensure the lists are sorted
video_files.sort()

# Create an empty DataFrame to hold the vectors and labels
eval = pd.DataFrame(columns=['Label', 'Prediction', 'Match'])

# Helper function to normalize strings
def normalize_string(s):
    return ''.join(e for e in s.lower() if e.isalnum())

# Process each video, extract vectors, and add to the dataset
for video_file in video_files:
    # Example: Infer label from filename (assuming the format is 'label.mp4')
    label = os.path.basename(video_file).split('1')[0]
    print(f"Processing: {label}")
    prediction = predict_label(video_file)  # Assuming predict_label is a typo and should be predict

    # Normalize label and prediction for comparison
    # normalized_label = normalize_string(label)
    # normalized_prediction = normalize_string(prediction)

    # Determine if they match
    match = 1 if label == prediction else 0

    # Append to DataFrame
    eval = pd.concat([eval, pd.DataFrame({'Label': [label], 'Prediction': [prediction], 'Match': [match]})], ignore_index=True)

# Calculate accuracy and number of matches
num_matches = eval['Match'].sum()
accuracy = (num_matches / len(eval)) * 100 if len(eval) > 0 else 0

print(f"Accuracy: {accuracy}%")
print(f"Number of matches: {num_matches}")
print(f"Number of words: {len(eval)}")

# Add accuracy and number of matches as new rows in the DataFrame (optional)
eval = pd.concat([eval, pd.DataFrame({'Label': ['Accuracy'], 'Prediction': [f'{accuracy}%'], 'Match': [num_matches]})], ignore_index=True)

# Save the dataset to Google Drive
eval_path = '/content/drive/My Drive/EngineeringProject/KNN-evaluation.csv'
eval.to_csv(eval_path, index=False)

