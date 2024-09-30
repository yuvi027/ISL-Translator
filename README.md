# A Bit About The Project
<img width="521" alt="Screenshot 2024-09-30 193408" src="https://github.com/user-attachments/assets/a321cec9-37e5-4dd2-84b5-ee2e8e7d5dd9">

Our project focuses on developing a post-processing tool to translate recorded videos of simplified Israeli Sign Language (ISL) into written words. This addresses significant communication challenges faced by non-verbal individuals in Israel, such as those on life support that temporarily can’t speak and therefore are taught simplified ISL through organizations like Ezer Mizion (עזר מציון).
Miscommunication with their families and healthcare providers often lead to inaccurate medical assessments and insufficient support. Currently, no tool exists for translating simplified ISL signs into written words, and this is where we come into part to help.
## 👥 The Team
### Team Members
* @yuvi027
* @shirsaadon

### Supervisor
* @eliahuhorwitz
  
## 📚 Project Description

### Overview of the Reposity
The Git directory structure is organized to facilitate the tasks needed for the project. 
- The “data” directory contains subdirectories such as “poses” for pose data and “videos” for training videos and metadata.
- The “static” directory houses static assets like images for the website, while the “templates” directory includes HTML templates for various web pages such as debugging, results, and uploads.
- The “uploads” directory holds uploaded video files and their corresponding pose data. 

The root directory includes essential scripts and models, such as “app.py” (main application file), “knnModel.py” (methods for updating the KNN model), “ourModel.py” (functions for the neural network model), and serialized model files (“.joblib”, “.tflite”). 
The entire codebase is accessible through a GitHub repository, and the website can be accessed via a provided URL. The user interface is designed to be intuitive, allowing users to upload videos for translation and provide feedback with ease.
### The main technologies used in the project:

ResNet neural network (specifically ResNet50)

EfficientNetB5 neural network

K-Nearest Neighbors (KNN) classifier

TensorFlow

OpenCV

Flask

Pandas

Joblib

Scikit-learn (sklearn)

HTML and CSS (for web interface)

Google Colab (for initial development)

Git (for version control)

Mediapipe (mentioned in related work, possibly used for keypoint extraction)

Python (implied as the main programming language)


The project leveraged these technologies to create a pipeline for translating simplified Israeli Sign Language (ISL) videos into written words, including preprocessing, keypoint extraction, embedding extraction, classification, and a web interface for user interaction.

## 🧱 Prerequisites
All requirements are listed in the requirements.txt file

## Some Useful Links
First Collab (Tracking): https://colab.research.google.com/drive/1jcPzlofFFmNw_83Ff9pZwkeE6b0SlRb3?ouid=104290483502975503034&usp=drive_link 

Second Collab (Pipeline): https://colab.research.google.com/drive/1hOjBiOLSbb_qAnK5wlvaJYaBHaCJRAfG?usp=sharing 
