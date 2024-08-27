# A Bit About The Project
Our project focuses on developing a post-processing tool to translate recorded videos of simplified Israeli Sign Language (ISL) into written words. This addresses significant communication challenges faced by non-verbal individuals in Israel, such as those on life support that temporarily can’t speak and therefore are taught simplified ISL through organizations like Ezer Mizion (עזר מציון).
Miscommunication with their families and healthcare providers often lead to inaccurate medical assessments and insufficient support. Currently, no tool exists for translating simplified ISL signs into written words, and this is where we come into part to help.
## Overview of the Reposity
The Git directory structure is organized to facilitate the tasks needed for the project. 
- The “data” directory contains subdirectories such as “poses” for pose data and “videos” for training videos and metadata.
- The “static” directory houses static assets like images for the website, while the “templates” directory includes HTML templates for various web pages such as debugging, results, and uploads.
- The “uploads” directory holds uploaded video files and their corresponding pose data. 

The root directory includes essential scripts and models, such as “app.py” (main application file), “knnModel.py” (methods for updating the KNN model), “ourModel.py” (functions for the neural network model), and serialized model files (“.joblib”, “.tflite”). 
The entire codebase is accessible through a GitHub repository, and the website can be accessed via a provided URL. The user interface is designed to be intuitive, allowing users to upload videos for translation and provide feedback with ease.
