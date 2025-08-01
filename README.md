# Sign-Language-Interpretation
Real-time ASL (American Sign Language) alphabet recognition using MediaPipe hand landmarks and an LSTM deep learning model.

Features
Capture hand gesture images (A–Z) via webcam

Extract hand keypoints using MediaPipe

Train an LSTM model for sequence classification

Live prediction and display of detected alphabets

Requirements
Python 3.10+

OpenCV

MediaPipe

TensorFlow / Keras

NumPy, scikit-learn


Extract keypoints:
python data.py

Train the model:
python trainmodel.py

Run real-time prediction:
python app.py

Project Structure
collectdata.py — Capture gesture images

function.py — MediaPipe helpers & keypoint extraction

data.py — Prepare keypoint datasets

trainmodel.py — Train LSTM model

app.py — Live prediction app

Image/ — Gesture images dataset

MP_Data/ — Extracted keypoints data

model.json & model.h5 — Saved trained model

Dataset
Includes images and extracted keypoints for alphabets A-Z.
If dataset size is large, download link:  https://drive.google.com/drive/folders/1C2lD67kPQn6sRC_DApfkfgkPZXiUxuL7?usp=sharing
