**Age & Gender Recognition using OpenCV DNN**

This project uses OpenCV's deep learning module (DNN) to detect faces in video frames and predict the age and gender of individuals in real-time. The model utilizes pre-trained convolutional neural networks for accurate age and gender classification.

**Introduction**

The Age & Gender Recognition system is designed to detect faces and predict the age range and gender of individuals from video input. This project makes use of pre-trained models for face detection, age classification, and gender classification, enabling real-time predictions.

**Features**

1) Real-time face detection in video.
2) Age classification into predefined age groups.
3) Gender classification (Male/Female).
4) Supports both real-time video capture and video file input.

**Requirements**
1) Python 3.x
2) OpenCV
3) Pre-trained models for:
   
    -Face detection
   
    -Age prediction
   
    -Gender prediction

**Installation**
1) **Clone the repository:**

		git clone https://github.com/sanskarsri26/Age_gender_recognize.git

		cd Age_gender_recognize

2) **Install dependencies:**

Make sure you have OpenCV installed. You can install OpenCV using pip:

	pip install opencv-python

3) **Download the pre-trained models:**

Download the following model files and place them in your project directory:

  --opencv_face_detector.pbtxt and opencv_face_detector_uint8.pb (for face detection)

  --age_deploy.prototxt and age_net.caffemodel (for age prediction)

  --gender_deploy.prototxt and gender_net.caffemodel (for gender prediction)

** Usage**

1) Running the Project
   
	Run the Python script:

This script processes video input, detects faces, and predicts the age and gender of individuals.

	python age_gender_recognition.py
 
2) Input Video:

The script is set to read from a video file (4.mp4 in this case). You can change this to use your webcam or another video source.

To use the webcam, replace:

	video = cv2.VideoCapture('4.mp4')
 
with:

	video = cv2.VideoCapture(0)
 
**Exiting:**

Press q to quit the video processing window.

**Explanation**

Face Detection: Detects faces in each video frame using a pre-trained model.

Age Prediction: Predicts the age group of the detected faces based on the age_net.caffemodel.

Gender Prediction: Classifies gender as Male or Female using the gender_net.caffemodel.

The predicted age and gender are displayed as text labels over each detected face in the video feed.


**Models**

Ensure the following models are available in the root directory:

**Face Detection:**

opencv_face_detector.pbtxt

opencv_face_detector_uint8.pb

**Age Detection:**

age_deploy.prototxt

age_net.caffemodel

**Gender Detection:**

gender_deploy.prototxt

gender_net.caffemodel

The project uses OpenCV's DNN module to load these models and make predictions.

