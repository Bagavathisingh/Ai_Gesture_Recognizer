# Ai_Gesture_Recognizer: Real-Time Hand Gesture Recognition

Ai_Gesture_Recognizer is a real-time hand gesture recognition system that leverages **MediaPipe**, **OpenCV**, and a custom-trained neural network to detect and classify hand gestures. This project is designed for applications in human-computer interaction, accessibility, and more.

---

## Features
- **Real-Time Gesture Recognition**: Detects and classifies hand gestures in real-time using a webcam.
- **Custom Gesture Classifier**: Trained on hand keypoints to recognize specific gestures.
- **MediaPipe Integration**: Utilizes MediaPipe for efficient hand landmark detection.
- **TensorFlow Lite Deployment**: Optimized for real-time performance with TensorFlow Lite.
- **Logging and Visualization**: Logs keypoints and gesture history for analysis and debugging.

---

## Technologies Used
- **Python**: Core programming language.
- **MediaPipe**: For hand landmark detection.
- **OpenCV**: For video processing and visualization.
- **TensorFlow/Keras**: For building and training the gesture classifier.
- **NumPy**: For numerical operations.
- **Seaborn**: For visualizing the confusion matrix.

---

## Folder Structure
```
Ai_Gesture_Recognizer/
├── main_keypoint_app.py          # Main application for real-time gesture recognition
├── train_keypoint_classifier.ipynb # Jupyter Notebook for training the gesture classifier
├── app/
│   └── cnn_gesture_predictor.py # CNN-based gesture prediction (future work)
├── model/
│   ├── keypoint_classifier/     # Contains the trained model and labels
│   │   ├── keypoint_classifier_label.csv
│   │   ├── keypoint_classifier.h5
│   │   ├── keypoint_classifier.tflite
│   │   ├── keypoint.csv
│   │   └── keypoint_classifier.py
├── utils/
│   ├── cvfpscalc.py             # Utility for FPS calculation
└── README.md                    # Project documentation (this file)
```

---

## Main Components

### **1. main_keypoint_app.py**
This script is the core of the HANDSIGHT application. It performs real-time hand gesture recognition using MediaPipe and a custom-trained classifier. Key functionalities include:
- **Hand Detection**: Uses MediaPipe to detect hand landmarks.
- **Gesture Classification**: Classifies gestures using a TensorFlow model.
- **Visualization**: Displays bounding boxes, landmarks, and gesture information on the video feed.
- **Logging**: Logs keypoints and gesture history for training and debugging.

#### **Key Features**:
- **Modes**:
  - Default Mode: No logging.
  - Keypoint Logging Mode: Logs hand landmarks to `keypoint.csv`.
  - Point History Logging Mode: Logs finger point history to `point_history.csv`.
- **Drawing Utilities**: Visualizes landmarks, bounding boxes, and gesture information.

### **2. train_keypoint_classifier.ipynb**
This Jupyter Notebook is used to train the hand gesture classifier. It processes a dataset of hand keypoints, builds a neural network model, and converts it into a TensorFlow Lite model for deployment.

#### **Key Steps**:
1. **Data Loading**: Reads and preprocesses the dataset (`keypoint.csv`).
2. **Model Training**: Builds and trains a neural network using TensorFlow.
3. **Evaluation**: Evaluates the model's performance using accuracy and a confusion matrix.
4. **Deployment**: Converts the model to TensorFlow Lite for real-time inference.

---

## Setup Instructions

### **1. Clone the Repository**
```bash
git clone https://github.com/Bagavathisingh/Ai_Gesture_Recognizer.git
cd Ai_Gesture_Recognizer
```

### **3. Run the Application**
Run the real-time gesture recognition application:
```bash
python main_keypoint_app.py --device 0 --width 1280 --height 720
```

### **4. Train the Classifier**
Open the Jupyter Notebook and follow the steps to train the classifier:
```bash
jupyter notebook train_keypoint_classifier.ipynb
```

---

## Usage
- **Real-Time Recognition**: Use the application to detect and classify gestures in real-time.
- **Training**: Train the classifier on new gestures using the provided notebook.
- **Logging**: Log keypoints and gesture history for analysis or retraining.
