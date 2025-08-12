# HANDSIGHT: Real-Time Hand Gesture Recognition

HANDSIGHT is a real-time hand gesture recognition system that leverages **MediaPipe**, **OpenCV**, and a custom-trained neural network to detect and classify hand gestures. This project is designed for applications in human-computer interaction, accessibility, and more.

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
HANDSIGHT/
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
├── LICENSE                      # License file
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
git clone https://github.com/AyushRanaDev/HANDSIGHT-HandGestureRecognition.git
cd HANDSIGHT-HandGestureRecognition
```

### **2. Install Dependencies**
Install the required Python libraries:
```bash
pip install -r requirements.txt
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

---

## Screenshots

![Screenshot 2025-04-15 014345](https://github.com/user-attachments/assets/ea7b2770-b31f-4e5b-8868-0190459a7ff3)
![Screenshot 2025-04-15 014353](https://github.com/user-attachments/assets/3dbe814f-b6ee-4458-8999-c79b446fda35)
![Screenshot 2025-04-15 014405](https://github.com/user-attachments/assets/e4e7d7f5-2450-4bd1-a0fa-7a921c02b33b)
![Screenshot 2025-04-15 014458](https://github.com/user-attachments/assets/7857d4cf-015e-48d3-a239-15c01f216639)
![Screenshot 2025-04-15 014527](https://github.com/user-attachments/assets/bf1fc03e-2f13-40bb-8989-9c2fc0d6c37b)
![Screenshot 2025-04-15 014422](https://github.com/user-attachments/assets/989adad2-6073-4229-80d2-460cb2f090c3)

## Confusion Matrix
![Screenshot 2025-04-15 015418](https://github.com/user-attachments/assets/78134de4-7d86-4e3a-881c-8f4346bc3205)
![Screenshot 2025-04-15 015438](https://github.com/user-attachments/assets/21b96a6c-9890-4d31-8271-81b589f4b8a3)
![Screenshot 2025-04-15 015734](https://github.com/user-attachments/assets/f082d5bf-b3d7-473c-bdca-2e4cfb5adc85)


## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Commit your changes and push them to your fork.
4. Submit a pull request.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
