import cv2
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model = load_model('models\asl_alphabet_model.h5')

# Define classes for alphabet (adjust according to your dataset)
class_names = ['A', 'B', 'C', 'D', ...]

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32))
    normalized = resized / 255.0
    input_data = np.expand_dims(normalized, axis=-1)
    input_data = np.expand_dims(input_data, axis=0)

    # Predict gesture
    prediction = model.predict(input_data)
    predicted_class = np.argmax(prediction)

    # Display result
    cv2.putText(frame, class_names[predicted_class], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
