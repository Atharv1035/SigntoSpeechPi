import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import joblib
from collections import deque

# ==== Load Scaler & Label Encoder ====
scaler = joblib.load("../models/scaler0.8953.pkl")
label_encoder = joblib.load("../models/label_encoder0.8953.pkl")

# ==== Load TFLite Model ====
interpreter = tf.lite.Interpreter(model_path="../models/255_labels0.8953.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== Setup MediaPipe Holistic ====
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# ==== Landmark Extraction ====
def extract_landmarks(results):
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 4))
    face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]) if results.face_landmarks else np.zeros((468, 3))
    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]) if results.left_hand_landmarks else np.zeros((21, 3))
    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]) if results.right_hand_landmarks else np.zeros((21, 3))
    return np.concatenate([pose.flatten(), face.flatten(), lh.flatten(), rh.flatten()])

# ==== Setup Sequence Buffer ====
sequence = deque(maxlen=30)
predictions = deque(maxlen=20)

# ==== Initialize Webcam ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

# ==== MediaPipe Inference Loop ====
with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=1
) as holistic:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break

        image = cv2.flip(frame, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)

        # Draw landmarks
        mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        try:
            landmarks = extract_landmarks(results)
            landmarks = scaler.transform([landmarks])
            sequence.append(landmarks[0])

            if len(sequence) == 30:
                input_data = np.expand_dims(sequence, axis=0).astype(np.float32)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output = interpreter.get_tensor(output_details[0]['index'])
                predicted_label = np.argmax(output)
                prediction = label_encoder.inverse_transform([predicted_label])[0]
                predictions.append(prediction)

                if np.max(output) > 0.8:
                    cv2.putText(image, f'{prediction}', (10, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        except Exception:
            pass

        # Show the frame with prediction
        cv2.imshow("Gesture Recognition", image)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
